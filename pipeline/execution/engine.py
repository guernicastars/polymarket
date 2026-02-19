"""Order execution engine wrapping py-clob-client.

Handles order creation, signing, submission, cancellation, and fill tracking
against the Polymarket CLOB API. All order operations go through this single
interface so risk checks and logging are centralized.

The engine operates in two modes:
- DRY_RUN: logs orders but doesn't submit them (for testing/paper trading)
- LIVE: submits signed orders to the CLOB

Every order is logged to ClickHouse for post-hoc analysis regardless of mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from pipeline.config import (
    CLOB_API_URL,
    EXECUTION_CHAIN_ID,
    EXECUTION_DRY_RUN,
    EXECUTION_SIGNATURE_TYPE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING = "pending"           # Created, not yet submitted
    SUBMITTED = "submitted"       # Sent to CLOB
    LIVE = "live"                 # Resting on order book
    MATCHED = "matched"           # Fully filled
    PARTIALLY_FILLED = "partial"  # Partially filled
    CANCELLED = "cancelled"       # Cancelled by us
    REJECTED = "rejected"         # Rejected by CLOB
    FAILED = "failed"             # Submission error
    DRY_RUN = "dry_run"           # Paper trade (not submitted)


class OrderRequest(BaseModel):
    """Request to place an order on the CLOB."""

    condition_id: str = Field(..., description="Market condition ID.")
    token_id: str = Field(..., description="Token ID (Yes or No outcome).")
    side: str = Field(..., description="BUY or SELL.")
    price: float = Field(..., ge=0.01, le=0.99, description="Limit price (0.01-0.99).")
    size: float = Field(..., gt=0, description="Size in outcome tokens.")
    signal_source: str = Field(default="ensemble", description="Which signal triggered this.")
    edge: float = Field(default=0.0, description="Model probability minus market price.")
    kelly_fraction: float = Field(default=0.0, description="Raw Kelly fraction.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Signal confidence.")
    tick_size: str = Field(default="0.01", description="Market tick size.")
    neg_risk: bool = Field(default=False, description="True for 3+ outcome markets.")


class OrderResult(BaseModel):
    """Result from order submission."""

    request: OrderRequest
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = ""
    error_msg: str = ""
    submitted_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    filled_size: Optional[float] = None
    transaction_hashes: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Centralized order execution against the Polymarket CLOB.

    All orders flow through this engine so that:
    - Risk checks are applied before submission
    - Every order is logged (ClickHouse + structured logs)
    - Dry-run mode can be toggled without changing upstream code
    - Heartbeat is maintained to keep orders alive

    Attributes:
        dry_run: If True, orders are logged but not submitted.
        _client: py-clob-client ClobClient instance (lazy-initialized).
        _order_log: In-memory log of all order results.
    """

    def __init__(
        self,
        private_key: str = "",
        funder_address: str = "",
        dry_run: bool = EXECUTION_DRY_RUN,
    ) -> None:
        self.dry_run = dry_run
        self._private_key = private_key
        self._funder_address = funder_address
        self._client: Any = None
        self._order_log: list[OrderResult] = []
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat_id: str = ""
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy-initialize the CLOB client and API credentials.

        Separated from __init__ so that the engine can be constructed
        without blocking and without requiring credentials in dry-run mode.
        """
        if self._initialized:
            return

        if self.dry_run:
            logger.info("execution_engine_init", extra={"mode": "DRY_RUN"})
            self._initialized = True
            return

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            self._client = ClobClient(
                host=CLOB_API_URL,
                key=self._private_key,
                chain_id=EXECUTION_CHAIN_ID,
                signature_type=EXECUTION_SIGNATURE_TYPE,
                funder=self._funder_address or None,
            )

            # Derive or create API credentials
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)

            self._initialized = True
            logger.info("execution_engine_init", extra={"mode": "LIVE"})

        except ImportError:
            logger.error(
                "py-clob-client not installed. Install with: "
                "pip install py-clob-client"
            )
            raise
        except Exception:
            logger.error("execution_engine_init_failed", exc_info=True)
            raise

    async def start_heartbeat(self) -> None:
        """Start the heartbeat loop to keep orders alive.

        Polymarket cancels all open orders if no heartbeat is received
        within 10 seconds. We send one every 5 seconds.
        """
        if self.dry_run or self._heartbeat_task is not None:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("heartbeat_started")

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
            logger.info("heartbeat_stopped")

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats every 5 seconds."""
        while True:
            try:
                if self._client is not None:
                    resp = await asyncio.to_thread(
                        self._client.post_heartbeat,
                        self._last_heartbeat_id,
                    )
                    if isinstance(resp, dict):
                        self._last_heartbeat_id = resp.get("heartbeat_id", "")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("heartbeat_error", exc_info=True)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a limit order on the CLOB.

        Args:
            request: Order parameters.

        Returns:
            OrderResult with status and fill info.
        """
        await self.initialize()

        result = OrderResult(request=request, submitted_at=datetime.now(timezone.utc))

        if self.dry_run:
            result.status = OrderStatus.DRY_RUN
            result.order_id = f"dry_{int(time.time() * 1000)}"
            logger.info(
                "order_dry_run",
                extra={
                    "condition_id": request.condition_id,
                    "side": request.side,
                    "price": request.price,
                    "size": request.size,
                    "edge": request.edge,
                    "signal": request.signal_source,
                },
            )
            self._order_log.append(result)
            return result

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            start = time.monotonic()

            order_args = OrderArgs(
                token_id=request.token_id,
                price=request.price,
                size=request.size,
                side=request.side,
            )

            signed_order = await asyncio.to_thread(
                self._client.create_order, order_args
            )

            resp = await asyncio.to_thread(
                self._client.post_order,
                signed_order,
                OrderType.GTC,
                options={
                    "tick_size": request.tick_size,
                    "neg_risk": request.neg_risk,
                },
            )

            result.latency_ms = (time.monotonic() - start) * 1000

            if isinstance(resp, dict):
                result.order_id = resp.get("orderID", "")
                result.error_msg = resp.get("errorMsg", "")
                result.transaction_hashes = resp.get("transactionsHashes", [])

                status_str = resp.get("status", "")
                if resp.get("success"):
                    result.status = self._map_status(status_str)
                else:
                    result.status = OrderStatus.REJECTED
            else:
                result.status = OrderStatus.FAILED
                result.error_msg = f"Unexpected response type: {type(resp)}"

            logger.info(
                "order_placed",
                extra={
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "condition_id": request.condition_id,
                    "side": request.side,
                    "price": request.price,
                    "size": request.size,
                    "edge": request.edge,
                    "latency_ms": result.latency_ms,
                },
            )

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_msg = str(e)
            logger.error(
                "order_failed",
                extra={
                    "condition_id": request.condition_id,
                    "error": str(e),
                },
                exc_info=True,
            )

        self._order_log.append(result)
        return result

    async def place_market_order(
        self,
        token_id: str,
        side: str,
        amount_usd: float,
        price: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
        signal_source: str = "ensemble",
    ) -> OrderResult:
        """Place a fill-or-kill market order.

        Args:
            token_id: Token to trade.
            side: BUY or SELL.
            amount_usd: Dollar amount to trade.
            price: Worst acceptable price.
            tick_size: Market tick size.
            neg_risk: True for 3+ outcome markets.
            signal_source: Which signal triggered this.

        Returns:
            OrderResult.
        """
        await self.initialize()

        request = OrderRequest(
            condition_id="",
            token_id=token_id,
            side=side,
            price=price,
            size=amount_usd / price if price > 0 else 0,
            signal_source=signal_source,
            tick_size=tick_size,
            neg_risk=neg_risk,
        )
        result = OrderResult(request=request, submitted_at=datetime.now(timezone.utc))

        if self.dry_run:
            result.status = OrderStatus.DRY_RUN
            result.order_id = f"dry_mkt_{int(time.time() * 1000)}"
            self._order_log.append(result)
            return result

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType

            start = time.monotonic()

            mo_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usd,
                side=side,
                order_type=OrderType.FOK,
            )

            signed = await asyncio.to_thread(
                self._client.create_market_order, mo_args
            )

            resp = await asyncio.to_thread(
                self._client.post_order,
                signed,
                OrderType.FOK,
                options={"tick_size": tick_size, "neg_risk": neg_risk},
            )

            result.latency_ms = (time.monotonic() - start) * 1000

            if isinstance(resp, dict):
                result.order_id = resp.get("orderID", "")
                result.error_msg = resp.get("errorMsg", "")
                result.status = (
                    OrderStatus.MATCHED if resp.get("success") else OrderStatus.REJECTED
                )
            else:
                result.status = OrderStatus.FAILED

            logger.info(
                "market_order_placed",
                extra={
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "side": side,
                    "amount_usd": amount_usd,
                    "latency_ms": result.latency_ms,
                },
            )

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_msg = str(e)
            logger.error("market_order_failed", exc_info=True)

        self._order_log.append(result)
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order.

        Args:
            order_id: CLOB order ID.

        Returns:
            True if cancellation succeeded.
        """
        if self.dry_run:
            logger.info("cancel_dry_run", extra={"order_id": order_id})
            return True

        try:
            resp = await asyncio.to_thread(self._client.cancel, order_id)
            success = isinstance(resp, dict) and resp.get("canceled", False)
            logger.info(
                "order_cancelled",
                extra={"order_id": order_id, "success": success},
            )
            return success
        except Exception:
            logger.error("cancel_failed", extra={"order_id": order_id}, exc_info=True)
            return False

    async def cancel_all(self) -> bool:
        """Cancel all open orders (emergency kill switch).

        Returns:
            True if cancellation succeeded.
        """
        if self.dry_run:
            logger.info("cancel_all_dry_run")
            return True

        try:
            resp = await asyncio.to_thread(self._client.cancel_all)
            logger.info("all_orders_cancelled", extra={"response": str(resp)})
            return True
        except Exception:
            logger.error("cancel_all_failed", exc_info=True)
            return False

    async def get_open_orders(self) -> list[dict]:
        """Fetch all open orders from the CLOB.

        Returns:
            List of order dicts from the CLOB API.
        """
        if self.dry_run:
            return []

        try:
            resp = await asyncio.to_thread(self._client.get_orders)
            if isinstance(resp, list):
                return resp
            return []
        except Exception:
            logger.error("get_orders_failed", exc_info=True)
            return []

    async def get_trades(self) -> list[dict]:
        """Fetch recent trades for the account.

        Returns:
            List of trade dicts.
        """
        if self.dry_run:
            return []

        try:
            resp = await asyncio.to_thread(self._client.get_trades)
            if isinstance(resp, list):
                return resp
            return []
        except Exception:
            logger.error("get_trades_failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_status(status_str: str) -> OrderStatus:
        """Map CLOB status string to OrderStatus enum."""
        mapping = {
            "live": OrderStatus.LIVE,
            "matched": OrderStatus.MATCHED,
            "delayed": OrderStatus.SUBMITTED,
            "unmatched": OrderStatus.SUBMITTED,
        }
        return mapping.get(status_str, OrderStatus.SUBMITTED)

    @property
    def order_log(self) -> list[OrderResult]:
        """Return the in-memory order log."""
        return list(self._order_log)

    @property
    def is_live(self) -> bool:
        """True if engine is in live trading mode."""
        return not self.dry_run and self._initialized

    async def close(self) -> None:
        """Shutdown: cancel all orders and stop heartbeat."""
        await self.stop_heartbeat()
        if self.is_live:
            await self.cancel_all()
        logger.info("execution_engine_closed")
