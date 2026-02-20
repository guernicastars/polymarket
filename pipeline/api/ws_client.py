"""WebSocket client for real-time Polymarket CLOB market data."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

import websockets
import websockets.exceptions

from pipeline.config import (
    WS_MAX_TOKENS_PER_CONN,
    WS_RECONNECT_BASE_DELAY,
    WS_RECONNECT_MAX_DELAY,
    WS_URL,
)
from pipeline.jobs.market_sync import token_mapping

logger = logging.getLogger(__name__)

# Type alias for the callback used to push data into the writer.
Callback = Callable[[str, list[list[Any]]], Coroutine[Any, Any, None]]


class WebSocketClient:
    """Manages a WebSocket connection to the CLOB market channel.

    Subscribes to token IDs in batches (max 500 per connection) and
    dispatches incoming messages to a callback that writes to ClickHouse.
    """

    def __init__(self, callback: Callback) -> None:
        self._callback = callback
        self._token_ids: list[str] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self, token_ids: list[str]) -> None:
        """Begin listening. Splits token_ids across connections if > 500."""
        self._running = True
        self._token_ids = token_ids

        chunks = [
            token_ids[i : i + WS_MAX_TOKENS_PER_CONN]
            for i in range(0, len(token_ids), WS_MAX_TOKENS_PER_CONN)
        ]
        for idx, chunk in enumerate(chunks):
            task = asyncio.create_task(
                self._listen_forever(chunk, conn_id=idx),
                name=f"ws-conn-{idx}",
            )
            self._tasks.append(task)

        logger.info(
            "ws_started",
            extra={
                "total_tokens": len(token_ids),
                "connections": len(chunks),
            },
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        """Replace subscriptions with a new set of token IDs."""
        await self.stop()
        await self.start(token_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _listen_forever(self, token_ids: list[str], conn_id: int) -> None:
        delay = WS_RECONNECT_BASE_DELAY
        while self._running:
            try:
                async with websockets.connect(WS_URL) as ws:
                    delay = WS_RECONNECT_BASE_DELAY
                    # Subscribe
                    sub_msg = json.dumps({
                        "type": "MARKET",
                        "assets_ids": token_ids,
                    })
                    await ws.send(sub_msg)
                    logger.info(
                        "ws_subscribed",
                        extra={"conn_id": conn_id, "tokens": len(token_ids)},
                    )

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            await self._dispatch(msg)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                return
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ):
                if not self._running:
                    return
                logger.warning(
                    "ws_reconnecting",
                    extra={"conn_id": conn_id, "delay": delay},
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, WS_RECONNECT_MAX_DELAY)

    async def _dispatch(self, msg: dict) -> None:
        event_type = msg.get("event_type")
        if event_type == "last_trade_price":
            await self._handle_trade(msg)
        elif event_type == "price_change":
            await self._handle_price_change(msg)
        elif event_type == "book":
            await self._handle_book(msg)

    async def _handle_trade(self, msg: dict) -> None:
        """Write a trade event as both a price observation and a trade."""
        ts = self._parse_ts(msg.get("timestamp"))
        condition_id = msg.get("market", "")
        token_id = msg.get("asset_id", "")
        price = float(msg.get("price", 0))
        size = float(msg.get("size", 0))
        side_raw = (msg.get("side") or "BUY").upper()
        side = "buy" if side_raw == "BUY" else "sell"

        # Resolve outcome from token mapping
        mapping = token_mapping.get(token_id, {})
        outcome = mapping.get("outcome", "")

        # Price row
        price_row = [condition_id, token_id, outcome, price, 0.0, 0.0, 0.0, ts]
        await self._callback("market_prices", [price_row])

        # Trade row
        trade_id = f"ws-{token_id}-{int(ts.timestamp() * 1000)}"
        trade_row = [condition_id, token_id, outcome, price, size, side, trade_id, ts]
        await self._callback("market_trades", [trade_row])

    async def _handle_price_change(self, msg: dict) -> None:
        ts = self._parse_ts(msg.get("timestamp"))
        condition_id = msg.get("market", "")

        rows = []
        for change in msg.get("price_changes", []):
            token_id = change.get("asset_id", "")
            price = float(change.get("price", 0))
            best_bid = float(change.get("best_bid", 0))
            best_ask = float(change.get("best_ask", 0))
            mapping = token_mapping.get(token_id, {})
            outcome = mapping.get("outcome", "")
            rows.append([condition_id, token_id, outcome, price, best_bid, best_ask, 0.0, ts])

        if rows:
            await self._callback("market_prices", rows)

    async def _handle_book(self, msg: dict) -> None:
        ts = self._parse_ts(msg.get("timestamp"))
        condition_id = msg.get("market", "")
        token_id = msg.get("asset_id", "")

        bids = msg.get("buys") or []
        asks = msg.get("sells") or []

        bid_prices = [float(b["price"]) for b in bids if "price" in b]
        bid_sizes = [float(b["size"]) for b in bids if "size" in b]
        ask_prices = [float(a["price"]) for a in asks if "price" in a]
        ask_sizes = [float(a["size"]) for a in asks if "size" in a]

        mapping = token_mapping.get(token_id, {})
        outcome = mapping.get("outcome", "")

        row = [
            condition_id, token_id, outcome,
            bid_prices, bid_sizes, ask_prices, ask_sizes,
            ts,
        ]
        await self._callback("orderbook_snapshots", [row])

    @staticmethod
    def _parse_ts(raw: str | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        try:
            # WS timestamps are UNIX milliseconds
            return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
        except (ValueError, TypeError):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return datetime.now(timezone.utc)
