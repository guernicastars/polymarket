"""Client for the Polymarket Data API (trades, activity)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from pipeline.config import DATA_API_URL, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class DataClient:
    """Fetch public trade data from the Data API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=DATA_API_URL,
            timeout=HTTP_TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_recent_trades(
        self,
        *,
        market: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict]:
        """GET /trades — fetch recent trades, optionally filtered by market.

        No auth required; uses public endpoint.
        """
        params: dict = {"limit": limit, "offset": offset}
        if market:
            params["market"] = market

        try:
            resp = await self._client.get("/trades", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("fetch_trades_error", extra={"market": market}, exc_info=True)
            return []

    async def fetch_all_recent_trades(
        self,
        *,
        market: str | None = None,
        max_pages: int = 5,
    ) -> list[dict]:
        """Paginate through recent trades up to max_pages pages."""
        all_trades: list[dict] = []
        offset = 0
        limit = 500

        for _ in range(max_pages):
            trades = await self.fetch_recent_trades(
                market=market, limit=limit, offset=offset,
            )
            if not trades:
                break
            all_trades.extend(trades)
            if len(trades) < limit:
                break
            offset += limit

        return all_trades

    @staticmethod
    def parse_trade(raw: dict) -> dict | None:
        """Convert a raw Data API trade into a schema-compatible dict.

        Expected raw keys: conditionId, asset, size, price, side,
        timestamp, outcome, transactionHash.
        """
        condition_id = raw.get("conditionId")
        if not condition_id:
            return None

        side_raw = (raw.get("side") or "").upper()
        side = "buy" if side_raw == "BUY" else "sell"

        ts_raw = raw.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            ts = datetime.now(timezone.utc)

        # Use transactionHash as trade_id (unique per on-chain tx)
        trade_id = raw.get("transactionHash") or ""

        return {
            "condition_id": condition_id,
            "token_id": raw.get("asset", ""),
            "outcome": raw.get("outcome", ""),
            "price": float(raw.get("price") or 0),
            "size": float(raw.get("size") or 0),
            "side": side,
            "trade_id": trade_id,
            "timestamp": ts,
        }
