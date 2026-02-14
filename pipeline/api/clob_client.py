"""Client for the Polymarket CLOB API (prices, orderbook, history)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from pipeline.config import CLOB_API_URL, HTTP_TIMEOUT, PRICE_BATCH_SIZE

logger = logging.getLogger(__name__)

# Rate-limit: market data endpoints allow 1500 reqs / 10s.
# We add a small sleep between batches to stay well under the limit.
_RATE_LIMIT_SLEEP = 0.1  # seconds between batch calls


class ClobClient:
    """Async client for CLOB API price and orderbook endpoints (L0 / public)."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=CLOB_API_URL,
            timeout=HTTP_TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    async def fetch_prices(
        self, token_ids: list[str]
    ) -> dict[str, dict[str, str]]:
        """POST /prices for multiple tokens.

        Returns {token_id: {"BUY": "0.65", "SELL": "0.35"}, ...}.
        Batches into groups of PRICE_BATCH_SIZE.
        """
        result: dict[str, dict[str, str]] = {}
        for batch in _chunks(token_ids, PRICE_BATCH_SIZE):
            body = [{"token_id": tid, "side": "BUY"} for tid in batch]
            try:
                resp = await self._client.post("/prices", json=body)
                resp.raise_for_status()
                result.update(resp.json())
            except Exception:
                logger.warning("fetch_prices_error", exc_info=True)
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
        return result

    async def fetch_midpoints(
        self, token_ids: list[str]
    ) -> dict[str, str]:
        """POST /midpoints.  Returns {token_id: "0.50", ...}."""
        result: dict[str, str] = {}
        for batch in _chunks(token_ids, PRICE_BATCH_SIZE):
            body = [{"token_id": tid} for tid in batch]
            try:
                resp = await self._client.post("/midpoints", json=body)
                resp.raise_for_status()
                result.update(resp.json())
            except Exception:
                logger.warning("fetch_midpoints_error", exc_info=True)
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
        return result

    async def fetch_spreads(
        self, token_ids: list[str]
    ) -> dict[str, str]:
        """POST /spreads.  Returns {token_id: "0.02", ...}."""
        result: dict[str, str] = {}
        for batch in _chunks(token_ids, PRICE_BATCH_SIZE):
            body = [{"token_id": tid} for tid in batch]
            try:
                resp = await self._client.post("/spreads", json=body)
                resp.raise_for_status()
                result.update(resp.json())
            except Exception:
                logger.warning("fetch_spreads_error", exc_info=True)
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
        return result

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------

    async def fetch_orderbook(self, token_id: str) -> dict[str, Any] | None:
        """GET /book for a single token."""
        try:
            resp = await self._client.get("/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            logger.warning("fetch_orderbook_error", extra={"token_id": token_id}, exc_info=True)
            return None
        except Exception:
            logger.warning("fetch_orderbook_error", extra={"token_id": token_id}, exc_info=True)
            return None

    async def fetch_orderbooks(
        self, token_ids: list[str]
    ) -> list[dict[str, Any]]:
        """POST /books for multiple tokens."""
        results: list[dict[str, Any]] = []
        for batch in _chunks(token_ids, PRICE_BATCH_SIZE):
            body = [{"token_id": tid} for tid in batch]
            try:
                resp = await self._client.post("/books", json=body)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    results.extend(data)
            except Exception:
                logger.warning("fetch_orderbooks_error", exc_info=True)
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
        return results

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    async def fetch_price_history(
        self,
        token_id: str,
        *,
        interval: str = "max",
        fidelity: int = 60,
    ) -> list[dict]:
        """GET /prices-history for a single token.

        Returns list of {t: timestamp, p: price}.
        """
        try:
            resp = await self._client.get(
                "/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": fidelity,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("history", [])
        except Exception:
            logger.warning("fetch_price_history_error", extra={"token_id": token_id}, exc_info=True)
            return []


def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
