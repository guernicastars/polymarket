"""Client for the Polymarket Gamma API (market discovery and metadata)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx

from pipeline.config import GAMMA_API_URL, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class GammaClient:
    """Fetch events and markets from the Gamma API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=GAMMA_API_URL,
            timeout=HTTP_TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def fetch_all_active_events(self) -> list[dict]:
        """Paginate through all active events with nested markets."""
        all_events: list[dict] = []
        offset = 0
        limit = 100

        while True:
            resp = await self._client.get(
                "/events",
                params={
                    "active": True,
                    "closed": False,
                    "limit": limit,
                    "offset": offset,
                    "order": "id",
                    "ascending": True,
                },
            )
            resp.raise_for_status()
            events = resp.json()

            if not events:
                break

            all_events.extend(events)
            if len(events) < limit:
                break
            offset += limit

        logger.info("gamma_events_fetched", extra={"count": len(all_events)})
        return all_events

    def parse_markets_from_events(
        self, events: list[dict]
    ) -> list[dict]:
        """Flatten events into market rows matching the ClickHouse schema."""
        markets: list[dict] = []
        for event in events:
            event_markets = event.get("markets") or []
            for m in event_markets:
                parsed = self._parse_market(m, event)
                if parsed is not None:
                    markets.append(parsed)
        logger.info("gamma_markets_parsed", extra={"count": len(markets)})
        return markets

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_market(self, m: dict, event: dict) -> dict | None:
        condition_id = m.get("conditionId") or m.get("condition_id")
        if not condition_id:
            return None

        outcomes = self._parse_json_field(m.get("outcomes", "[]"))
        outcome_prices = [
            float(p) for p in self._parse_json_field(m.get("outcomePrices", "[]"))
        ]
        token_ids = self._parse_json_field(m.get("clobTokenIds", "[]"))

        tags_raw = event.get("tags") or []
        tags = [t["label"] for t in tags_raw if isinstance(t, dict) and "label" in t]

        now = datetime.now(timezone.utc)

        return {
            "condition_id": condition_id,
            "market_slug": m.get("slug", ""),
            "question": m.get("question", ""),
            "description": m.get("description", ""),
            # Event grouping
            "event_id": str(event.get("id", "")),
            "event_title": event.get("title", ""),
            "event_slug": event.get("slug", ""),
            "neg_risk": 1 if m.get("negRisk") else 0,
            # Classification
            "category": tags[0] if tags else "",
            "tags": tags,
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "token_ids": token_ids,
            "active": 1 if m.get("active") else 0,
            "closed": 1 if m.get("closed") else 0,
            "resolved": self._is_resolved(m),
            "resolution_source": m.get("resolutionSource", ""),
            "winning_outcome": self._winning_outcome(m, outcomes, outcome_prices),
            "volume_24h": float(m.get("volume24hr") or 0),
            "volume_total": float(m.get("volumeNum") or m.get("volume") or 0),
            "liquidity": float(m.get("liquidityNum") or m.get("liquidity") or 0),
            "volume_1wk": float(m.get("volume1wk") or 0),
            "volume_1mo": float(m.get("volume1mo") or 0),
            "competitive_score": float(m.get("competitive") or 0),
            "one_day_price_change": float(m.get("oneDayPriceChange") or 0),
            "one_week_price_change": float(m.get("oneWeekPriceChange") or 0),
            "start_date": self._parse_dt(m.get("startDate")),
            "end_date": self._parse_dt(m.get("endDate")),
            "created_at": self._parse_dt(m.get("createdAt")),
            "updated_at": now,
            # Extra fields not in CH but useful for downstream jobs
            "_token_ids": token_ids,
            "_best_bid": float(m.get("bestBid") or 0),
            "_best_ask": float(m.get("bestAsk") or 0),
            "_last_trade_price": float(m.get("lastTradePrice") or 0),
        }

    @staticmethod
    def _parse_json_field(raw: str | list) -> list:
        """Handle double-encoded JSON fields (outcomes, outcomePrices, clobTokenIds)."""
        if isinstance(raw, list):
            return raw
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _is_resolved(m: dict) -> int:
        if m.get("closed") and m.get("umaResolutionStatus") == "resolved":
            return 1
        return 0

    @staticmethod
    def _winning_outcome(
        m: dict, outcomes: list, outcome_prices: list[float]
    ) -> str:
        if not m.get("closed"):
            return ""
        for i, price in enumerate(outcome_prices):
            if price == 1.0 and i < len(outcomes):
                return str(outcomes[i])
        return ""

    @staticmethod
    def _parse_dt(raw: str | None) -> datetime:
        if not raw:
            return datetime(2099, 1, 1, tzinfo=timezone.utc)
        try:
            # Handle various ISO formats
            cleaned = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except (ValueError, TypeError):
            return datetime(2099, 1, 1, tzinfo=timezone.utc)
