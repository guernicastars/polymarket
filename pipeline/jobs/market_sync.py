"""Job: sync market metadata from the Gamma API into ClickHouse."""

from __future__ import annotations

import logging
from datetime import datetime

from pipeline.api.gamma_client import GammaClient
from pipeline.clickhouse_writer import ClickHouseWriter

logger = logging.getLogger(__name__)


async def run_market_sync() -> list[str]:
    """Fetch all active events/markets and upsert into the markets table.

    Returns the list of active token IDs (for use by price_poller and WS).
    """
    gamma = GammaClient()
    writer = ClickHouseWriter.get_instance()

    try:
        events = await gamma.fetch_all_active_events()
        markets = gamma.parse_markets_from_events(events)

        # Build rows for ClickHouse
        rows = []
        active_token_ids: list[str] = []
        resolved_count = 0
        new_count = 0

        # Sort by 24h volume descending so the most active tokens come first
        # when we collect token IDs — this lets downstream jobs slice top-N.
        markets.sort(key=lambda m: m["volume_24h"], reverse=True)

        for m in markets:
            row = [
                m["condition_id"],
                m["market_slug"],
                m["question"],
                m["description"],
                m["event_id"],
                m["event_title"],
                m["event_slug"],
                m["neg_risk"],
                m["category"],
                m["tags"],
                m["outcomes"],
                m["outcome_prices"],
                m["token_ids"],
                m["active"],
                m["closed"],
                m["resolved"],
                m["resolution_source"],
                m["winning_outcome"],
                m["volume_24h"],
                m["volume_total"],
                m["liquidity"],
                m["volume_1wk"],
                m["volume_1mo"],
                m["competitive_score"],
                m["one_day_price_change"],
                m["one_week_price_change"],
                m["start_date"],
                m["end_date"],
                m["created_at"],
                m["updated_at"],
            ]
            rows.append(row)

            # Collect active token IDs for downstream polling
            # (ordered by volume since markets are sorted above)
            if m["active"] and not m["closed"]:
                active_token_ids.extend(m["token_ids"])

            if m["resolved"]:
                resolved_count += 1

        if rows:
            await writer.write_markets(rows)
            await writer.flush_all()

        logger.info(
            "market_sync_complete",
            extra={
                "total_markets": len(markets),
                "active_tokens": len(active_token_ids),
                "resolved": resolved_count,
            },
        )
        return active_token_ids

    finally:
        await gamma.close()
