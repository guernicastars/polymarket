"""Job: sync market metadata from the Gamma API into ClickHouse."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pipeline.api.gamma_client import GammaClient
from pipeline.clickhouse_writer import ClickHouseWriter

logger = logging.getLogger(__name__)

# Module-level list of active condition IDs sorted by volume (updated each sync)
active_condition_ids: list[str] = []

# Mapping from token_id -> {condition_id, outcome} for enriching price/trade rows
token_mapping: dict[str, dict[str, str]] = {}

# Track previous market states to detect status changes for market_events
_previous_states: dict[str, dict] = {}  # condition_id -> {active, closed, resolved, winning_outcome}


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

        # Build token_id -> {condition_id, outcome} mapping
        global token_mapping
        new_mapping: dict[str, dict[str, str]] = {}
        for m in markets:
            cid = m["condition_id"]
            outcomes = m["outcomes"]
            tids = m["token_ids"]
            for idx, tid in enumerate(tids):
                outcome_label = outcomes[idx] if idx < len(outcomes) else ""
                new_mapping[tid] = {"condition_id": cid, "outcome": outcome_label}
        token_mapping = new_mapping

        # Detect status changes and emit market_events
        global _previous_states
        now = datetime.now(timezone.utc)
        event_rows = []

        for m in markets:
            cid = m["condition_id"]
            prev = _previous_states.get(cid)
            current = {
                "active": m["active"],
                "closed": m["closed"],
                "resolved": m["resolved"],
                "winning_outcome": m["winning_outcome"],
            }

            if prev is not None:
                if not prev["resolved"] and current["resolved"]:
                    event_rows.append([
                        cid, "resolved",
                        json.dumps({
                            "winning_outcome": m["winning_outcome"],
                            "resolution_source": m["resolution_source"],
                        }),
                        now,
                    ])
                elif not prev["closed"] and current["closed"] and not current["resolved"]:
                    event_rows.append([
                        cid, "closed",
                        json.dumps({"question": m["question"]}),
                        now,
                    ])
            else:
                # First time seeing this market
                event_rows.append([
                    cid, "created",
                    json.dumps({
                        "question": m["question"],
                        "category": m["category"],
                    }),
                    now,
                ])

            _previous_states[cid] = current

        # Update module-level condition IDs for Phase 2 holder sync
        global active_condition_ids
        active_condition_ids = [
            m["condition_id"] for m in markets
            if m["active"] and not m["closed"]
        ]

        if rows:
            await writer.write_markets(rows)
        if event_rows:
            await writer.write_events(event_rows)
            logger.info(
                "market_events_emitted",
                extra={"count": len(event_rows)},
            )
        if rows or event_rows:
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
