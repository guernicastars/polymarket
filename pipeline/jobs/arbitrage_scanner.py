"""Job: scan for cross-market arbitrage opportunities."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    ARB_FEE_THRESHOLD,
    ARB_RELATED_MARKET_THRESHOLD,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    """Create a read-only ClickHouse client for queries."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


async def run_arbitrage_scanner() -> None:
    """Scan for sum-to-one and related-market arbitrage opportunities.

    1. Query markets with outcome_prices to check sum-to-one.
    2. Group markets by event_slug to detect related-market inconsistencies.
    3. Write detected opportunities to arbitrage_opportunities table.
    4. Mark previously open opportunities as 'closed' if no longer valid.
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # --- 1. Sum-to-one check ---
        # Fetch active binary markets with their outcome prices
        sum_to_one_rows = await asyncio.to_thread(
            client.query,
            """
            SELECT
                condition_id,
                event_slug,
                outcome_prices,
                outcomes
            FROM markets FINAL
            WHERE active = 1
              AND closed = 0
              AND length(outcome_prices) = 2
            """,
        )

        arb_rows: list[list] = []
        active_arb_keys: set[tuple[str, str]] = set()

        for row in sum_to_one_rows.result_rows:
            condition_id = row[0]
            event_slug = row[1]
            prices = row[2]  # Array of floats

            if len(prices) < 2:
                continue

            actual_sum = sum(prices)
            spread = abs(actual_sum - 1.0)

            if spread > ARB_FEE_THRESHOLD:
                arb_rows.append([
                    condition_id,
                    event_slug,
                    "sum_to_one",           # arb_type
                    1.0,                    # expected_sum
                    actual_sum,             # actual_sum
                    spread,                 # spread
                    ARB_FEE_THRESHOLD,      # fee_threshold
                    [],                     # related_condition_ids
                    f"YES+NO sum={actual_sum:.4f}, spread={spread:.4f}",
                    "open",                 # status
                    now,                    # detected_at
                    datetime(2099, 1, 1, tzinfo=timezone.utc),  # resolved_at
                    now,                    # updated_at
                ])
                active_arb_keys.add((condition_id, "sum_to_one"))

        # --- 2. Related market check ---
        # Group markets by event_slug (non-empty), check if outcomes across
        # markets in the same event have inconsistent pricing
        related_rows = await asyncio.to_thread(
            client.query,
            """
            SELECT
                event_slug,
                groupArray(condition_id) AS condition_ids,
                groupArray(outcome_prices[1]) AS yes_prices,
                groupArray(question) AS questions
            FROM markets FINAL
            WHERE active = 1
              AND closed = 0
              AND event_slug != ''
              AND length(outcome_prices) >= 1
            GROUP BY event_slug
            HAVING count() > 1
            """,
        )

        for row in related_rows.result_rows:
            event_slug = row[0]
            condition_ids = row[1]
            yes_prices = row[2]

            if not yes_prices or len(yes_prices) < 2:
                continue

            # For multi-outcome events (e.g., "Who wins the election?"),
            # YES prices across all markets should sum to ~1.0.
            total = sum(p for p in yes_prices if p > 0)

            if len(yes_prices) >= 3 and total > 0:
                spread = abs(total - 1.0)
                if spread > ARB_RELATED_MARKET_THRESHOLD:
                    primary_id = condition_ids[0]
                    related_ids = condition_ids[1:]

                    arb_rows.append([
                        primary_id,
                        event_slug,
                        "related_market",       # arb_type
                        1.0,                    # expected_sum
                        total,                  # actual_sum
                        spread,                 # spread
                        ARB_RELATED_MARKET_THRESHOLD,
                        related_ids,            # related_condition_ids
                        f"Event '{event_slug}': {len(yes_prices)} outcomes sum={total:.4f}",
                        "open",                 # status
                        now,                    # detected_at
                        datetime(2099, 1, 1, tzinfo=timezone.utc),
                        now,                    # updated_at
                    ])
                    active_arb_keys.add((primary_id, "related_market"))

        # --- 3. Write new/updated opportunities ---
        if arb_rows:
            await writer.write_arbitrage(arb_rows)

        # --- 4. Close resolved opportunities ---
        # Query currently open opportunities and mark as closed if not in active set
        open_opps = await asyncio.to_thread(
            client.query,
            """
            SELECT condition_id, arb_type, detected_at
            FROM arbitrage_opportunities FINAL
            WHERE status = 'open'
            """,
        )

        close_rows = []
        for row in open_opps.result_rows:
            key = (row[0], row[1])
            if key not in active_arb_keys:
                close_rows.append([
                    row[0],                 # condition_id
                    "",                     # event_slug
                    row[1],                 # arb_type
                    1.0,                    # expected_sum
                    0.0,                    # actual_sum
                    0.0,                    # spread
                    ARB_FEE_THRESHOLD,      # fee_threshold
                    [],                     # related_condition_ids
                    "",                     # description
                    "closed",               # status
                    row[2],                 # detected_at (preserve original)
                    now,                    # resolved_at
                    now,                    # updated_at
                ])

        if close_rows:
            await writer.write_arbitrage(close_rows)

        await writer.flush_all()

        logger.info(
            "arbitrage_scan_complete",
            extra={
                "open_opportunities": len(arb_rows),
                "closed_opportunities": len(close_rows),
            },
        )

    except Exception:
        logger.error("arbitrage_scan_error", exc_info=True)
