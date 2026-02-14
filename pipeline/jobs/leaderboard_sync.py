"""Job: sync trader leaderboard data from the Data API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import (
    DataClient,
    _LEADERBOARD_CATEGORIES,
    _LEADERBOARD_TIME_PERIODS,
    _LEADERBOARD_ORDER_TYPES,
)
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import LEADERBOARD_MAX_RESULTS

logger = logging.getLogger(__name__)

# Module-level set of discovered wallet addresses (shared with other jobs)
discovered_wallets: set[str] = set()


async def run_leaderboard_sync() -> set[str]:
    """Fetch leaderboard across all combos and write to trader_rankings.

    Returns the set of all discovered wallet addresses (for downstream jobs).
    """
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)
    new_wallets: set[str] = set()

    try:
        total_rows = 0

        for category in _LEADERBOARD_CATEGORIES:
            for time_period in _LEADERBOARD_TIME_PERIODS:
                for order_by in _LEADERBOARD_ORDER_TYPES:
                    entries = await client.fetch_leaderboard_page(
                        category=category,
                        time_period=time_period,
                        order_by=order_by,
                        max_results=LEADERBOARD_MAX_RESULTS,
                    )

                    rows = []
                    for entry in entries:
                        parsed = DataClient.parse_leaderboard_entry(
                            entry, category, time_period, order_by,
                        )
                        wallet = parsed["proxy_wallet"]
                        if wallet:
                            new_wallets.add(wallet)
                            rows.append([
                                parsed["proxy_wallet"],
                                parsed["user_name"],
                                parsed["profile_image"],
                                parsed["rank"],
                                parsed["category"],
                                parsed["time_period"],
                                parsed["order_by"],
                                parsed["pnl"],
                                parsed["volume"],
                                parsed["verified_badge"],
                                parsed["x_username"],
                                now,
                            ])

                    if rows:
                        await writer.write_rankings(rows)
                        total_rows += len(rows)

                    # Small delay between API calls to be respectful
                    await asyncio.sleep(0.2)

        await writer.flush_all()
        discovered_wallets.update(new_wallets)

        logger.info(
            "leaderboard_sync_complete",
            extra={
                "total_entries": total_rows,
                "unique_wallets": len(new_wallets),
                "total_tracked": len(discovered_wallets),
            },
        )
        return discovered_wallets

    finally:
        await client.close()
