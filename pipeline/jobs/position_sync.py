"""Job: sync positions and activity for tracked wallets."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import TRACKED_WALLET_MAX
from pipeline.jobs.leaderboard_sync import discovered_wallets

logger = logging.getLogger(__name__)

# Track the last activity timestamp per wallet for dedup
_last_activity_ts: dict[str, datetime] = {}


async def run_position_sync() -> None:
    """Fetch positions and recent activity for all tracked wallets.

    Tracked wallets are sourced from the leaderboard_sync discovered_wallets set.
    """
    wallets = list(discovered_wallets)[:TRACKED_WALLET_MAX]
    if not wallets:
        logger.debug("position_sync_skip", extra={"reason": "no_tracked_wallets"})
        return

    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        total_positions = 0
        total_activities = 0

        for wallet in wallets:
            # --- Positions ---
            positions = await client.fetch_all_positions(wallet, max_pages=2)
            pos_rows = []
            for raw_pos in positions:
                parsed = DataClient.parse_position(raw_pos)
                if not parsed["condition_id"]:
                    continue
                pos_rows.append([
                    parsed["proxy_wallet"],
                    parsed["condition_id"],
                    parsed["asset"],
                    parsed["outcome"],
                    parsed["outcome_index"],
                    parsed["size"],
                    parsed["avg_price"],
                    parsed["initial_value"],
                    parsed["current_value"],
                    parsed["cur_price"],
                    parsed["cash_pnl"],
                    parsed["percent_pnl"],
                    parsed["realized_pnl"],
                    parsed["title"],
                    parsed["market_slug"],
                    parsed["end_date"],
                    now,
                ])

            if pos_rows:
                await writer.write_positions(pos_rows)
                total_positions += len(pos_rows)

            # --- Activity (recent only) ---
            activities = await client.fetch_activity(wallet, limit=100)
            act_rows = []
            for raw_act in activities:
                parsed = DataClient.parse_activity(raw_act)
                ts = parsed["timestamp"]

                # Deduplicate: skip activities older than last seen
                if wallet in _last_activity_ts and ts <= _last_activity_ts[wallet]:
                    continue

                act_rows.append([
                    parsed["proxy_wallet"],
                    parsed["condition_id"],
                    parsed["asset"],
                    parsed["activity_type"],
                    parsed["side"],
                    parsed["outcome"],
                    parsed["outcome_index"],
                    parsed["size"],
                    parsed["usdc_size"],
                    parsed["price"],
                    parsed["transaction_hash"],
                    parsed["title"],
                    parsed["market_slug"],
                    parsed["timestamp"],
                ])

            if act_rows:
                await writer.write_activity(act_rows)
                total_activities += len(act_rows)

                # Update watermark
                latest_ts = max(
                    DataClient.parse_activity(a)["timestamp"]
                    for a in activities
                    if DataClient.parse_activity(a)["proxy_wallet"] == wallet
                )
                _last_activity_ts[wallet] = latest_ts

            # Rate limiting between wallets
            await asyncio.sleep(0.1)

        await writer.flush_all()
        logger.info(
            "position_sync_complete",
            extra={
                "wallets_polled": len(wallets),
                "total_positions": total_positions,
                "total_activities": total_activities,
            },
        )

    finally:
        await client.close()
