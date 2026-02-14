"""Job: collect recent trades from the Data API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter

logger = logging.getLogger(__name__)

# Track the most recent trade timestamp per market to avoid duplicates.
_last_seen: dict[str, datetime] = {}


async def run_trade_collector() -> None:
    """Fetch recent trades and write new ones to market_trades.

    Uses _last_seen to deduplicate across polling cycles.
    """
    client = DataClient()
    writer = ClickHouseWriter.get_instance()

    try:
        raw_trades = await client.fetch_all_recent_trades(max_pages=3)

        new_rows: list[list] = []
        for raw in raw_trades:
            parsed = DataClient.parse_trade(raw)
            if parsed is None:
                continue

            # Deduplicate: skip trades older than our last-seen watermark
            cid = parsed["condition_id"]
            ts = parsed["timestamp"]
            if cid in _last_seen and ts <= _last_seen[cid]:
                continue

            row = [
                parsed["condition_id"],
                parsed["token_id"],
                parsed["outcome"],
                parsed["price"],
                parsed["size"],
                parsed["side"],
                parsed["trade_id"],
                parsed["timestamp"],
            ]
            new_rows.append(row)

            # Update watermark
            if cid not in _last_seen or ts > _last_seen[cid]:
                _last_seen[cid] = ts

        if new_rows:
            await writer.write_trades(new_rows)
            logger.info("trade_collect_complete", extra={"new_trades": len(new_rows)})
        else:
            logger.debug("trade_collect_empty")

    finally:
        await client.close()
