"""Job: sync top holders for active markets from the Data API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import HOLDER_SYNC_TOP_MARKETS

logger = logging.getLogger(__name__)


async def run_holder_sync(active_condition_ids: list[str]) -> None:
    """Fetch top holders for the top N markets by volume.

    active_condition_ids should be pre-sorted by volume (descending),
    as provided by market_sync.
    """
    if not active_condition_ids:
        logger.debug("holder_sync_skip", extra={"reason": "no_markets"})
        return

    markets_to_poll = active_condition_ids[:HOLDER_SYNC_TOP_MARKETS]
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        total_holders = 0

        for condition_id in markets_to_poll:
            holder_groups = await client.fetch_holders(condition_id)

            rows = []
            for group in holder_groups:
                token_id = group.get("token", "")
                holders = group.get("holders") or []

                for h in holders:
                    parsed = DataClient.parse_holder(h, condition_id, token_id)
                    rows.append([
                        parsed["condition_id"],
                        parsed["token_id"],
                        parsed["proxy_wallet"],
                        parsed["pseudonym"],
                        parsed["profile_image"],
                        parsed["outcome_index"],
                        parsed["amount"],
                        now,
                    ])

            if rows:
                await writer.write_holders(rows)
                total_holders += len(rows)

            # Rate limiting
            await asyncio.sleep(0.15)

        await writer.flush_all()
        logger.info(
            "holder_sync_complete",
            extra={
                "markets_polled": len(markets_to_poll),
                "total_holders": total_holders,
            },
        )

    finally:
        await client.close()
