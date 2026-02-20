"""Job: poll current prices for active tokens from the CLOB API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pipeline.api.clob_client import ClobClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.jobs.market_sync import token_mapping

logger = logging.getLogger(__name__)


async def run_price_poller(active_token_ids: list[str]) -> None:
    """Fetch prices, midpoints, and spreads for all active tokens.

    Writes price snapshots to the market_prices table.
    """
    if not active_token_ids:
        logger.debug("price_poller_skip", extra={"reason": "no_active_tokens"})
        return

    clob = ClobClient()
    writer = ClickHouseWriter.get_instance()

    try:
        # Fetch prices (returns {token_id: {"BUY": "0.65", "SELL": "0.35"}})
        prices = await clob.fetch_prices(active_token_ids)
        # Fetch spreads (returns {token_id: "0.02"})
        spreads = await clob.fetch_spreads(active_token_ids)

        now = datetime.now(timezone.utc)
        rows = []

        for token_id in active_token_ids:
            price_data = prices.get(token_id)
            if not price_data:
                continue

            bid_str = price_data.get("BUY", "0")
            ask_str = price_data.get("SELL", "0")
            bid = float(bid_str) if bid_str else 0.0
            ask = float(ask_str) if ask_str else 0.0
            # Use midpoint as the canonical price
            price = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)

            # Look up condition_id and outcome from the market_sync mapping
            mapping = token_mapping.get(token_id, {})
            condition_id = mapping.get("condition_id", "")
            outcome = mapping.get("outcome", "")

            row = [
                condition_id,
                token_id,
                outcome,
                price,
                bid,
                ask,
                0.0,           # volume (not available from /prices)
                now,
            ]
            rows.append(row)

        if rows:
            await writer.write_prices(rows)
            logger.info("price_poll_complete", extra={"snapshots": len(rows)})
        else:
            logger.debug("price_poll_empty")

    finally:
        await clob.close()
