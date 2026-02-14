"""Job: snapshot orderbook depth for top markets by volume."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pipeline.api.clob_client import ClobClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import ORDERBOOK_TOP_N

logger = logging.getLogger(__name__)


async def run_orderbook_snapshot(
    active_token_ids: list[str],
    top_n: int = ORDERBOOK_TOP_N,
) -> None:
    """Fetch orderbook for the top N tokens and write to orderbook_snapshots.

    Uses POST /books for batch efficiency. Only snapshots the first top_n
    tokens (the scheduler should sort by volume before passing them in).
    """
    if not active_token_ids:
        logger.debug("orderbook_skip", extra={"reason": "no_active_tokens"})
        return

    tokens_to_snapshot = active_token_ids[:top_n]
    clob = ClobClient()
    writer = ClickHouseWriter.get_instance()

    try:
        books = await clob.fetch_orderbooks(tokens_to_snapshot)
        now = datetime.now(timezone.utc)

        rows = []
        for book in books:
            if not book:
                continue

            condition_id = book.get("market", "")
            token_id = book.get("asset_id", "")

            bids = book.get("bids") or []
            asks = book.get("asks") or []

            bid_prices = [float(b["price"]) for b in bids if "price" in b]
            bid_sizes = [float(b["size"]) for b in bids if "size" in b]
            ask_prices = [float(a["price"]) for a in asks if "price" in a]
            ask_sizes = [float(a["size"]) for a in asks if "size" in a]

            row = [
                condition_id,
                token_id,
                "",  # outcome (not available from /books)
                bid_prices,
                bid_sizes,
                ask_prices,
                ask_sizes,
                now,
            ]
            rows.append(row)

        if rows:
            await writer.write_orderbooks(rows)
            logger.info(
                "orderbook_snapshot_complete",
                extra={"books": len(rows)},
            )
        else:
            logger.debug("orderbook_snapshot_empty")

    finally:
        await clob.close()
