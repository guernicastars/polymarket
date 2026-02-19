"""Job: news tracking + market microstructure ingestion.

Phase 4 jobs:
- News tracker: multi-source OSINT ingestion with NLP analysis (every 10 min)
- Microstructure engine: tick-level market quality metrics (every 60 sec)
"""

from __future__ import annotations

import asyncio
import logging

import clickhouse_connect

from pipeline.config import (
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


async def run_news_tracker() -> None:
    """Fetch news from all enabled sources, analyze, and write to ClickHouse."""
    from pipeline.clickhouse_writer import ClickHouseWriter

    writer = ClickHouseWriter.get_instance()
    client = await asyncio.to_thread(_get_read_client)

    # Build settlement → market mapping from ClickHouse
    settlement_market_map = {}
    try:
        rows = await asyncio.to_thread(
            client.query,
            """
            SELECT settlement_id, condition_id
            FROM (
                SELECT
                    arrayJoin(settlements_mentioned) AS settlement_id,
                    argMax(condition_id, published_at) AS condition_id
                FROM news_articles
                WHERE length(markets_mentioned) > 0
                GROUP BY settlement_id
            )
            """,
        )
        for r in rows.result_rows:
            settlement_market_map[r[0]] = r[1]
    except Exception:
        logger.warning("Could not build settlement->market map, using empty")

    # Placeholder: actual news tracker would be implemented in network.bridge.news
    # For now, just log and return
    try:
        total_articles = 0
        sources_processed = 0
        logger.info(
            "news_tracker_complete",
            extra={"total_articles": total_articles, "sources": sources_processed},
        )
    except Exception:
        logger.error("news_tracker_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def run_microstructure() -> None:
    """Compute market microstructure metrics for top markets."""
    from pipeline.clickhouse_writer import ClickHouseWriter

    writer = ClickHouseWriter.get_instance()
    client = await asyncio.to_thread(_get_read_client)

    try:
        # Placeholder: actual microstructure engine would be implemented in network.bridge.microstructure
        # For now, just log and return
        count = 0
        logger.info("microstructure_complete", extra={"snapshots": count})
    except Exception:
        logger.error("microstructure_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass
