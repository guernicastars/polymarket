"""Batched ClickHouse writer with buffer and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from pipeline.config import (
    BUFFER_FLUSH_INTERVAL,
    BUFFER_FLUSH_SIZE,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    WRITER_BASE_BACKOFF,
    WRITER_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Column definitions for each table
TABLE_COLUMNS: dict[str, list[str]] = {
    "markets": [
        "condition_id", "market_slug", "question", "description",
        "event_id", "event_title", "event_slug", "neg_risk",
        "category", "tags", "outcomes", "outcome_prices", "token_ids",
        "active", "closed", "resolved", "resolution_source", "winning_outcome",
        "volume_24h", "volume_total", "liquidity", "volume_1wk", "volume_1mo",
        "competitive_score", "one_day_price_change", "one_week_price_change",
        "start_date", "end_date", "created_at", "updated_at",
    ],
    "market_prices": [
        "condition_id", "token_id", "outcome",
        "price", "bid", "ask", "volume", "timestamp",
    ],
    "market_trades": [
        "condition_id", "token_id", "outcome",
        "price", "size", "side", "trade_id", "timestamp",
    ],
    "orderbook_snapshots": [
        "condition_id", "token_id", "outcome",
        "bid_prices", "bid_sizes", "ask_prices", "ask_sizes",
        "snapshot_time",
    ],
    "market_events": [
        "condition_id", "event_type", "event_data", "event_time",
    ],
    # Phase 2: User/wallet tables
    "trader_rankings": [
        "proxy_wallet", "user_name", "profile_image",
        "rank", "category", "time_period", "order_by",
        "pnl", "volume",
        "verified_badge", "x_username",
        "snapshot_time",
    ],
    "market_holders": [
        "condition_id", "token_id",
        "proxy_wallet", "pseudonym", "profile_image", "outcome_index",
        "amount",
        "snapshot_time",
    ],
    "wallet_positions": [
        "proxy_wallet", "condition_id", "asset", "outcome", "outcome_index",
        "size", "avg_price", "initial_value", "current_value", "cur_price",
        "cash_pnl", "percent_pnl", "realized_pnl",
        "title", "market_slug", "end_date",
        "updated_at",
    ],
    "wallet_activity": [
        "proxy_wallet", "condition_id", "asset",
        "activity_type", "side", "outcome", "outcome_index",
        "size", "usdc_size", "price",
        "transaction_hash",
        "title", "market_slug",
        "timestamp",
    ],
    "trader_profiles": [
        "proxy_wallet", "pseudonym", "name", "bio", "profile_image",
        "x_username", "verified_badge",
        "display_username_public", "profile_created_at",
        "discovered_via", "first_seen_at", "updated_at",
    ],
    # Phase 3: Advanced analytics tables
    "arbitrage_opportunities": [
        "condition_id", "event_slug",
        "arb_type", "expected_sum", "actual_sum", "spread", "fee_threshold",
        "related_condition_ids", "description",
        "status",
        "detected_at", "resolved_at", "updated_at",
    ],
    "wallet_clusters": [
        "cluster_id",
        "wallets", "size",
        "similarity_score", "timing_corr", "market_overlap", "direction_agreement",
        "common_markets", "label",
        "created_at", "updated_at",
    ],
    "insider_scores": [
        "proxy_wallet",
        "score",
        "factors",
        "freshness_score", "win_rate_score", "niche_score", "size_score", "timing_score",
        "computed_at",
    ],
    "composite_signals": [
        "condition_id",
        "score", "confidence",
        "components",
        "obi_score", "volume_score", "trade_bias_score", "momentum_score",
        "smart_money_score", "concentration_score", "arbitrage_flag", "insider_activity",
        "computed_at",
    ],
}


class ClickHouseWriter:
    """Singleton-style batched writer for ClickHouse.

    Accumulates rows in an in-memory buffer per table and flushes when the
    buffer reaches BUFFER_FLUSH_SIZE rows or BUFFER_FLUSH_INTERVAL seconds.
    """

    _instance: ClickHouseWriter | None = None

    def __init__(self) -> None:
        self._client: Client | None = None
        self._buffers: dict[str, list[list[Any]]] = defaultdict(list)
        self._last_flush: dict[str, float] = defaultdict(time.monotonic)
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> ClickHouseWriter:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=CLICKHOUSE_HOST,
                port=CLICKHOUSE_PORT,
                username=CLICKHOUSE_USER,
                password=CLICKHOUSE_PASSWORD,
                database=CLICKHOUSE_DATABASE,
                secure=True,
                compress="lz4",
                connect_timeout=30,
                send_receive_timeout=300,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(self, table: str, rows: list[list[Any]]) -> None:
        """Add rows to the buffer for *table*. Flushes if threshold met."""
        if not rows:
            return
        async with self._lock:
            self._buffers[table].extend(rows)
            if self._should_flush(table):
                await self._flush_table(table)

    async def flush_all(self) -> None:
        """Force-flush every buffer."""
        async with self._lock:
            for table in list(self._buffers.keys()):
                if self._buffers[table]:
                    await self._flush_table(table)

    async def flush_stale(self) -> None:
        """Flush any buffer older than BUFFER_FLUSH_INTERVAL seconds."""
        now = time.monotonic()
        async with self._lock:
            for table in list(self._buffers.keys()):
                if (
                    self._buffers[table]
                    and (now - self._last_flush[table]) >= BUFFER_FLUSH_INTERVAL
                ):
                    await self._flush_table(table)

    # ------------------------------------------------------------------
    # Convenience helpers for typed inserts
    # ------------------------------------------------------------------

    async def write_markets(self, rows: list[list[Any]]) -> None:
        await self.write("markets", rows)

    async def write_prices(self, rows: list[list[Any]]) -> None:
        await self.write("market_prices", rows)

    async def write_trades(self, rows: list[list[Any]]) -> None:
        await self.write("market_trades", rows)

    async def write_orderbooks(self, rows: list[list[Any]]) -> None:
        await self.write("orderbook_snapshots", rows)

    async def write_events(self, rows: list[list[Any]]) -> None:
        await self.write("market_events", rows)

    # Phase 2 convenience helpers

    async def write_rankings(self, rows: list[list[Any]]) -> None:
        await self.write("trader_rankings", rows)

    async def write_holders(self, rows: list[list[Any]]) -> None:
        await self.write("market_holders", rows)

    async def write_positions(self, rows: list[list[Any]]) -> None:
        await self.write("wallet_positions", rows)

    async def write_activity(self, rows: list[list[Any]]) -> None:
        await self.write("wallet_activity", rows)

    async def write_profiles(self, rows: list[list[Any]]) -> None:
        await self.write("trader_profiles", rows)

    # Phase 3 convenience helpers

    async def write_arbitrage(self, rows: list[list[Any]]) -> None:
        await self.write("arbitrage_opportunities", rows)

    async def write_clusters(self, rows: list[list[Any]]) -> None:
        await self.write("wallet_clusters", rows)

    async def write_insider_scores(self, rows: list[list[Any]]) -> None:
        await self.write("insider_scores", rows)

    async def write_composite_signals(self, rows: list[list[Any]]) -> None:
        await self.write("composite_signals", rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_flush(self, table: str) -> bool:
        if len(self._buffers[table]) >= BUFFER_FLUSH_SIZE:
            return True
        elapsed = time.monotonic() - self._last_flush[table]
        return elapsed >= BUFFER_FLUSH_INTERVAL

    async def _flush_table(self, table: str) -> None:
        rows = self._buffers[table]
        if not rows:
            return

        self._buffers[table] = []
        self._last_flush[table] = time.monotonic()

        columns = TABLE_COLUMNS.get(table)
        if columns is None:
            logger.error("unknown_table", extra={"table": table})
            return

        await self._insert_with_retry(table, rows, columns)

    async def _insert_with_retry(
        self,
        table: str,
        rows: list[list[Any]],
        columns: list[str],
    ) -> None:
        client = self._get_client()
        backoff = WRITER_BASE_BACKOFF

        for attempt in range(1, WRITER_MAX_RETRIES + 1):
            try:
                await asyncio.to_thread(
                    client.insert, table, rows, column_names=columns,
                )
                logger.info(
                    "flush_ok",
                    extra={"table": table, "rows": len(rows)},
                )
                return
            except Exception:
                logger.warning(
                    "insert_retry",
                    extra={
                        "table": table,
                        "attempt": attempt,
                        "backoff": backoff,
                        "rows": len(rows),
                    },
                    exc_info=True,
                )
                if attempt == WRITER_MAX_RETRIES:
                    logger.error(
                        "insert_failed",
                        extra={"table": table, "rows": len(rows)},
                        exc_info=True,
                    )
                    return
                await asyncio.sleep(backoff)
                backoff *= 2
                # Reconnect on next attempt
                self._client = None

    def run_migration(self, sql: str) -> None:
        """Execute raw SQL (for schema migration)."""
        client = self._get_client()
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                client.command(statement)
        logger.info("migration_complete")
