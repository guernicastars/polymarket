"""Per-market Beta posterior state management with ClickHouse persistence.

Maintains an in-memory cache of active posteriors and syncs to
ClickHouse periodically. On startup, loads the most recent state.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from .combiner import BetaPosterior

logger = logging.getLogger(__name__)


class PosteriorStateStore:
    """Manages per-market Beta posterior state with ClickHouse persistence."""

    def __init__(self, read_client) -> None:
        self._client = read_client
        self._cache: dict[str, BetaPosterior] = {}
        self._dirty: set[str] = set()

    async def load_state(self) -> int:
        """Load latest posterior state from ClickHouse. Returns count loaded."""
        try:
            rows = await asyncio.to_thread(
                self._client.query,
                """SELECT condition_id, alpha, beta, n_updates, last_market_price, updated_at
                   FROM bayesian_state FINAL
                   ORDER BY condition_id""",
            )
            count = 0
            for row in rows.result_rows:
                cid = row[0]
                self._cache[cid] = BetaPosterior(
                    alpha=float(row[1]),
                    beta=float(row[2]),
                    last_updated=row[5] if row[5] else None,
                    n_updates=int(row[3]),
                )
                count += 1
            logger.info("Loaded %d posterior states from ClickHouse", count)
            return count
        except Exception as e:
            logger.warning("Failed to load posterior state: %s", e)
            return 0

    def get(self, condition_id: str) -> Optional[BetaPosterior]:
        return self._cache.get(condition_id)

    def set(self, condition_id: str, posterior: BetaPosterior) -> None:
        self._cache[condition_id] = posterior
        self._dirty.add(condition_id)

    async def sync_to_clickhouse(self, writer) -> int:
        """Flush dirty posteriors to ClickHouse. Returns count written."""
        if not self._dirty:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for cid in self._dirty:
            p = self._cache.get(cid)
            if p is None:
                continue
            rows.append([
                cid,
                p.alpha,
                p.beta,
                p.n_updates,
                p.mean,
                now,
            ])

        if rows:
            try:
                await writer.write("bayesian_state", rows)
                await writer.flush_all()
                count = len(rows)
                self._dirty.clear()
                logger.info("Synced %d posteriors to ClickHouse", count)
                return count
            except Exception as e:
                logger.error("Failed to sync posteriors: %s", e)
                return 0
        return 0

    @property
    def active_markets(self) -> int:
        return len(self._cache)

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)
