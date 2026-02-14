# Phase 3 Spec — Advanced Analytics

> Single source of truth for all Phase 3 implementation. Every DDL statement, Python function signature, TypeScript type, query function, and component is defined here. Agents implement directly from this spec.

## Overview

Phase 3 builds **advanced analytics** on top of the Phase 1 signals (OBI, volume anomalies, large trades, technicals) and Phase 2 user/wallet data (leaderboard, positions, activity, profiles). It adds four new capabilities:

1. **Cross-Market Arbitrage Detection** -- Finds pricing inconsistencies within markets and across related markets
2. **Wallet Clustering** -- Groups wallets exhibiting synchronized trading behavior
3. **Insider Scoring** -- Scores wallets on insider-risk factors (freshness, win rate, niche focus, size vs liquidity)
4. **Composite Signal Engine** -- Combines all signal sources into a single -100 to +100 score per market

### Data Flow

```
Existing tables (markets, market_prices, market_trades, orderbook_snapshots,
                 wallet_activity, wallet_positions, trader_rankings, market_holders)
        |
        v
  [Phase 3 Pipeline Jobs]
    arbitrage_scanner (every 2 min)  --> arbitrage_opportunities
    wallet_analyzer   (every 30 min) --> wallet_clusters, insider_scores
    signal_compositor (every 5 min)  --> composite_signals
        |
        v
  [Dashboard Queries]
    getArbitrageOpportunities, getWalletClusters, getInsiderAlerts,
    getCompositeSignals, getAnalyticsOverview
        |
        v
  [Dashboard Pages]
    /analytics (Arbitrage, Clusters, Insider, Composite tabs)
    /signals   (enhanced with composite score column)
```

---

## 1. ClickHouse Schema (DDL)

Create `pipeline/schema/003_phase3_analytics.sql`. Run automatically on startup after `002_phase2_users.sql`.

### 1.1 `arbitrage_opportunities` -- Detected Pricing Inconsistencies

Stores each detected arbitrage opportunity with its current status. ReplacingMergeTree deduplicates by condition_id+arb_type+detected_at, keeping the latest `updated_at`.

```sql
CREATE TABLE IF NOT EXISTS arbitrage_opportunities
(
    -- Identity
    condition_id       LowCardinality(String),          -- Primary market condition ID
    event_slug         String DEFAULT '',                -- Event group for related-market arbs

    -- Arbitrage details
    arb_type           LowCardinality(String),           -- 'sum_to_one' or 'related_market'
    expected_sum       Float64 DEFAULT 1.0,              -- Expected total (1.0 for binary)
    actual_sum         Float64 DEFAULT 0,                -- Observed total
    spread             Float64 DEFAULT 0,                -- |expected - actual|
    fee_threshold      Float64 DEFAULT 0.02,             -- Min spread to flag

    -- Related market details (for related_market type)
    related_condition_ids  Array(String),                 -- Other markets in the event
    description        String DEFAULT '' CODEC(ZSTD(3)),  -- Human-readable arb description

    -- Status
    status             LowCardinality(String) DEFAULT 'open',  -- 'open', 'closed', 'expired'

    -- Timestamps
    detected_at        DateTime64(3) CODEC(DoubleDelta, LZ4),
    resolved_at        DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX event_idx event_slug TYPE bloom_filter GRANULARITY 4,
    INDEX status_idx status TYPE set(5) GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id, arb_type, detected_at)
TTL detected_at + INTERVAL 30 DAY DELETE
SETTINGS index_granularity = 8192;
```

### 1.2 `wallet_clusters` -- Grouped Wallets with Synchronized Behavior

Stores detected wallet clusters. ReplacingMergeTree deduplicates by cluster_id, keeping the latest `updated_at`.

```sql
CREATE TABLE IF NOT EXISTS wallet_clusters
(
    -- Identity
    cluster_id         String,                            -- UUID for the cluster

    -- Cluster members
    wallets            Array(String),                     -- Array of proxy_wallet addresses
    size               UInt32 DEFAULT 0,                  -- Number of wallets in cluster

    -- Scoring
    similarity_score   Float64 DEFAULT 0,                 -- 0-1 behavioral similarity
    timing_corr        Float64 DEFAULT 0,                 -- Trade timing correlation
    market_overlap     Float64 DEFAULT 0,                 -- Fraction of shared markets
    direction_agreement Float64 DEFAULT 0,                -- Fraction of same-direction trades

    -- Metadata
    common_markets     Array(String),                     -- Markets where cluster trades together
    label              String DEFAULT '',                  -- Optional label (e.g., 'suspected_sybil')

    -- Timestamps
    created_at         DateTime64(3) DEFAULT now64(3),
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX cluster_idx cluster_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (cluster_id)
TTL created_at + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;
```

### 1.3 `insider_scores` -- Per-Wallet Insider Risk Score

Stores a composite insider-risk score per wallet. ReplacingMergeTree deduplicates by proxy_wallet, keeping the latest `computed_at`.

```sql
CREATE TABLE IF NOT EXISTS insider_scores
(
    -- Identity
    proxy_wallet       String,                             -- Wallet address

    -- Composite score
    score              Float64 DEFAULT 0,                  -- 0-100, higher = more suspicious

    -- Factor breakdown (JSON for flexibility)
    factors            String DEFAULT '{}' CODEC(ZSTD(3)), -- JSON: {freshness, win_rate, niche_focus, size_vs_liquidity, pre_announcement}

    -- Individual factors (denormalized for query/sort)
    freshness_score    Float64 DEFAULT 0,                  -- 0-100: how new the wallet is
    win_rate_score     Float64 DEFAULT 0,                  -- 0-100: unusually high win rate
    niche_score        Float64 DEFAULT 0,                  -- 0-100: trades only in low-liquidity markets
    size_score         Float64 DEFAULT 0,                  -- 0-100: position size vs market liquidity
    timing_score       Float64 DEFAULT 0,                  -- 0-100: trades before announcements

    -- Timestamps
    computed_at        DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX score_idx score TYPE minmax GRANULARITY 4
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (proxy_wallet)
TTL computed_at + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;
```

### 1.4 `composite_signals` -- Per-Market Multi-Factor Signal

Stores the composite signal score per market. ReplacingMergeTree deduplicates by condition_id, keeping the latest `computed_at`.

```sql
CREATE TABLE IF NOT EXISTS composite_signals
(
    -- Identity
    condition_id       LowCardinality(String),             -- Market condition ID

    -- Composite score
    score              Float64 DEFAULT 0,                  -- -100 (strong bearish) to +100 (strong bullish)
    confidence         Float64 DEFAULT 0,                  -- 0-1, how many signal sources contributed

    -- Component breakdown (JSON for flexibility)
    components         String DEFAULT '{}' CODEC(ZSTD(3)), -- JSON: {obi, volume_anomaly, large_trade_bias, momentum, smart_money, concentration, arbitrage, insider}

    -- Individual components (denormalized for query/sort)
    obi_score          Float64 DEFAULT 0,                  -- -100 to +100 from orderbook imbalance
    volume_score       Float64 DEFAULT 0,                  -- -100 to +100 from volume anomaly
    trade_bias_score   Float64 DEFAULT 0,                  -- -100 to +100 from large trade buy/sell bias
    momentum_score     Float64 DEFAULT 0,                  -- -100 to +100 from price momentum/RSI
    smart_money_score  Float64 DEFAULT 0,                  -- -100 to +100 from whale direction
    concentration_score Float64 DEFAULT 0,                 -- -100 to +100 from holder concentration risk
    arbitrage_flag     UInt8 DEFAULT 0,                    -- 1 = active arbitrage opportunity
    insider_activity   Float64 DEFAULT 0,                  -- 0-100 avg insider score of active wallets

    -- Timestamps
    computed_at        DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX score_idx score TYPE minmax GRANULARITY 4
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (condition_id)
SETTINGS index_granularity = 8192;
```

---

## 2. Pipeline: New Config Constants

Add to `pipeline/config.py`:

```python
# ---------------------------------------------------------------------------
# Phase 3: Advanced analytics intervals
# ---------------------------------------------------------------------------
ARBITRAGE_SCAN_INTERVAL = 120           # 2 minutes
WALLET_ANALYZE_INTERVAL = 1800          # 30 minutes
SIGNAL_COMPOSITE_INTERVAL = 300         # 5 minutes

# Phase 3: Tuning
ARB_FEE_THRESHOLD = 0.02               # Min |sum - 1.0| to flag as arbitrage
ARB_RELATED_MARKET_THRESHOLD = 0.05    # Min pricing inconsistency for related markets
CLUSTER_TIME_WINDOW = 60               # Seconds: trades within this window are "synchronized"
CLUSTER_MIN_OVERLAP = 3                # Min shared markets to consider clustering
CLUSTER_MIN_SIMILARITY = 0.6           # Min similarity score to form a cluster
INSIDER_FRESHNESS_DAYS = 30            # Wallet age below this is "fresh"
INSIDER_WIN_RATE_THRESHOLD = 0.75      # Win rate above this in niche markets is suspicious
COMPOSITE_TOP_MARKETS = 500            # Compute composite signals for top N markets by volume
```

---

## 3. Pipeline: Batched Writer Extension

### 3.1 New TABLE_COLUMNS Entries

Add to the `TABLE_COLUMNS` dict in `pipeline/clickhouse_writer.py`:

```python
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
```

### 3.2 New Convenience Methods

Add to `ClickHouseWriter`:

```python
async def write_arbitrage(self, rows: list[list[Any]]) -> None:
    await self.write("arbitrage_opportunities", rows)

async def write_clusters(self, rows: list[list[Any]]) -> None:
    await self.write("wallet_clusters", rows)

async def write_insider_scores(self, rows: list[list[Any]]) -> None:
    await self.write("insider_scores", rows)

async def write_composite_signals(self, rows: list[list[Any]]) -> None:
    await self.write("composite_signals", rows)
```

---

## 4. Pipeline Jobs

### 4.1 `pipeline/jobs/arbitrage_scanner.py` -- Arbitrage Detection (Every 2 min)

Detects two types of arbitrage:
- **Sum-to-one**: For binary markets, YES + NO outcome prices should sum to ~1.0. Flag if |sum - 1.0| > fee threshold.
- **Related market**: Group markets by `event_slug`, check for logical pricing inconsistencies across outcomes.

```python
"""Job: scan for cross-market arbitrage opportunities."""

from __future__ import annotations

import json
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

    import asyncio

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
            # outcomes = row[3]  # Array of strings

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
            # questions = row[3]

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
```

### 4.2 `pipeline/jobs/wallet_analyzer.py` -- Wallet Clustering + Insider Scoring (Every 30 min)

Two responsibilities in one job:
- **Clustering**: Find wallets that trade the same markets at the same time in the same direction.
- **Insider scoring**: Score each tracked wallet on insider-risk factors.

```python
"""Job: wallet clustering and insider scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    CLUSTER_MIN_OVERLAP,
    CLUSTER_MIN_SIMILARITY,
    CLUSTER_TIME_WINDOW,
    INSIDER_FRESHNESS_DAYS,
    INSIDER_WIN_RATE_THRESHOLD,
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


async def run_wallet_analyzer() -> None:
    """Run wallet clustering and insider scoring.

    1. Query recent wallet_activity for clustering signals.
    2. Compute pairwise wallet similarity based on timing, market overlap, direction.
    3. Group into clusters using greedy algorithm.
    4. Score each tracked wallet on insider-risk factors.
    5. Write results to wallet_clusters and insider_scores tables.
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # =====================================================================
        # PART 1: WALLET CLUSTERING
        # =====================================================================

        # Fetch recent trade activity (last 24h) for all tracked wallets
        activity_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                condition_id,
                side,
                timestamp
            FROM wallet_activity
            WHERE timestamp >= now() - INTERVAL 24 HOUR
              AND activity_type = 'TRADE'
              AND side != ''
            ORDER BY timestamp
            """,
        )

        # Build per-wallet trade profiles
        # wallet -> [(condition_id, side, timestamp), ...]
        wallet_trades: dict[str, list[tuple[str, str, datetime]]] = defaultdict(list)
        for row in activity_result.result_rows:
            wallet = row[0]
            cid = row[1]
            side = row[2]
            ts = row[3]
            wallet_trades[wallet].append((cid, side, ts))

        wallets = list(wallet_trades.keys())

        # Compute pairwise similarity for wallets with enough trades
        cluster_rows: list[list] = []
        clustered_wallets: set[str] = set()

        if len(wallets) >= 2:
            # Pre-compute per-wallet market sets and trade events
            wallet_markets: dict[str, set[str]] = {}
            wallet_events: dict[str, list[tuple[str, str, float]]] = {}

            for w in wallets:
                trades = wallet_trades[w]
                wallet_markets[w] = {t[0] for t in trades}
                wallet_events[w] = [
                    (t[0], t[1], t[2].timestamp() if hasattr(t[2], 'timestamp') else float(t[2]))
                    for t in trades
                ]

            # Pairwise comparison (limit to avoid O(n^2) explosion for large wallet sets)
            comparison_wallets = wallets[:200]  # Cap comparisons

            for i in range(len(comparison_wallets)):
                for j in range(i + 1, len(comparison_wallets)):
                    w1 = comparison_wallets[i]
                    w2 = comparison_wallets[j]

                    if w1 in clustered_wallets and w2 in clustered_wallets:
                        continue

                    # Market overlap
                    shared = wallet_markets[w1] & wallet_markets[w2]
                    if len(shared) < CLUSTER_MIN_OVERLAP:
                        continue

                    all_markets = wallet_markets[w1] | wallet_markets[w2]
                    overlap = len(shared) / len(all_markets) if all_markets else 0

                    # Direction agreement (in shared markets)
                    w1_directions: dict[str, str] = {}
                    w2_directions: dict[str, str] = {}
                    for cid, side, _ in wallet_events[w1]:
                        if cid in shared:
                            w1_directions[cid] = side
                    for cid, side, _ in wallet_events[w2]:
                        if cid in shared:
                            w2_directions[cid] = side

                    agree_count = sum(
                        1 for cid in shared
                        if cid in w1_directions and cid in w2_directions
                        and w1_directions[cid] == w2_directions[cid]
                    )
                    direction_agree = agree_count / len(shared) if shared else 0

                    # Timing correlation (trades within CLUSTER_TIME_WINDOW seconds)
                    timing_matches = 0
                    timing_total = 0
                    for cid in shared:
                        w1_times = [ts for c, _, ts in wallet_events[w1] if c == cid]
                        w2_times = [ts for c, _, ts in wallet_events[w2] if c == cid]
                        for t1 in w1_times:
                            for t2 in w2_times:
                                timing_total += 1
                                if abs(t1 - t2) <= CLUSTER_TIME_WINDOW:
                                    timing_matches += 1

                    timing_corr = timing_matches / timing_total if timing_total > 0 else 0

                    # Composite similarity
                    similarity = (
                        0.3 * overlap +
                        0.4 * direction_agree +
                        0.3 * timing_corr
                    )

                    if similarity >= CLUSTER_MIN_SIMILARITY:
                        cluster_id = str(uuid.uuid4())
                        cluster_wallets = [w1, w2]
                        common = list(shared)[:20]  # Cap stored common markets

                        cluster_rows.append([
                            cluster_id,
                            cluster_wallets,            # wallets array
                            len(cluster_wallets),       # size
                            similarity,                 # similarity_score
                            timing_corr,                # timing_corr
                            overlap,                    # market_overlap
                            direction_agree,            # direction_agreement
                            common,                     # common_markets
                            "",                         # label
                            now,                        # created_at
                            now,                        # updated_at
                        ])

                        clustered_wallets.add(w1)
                        clustered_wallets.add(w2)

        if cluster_rows:
            await writer.write_clusters(cluster_rows)

        # =====================================================================
        # PART 2: INSIDER SCORING
        # =====================================================================

        # Fetch tracked wallet metadata for scoring
        # We need: wallet age, win rate, market diversity, position sizes
        profile_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                profile_created_at,
                first_seen_at
            FROM trader_profiles FINAL
            """,
        )

        wallet_profiles: dict[str, dict] = {}
        for row in profile_result.result_rows:
            wallet_profiles[row[0]] = {
                "created_at": row[1],
                "first_seen": row[2],
            }

        # Win rate per wallet (resolved markets where they held the winning outcome)
        win_rate_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wp.proxy_wallet,
                countIf(m.resolved = 1 AND m.winning_outcome = wp.outcome) AS wins,
                countIf(m.resolved = 1) AS resolved_count
            FROM (SELECT * FROM wallet_positions FINAL) AS wp
            INNER JOIN (
                SELECT condition_id, resolved, winning_outcome
                FROM markets FINAL
                WHERE resolved = 1
            ) AS m ON wp.condition_id = m.condition_id
            WHERE wp.size > 0
            GROUP BY wp.proxy_wallet
            HAVING resolved_count >= 3
            """,
        )

        wallet_win_rates: dict[str, tuple[int, int]] = {}
        for row in win_rate_result.result_rows:
            wallet_win_rates[row[0]] = (row[1], row[2])  # (wins, total)

        # Position size vs market liquidity
        size_vs_liq_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wp.proxy_wallet,
                avg(wp.current_value / greatest(m.liquidity, 1.0)) AS avg_size_ratio
            FROM (SELECT * FROM wallet_positions FINAL) AS wp
            INNER JOIN (
                SELECT condition_id, liquidity
                FROM markets FINAL
                WHERE active = 1
            ) AS m ON wp.condition_id = m.condition_id
            WHERE wp.size > 0
            GROUP BY wp.proxy_wallet
            """,
        )

        wallet_size_ratios: dict[str, float] = {}
        for row in size_vs_liq_result.result_rows:
            wallet_size_ratios[row[0]] = float(row[1])

        # Market diversity (how many unique markets a wallet trades in)
        diversity_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                uniq(condition_id) AS unique_markets
            FROM wallet_activity
            WHERE activity_type = 'TRADE'
              AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY proxy_wallet
            """,
        )

        wallet_diversity: dict[str, int] = {}
        for row in diversity_result.result_rows:
            wallet_diversity[row[0]] = int(row[1])

        # Compute insider scores
        insider_rows: list[list] = []
        all_scored_wallets = set(wallet_profiles.keys()) | set(wallet_win_rates.keys())

        for wallet in all_scored_wallets:
            # --- Freshness score (0-100) ---
            profile = wallet_profiles.get(wallet, {})
            created = profile.get("created_at")
            if created and hasattr(created, 'timestamp'):
                age_days = (now - created.replace(tzinfo=timezone.utc if created.tzinfo is None else created.tzinfo)).days
            else:
                age_days = 999  # Unknown = not fresh

            if age_days <= INSIDER_FRESHNESS_DAYS:
                freshness = 100 * (1 - age_days / INSIDER_FRESHNESS_DAYS)
            else:
                freshness = 0.0

            # --- Win rate score (0-100) ---
            wins, total = wallet_win_rates.get(wallet, (0, 0))
            if total >= 3:
                wr = wins / total
                if wr >= INSIDER_WIN_RATE_THRESHOLD:
                    win_rate_s = min(100, (wr - 0.5) * 200)  # 0.5 -> 0, 1.0 -> 100
                else:
                    win_rate_s = 0.0
            else:
                win_rate_s = 0.0

            # --- Niche score (0-100) ---
            diversity = wallet_diversity.get(wallet, 0)
            if 0 < diversity <= 3:
                niche_s = 80.0  # Very focused
            elif diversity <= 5:
                niche_s = 40.0
            else:
                niche_s = 0.0

            # --- Size vs liquidity score (0-100) ---
            size_ratio = wallet_size_ratios.get(wallet, 0)
            if size_ratio > 0.1:
                size_s = min(100, size_ratio * 500)  # 0.1 -> 50, 0.2 -> 100
            else:
                size_s = 0.0

            # --- Timing score (0-100) ---
            # Placeholder: use 0 for now; requires event resolution timestamps
            # which we do not yet track. Future enhancement: compare trade timestamps
            # to market resolution or news event timestamps.
            timing_s = 0.0

            # --- Composite score ---
            composite = (
                0.20 * freshness +
                0.30 * win_rate_s +
                0.15 * niche_s +
                0.25 * size_s +
                0.10 * timing_s
            )

            factors_json = json.dumps({
                "freshness": round(freshness, 2),
                "win_rate": round(win_rate_s, 2),
                "niche_focus": round(niche_s, 2),
                "size_vs_liquidity": round(size_s, 2),
                "pre_announcement": round(timing_s, 2),
            })

            insider_rows.append([
                wallet,
                round(composite, 2),        # score
                factors_json,                # factors JSON
                round(freshness, 2),         # freshness_score
                round(win_rate_s, 2),        # win_rate_score
                round(niche_s, 2),           # niche_score
                round(size_s, 2),            # size_score
                round(timing_s, 2),          # timing_score
                now,                         # computed_at
            ])

        if insider_rows:
            await writer.write_insider_scores(insider_rows)

        await writer.flush_all()

        logger.info(
            "wallet_analyzer_complete",
            extra={
                "clusters_found": len(cluster_rows),
                "wallets_scored": len(insider_rows),
            },
        )

    except Exception:
        logger.error("wallet_analyzer_error", exc_info=True)
```

### 4.3 `pipeline/jobs/signal_compositor.py` -- Composite Signal Engine (Every 5 min)

Combines all signal sources into a single score per market.

```python
"""Job: compute composite signal scores per market."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    COMPOSITE_TOP_MARKETS,
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


def _clamp(value: float, lo: float = -100.0, hi: float = 100.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


async def run_signal_compositor() -> None:
    """Compute composite signal scores for top active markets.

    Components (each normalized to -100..+100):
    1. OBI direction: from latest orderbook snapshot
    2. Volume anomaly: 4h volume vs 7d average
    3. Large trade bias: net buy/sell from large trades (24h)
    4. Momentum: from hourly OHLCV (24h price change)
    5. Smart money direction: net buy/sell from top-ranked wallets
    6. Concentration risk: top-5 holder share (high = risky = negative)
    7. Arbitrage flag: 1 if active arbitrage opportunity exists
    8. Insider activity: average insider score of wallets active in this market
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # Get top active markets
        markets_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT condition_id
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            ORDER BY volume_24h DESC
            LIMIT {COMPOSITE_TOP_MARKETS}
            """,
        )

        market_ids = [row[0] for row in markets_result.result_rows]
        if not market_ids:
            logger.debug("signal_compositor_skip", extra={"reason": "no_markets"})
            return

        # --- 1. OBI scores ---
        obi_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                os.condition_id,
                arraySum(os.bid_sizes) / greatest(arraySum(os.bid_sizes) + arraySum(os.ask_sizes), 0.001) AS obi
            FROM orderbook_snapshots os
            INNER JOIN (
                SELECT condition_id, max(snapshot_time) AS max_time
                FROM orderbook_snapshots
                WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
                GROUP BY condition_id
            ) latest ON os.condition_id = latest.condition_id
              AND os.snapshot_time = latest.max_time
            WHERE (arraySum(os.bid_sizes) + arraySum(os.ask_sizes)) > 0
            """,
        )

        obi_scores: dict[str, float] = {}
        for row in obi_result.result_rows:
            # OBI 0.5 = neutral (0), 0 = full bearish (-100), 1 = full bullish (+100)
            obi_scores[row[0]] = _clamp((float(row[1]) - 0.5) * 200)

        # --- 2. Volume anomaly scores ---
        vol_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                t.condition_id,
                sum(t.size) AS vol_4h,
                m.volume_1wk / 7 / 6 AS avg_4h_vol
            FROM market_trades t
            INNER JOIN (
                SELECT condition_id, volume_1wk
                FROM markets FINAL
                WHERE active = 1 AND closed = 0 AND volume_1wk > 0
            ) AS m ON t.condition_id = m.condition_id
            WHERE t.timestamp >= now() - INTERVAL 4 HOUR
            GROUP BY t.condition_id, m.volume_1wk
            """,
        )

        volume_scores: dict[str, float] = {}
        for row in vol_result.result_rows:
            vol_4h = float(row[1])
            avg_4h = float(row[2])
            if avg_4h > 0:
                ratio = vol_4h / avg_4h
                # ratio 1.0 = normal (0), 3.0+ = strong anomaly (+100)
                volume_scores[row[0]] = _clamp((ratio - 1.0) * 50)
            else:
                volume_scores[row[0]] = 0.0

        # --- 3. Large trade bias ---
        trade_bias_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                condition_id,
                sumIf(price * size, side = 'buy') AS buy_usd,
                sumIf(price * size, side = 'sell') AS sell_usd
            FROM market_trades
            WHERE timestamp >= now() - INTERVAL 24 HOUR
              AND price * size >= 1000
            GROUP BY condition_id
            """,
        )

        trade_bias_scores: dict[str, float] = {}
        for row in trade_bias_result.result_rows:
            buy = float(row[1])
            sell = float(row[2])
            total = buy + sell
            if total > 0:
                # Net buy ratio: 1.0 = all buys (+100), 0.0 = all sells (-100)
                net_ratio = (buy - sell) / total
                trade_bias_scores[row[0]] = _clamp(net_ratio * 100)
            else:
                trade_bias_scores[row[0]] = 0.0

        # --- 4. Momentum (24h price change from Gamma API data) ---
        momentum_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                condition_id,
                one_day_price_change
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            """,
        )

        momentum_scores: dict[str, float] = {}
        for row in momentum_result.result_rows:
            change = float(row[1])
            # Price change is typically -1 to +1 range. Scale to -100..+100.
            momentum_scores[row[0]] = _clamp(change * 200)

        # --- 5. Smart money direction ---
        smart_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wa.condition_id,
                sumIf(wa.usdc_size, wa.side = 'BUY') AS sm_buy,
                sumIf(wa.usdc_size, wa.side = 'SELL') AS sm_sell
            FROM wallet_activity wa
            INNER JOIN (
                SELECT proxy_wallet
                FROM trader_rankings FINAL
                WHERE category = 'OVERALL' AND time_period = 'ALL' AND order_by = 'PNL'
                  AND rank <= 50
            ) AS tr ON wa.proxy_wallet = tr.proxy_wallet
            WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
              AND wa.activity_type = 'TRADE'
            GROUP BY wa.condition_id
            """,
        )

        smart_scores: dict[str, float] = {}
        for row in smart_result.result_rows:
            buy = float(row[1])
            sell = float(row[2])
            total = buy + sell
            if total > 0:
                net = (buy - sell) / total
                smart_scores[row[0]] = _clamp(net * 100)
            else:
                smart_scores[row[0]] = 0.0

        # --- 6. Concentration risk ---
        conc_result = await asyncio.to_thread(
            client.query,
            """
            WITH holder_stats AS (
                SELECT
                    condition_id,
                    sum(amount) AS total_amount,
                    arraySlice(groupArray(amount), 1, 5) AS top5_amounts
                FROM (
                    SELECT condition_id, amount
                    FROM market_holders FINAL
                    ORDER BY amount DESC
                )
                GROUP BY condition_id
                HAVING total_amount > 0
            )
            SELECT
                condition_id,
                arraySum(top5_amounts) / total_amount AS top5_share
            FROM holder_stats
            """,
        )

        concentration_scores: dict[str, float] = {}
        for row in conc_result.result_rows:
            share = float(row[1])
            # High concentration is risk (negative signal)
            # share 0.5 = neutral (0), 1.0 = full concentration (-100)
            concentration_scores[row[0]] = _clamp(-(share - 0.5) * 200)

        # --- 7. Arbitrage flags ---
        arb_result = await asyncio.to_thread(
            client.query,
            """
            SELECT DISTINCT condition_id
            FROM arbitrage_opportunities FINAL
            WHERE status = 'open'
            """,
        )

        arb_flags: set[str] = {row[0] for row in arb_result.result_rows}

        # --- 8. Insider activity ---
        insider_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wa.condition_id,
                avg(ins.score) AS avg_insider_score
            FROM wallet_activity wa
            INNER JOIN (
                SELECT proxy_wallet, score
                FROM insider_scores FINAL
                WHERE score > 20
            ) AS ins ON wa.proxy_wallet = ins.proxy_wallet
            WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
              AND wa.activity_type = 'TRADE'
            GROUP BY wa.condition_id
            """,
        )

        insider_scores_map: dict[str, float] = {}
        for row in insider_result.result_rows:
            insider_scores_map[row[0]] = float(row[1])

        # =====================================================================
        # COMPOSITE CALCULATION
        # =====================================================================

        # Weights for each component
        WEIGHTS = {
            "obi": 0.20,
            "volume": 0.10,
            "trade_bias": 0.15,
            "momentum": 0.15,
            "smart_money": 0.25,
            "concentration": 0.10,
            "insider": 0.05,
        }

        signal_rows: list[list] = []

        for cid in market_ids:
            obi_s = obi_scores.get(cid, 0.0)
            vol_s = volume_scores.get(cid, 0.0)
            bias_s = trade_bias_scores.get(cid, 0.0)
            mom_s = momentum_scores.get(cid, 0.0)
            smart_s = smart_scores.get(cid, 0.0)
            conc_s = concentration_scores.get(cid, 0.0)
            arb_f = 1 if cid in arb_flags else 0
            ins_s = insider_scores_map.get(cid, 0.0)

            # Weighted composite
            composite = (
                WEIGHTS["obi"] * obi_s +
                WEIGHTS["volume"] * vol_s +
                WEIGHTS["trade_bias"] * bias_s +
                WEIGHTS["momentum"] * mom_s +
                WEIGHTS["smart_money"] * smart_s +
                WEIGHTS["concentration"] * conc_s +
                WEIGHTS["insider"] * (ins_s - 50)  # Center insider around 0
            )
            composite = _clamp(composite)

            # Confidence: how many signal sources had non-zero data
            sources = [obi_s, vol_s, bias_s, mom_s, smart_s, conc_s]
            active_sources = sum(1 for s in sources if abs(s) > 1.0)
            confidence = active_sources / len(sources)

            components_json = json.dumps({
                "obi": round(obi_s, 2),
                "volume_anomaly": round(vol_s, 2),
                "large_trade_bias": round(bias_s, 2),
                "momentum": round(mom_s, 2),
                "smart_money": round(smart_s, 2),
                "concentration": round(conc_s, 2),
                "arbitrage": arb_f,
                "insider": round(ins_s, 2),
            })

            signal_rows.append([
                cid,                         # condition_id
                round(composite, 2),         # score
                round(confidence, 3),        # confidence
                components_json,             # components JSON
                round(obi_s, 2),            # obi_score
                round(vol_s, 2),            # volume_score
                round(bias_s, 2),           # trade_bias_score
                round(mom_s, 2),            # momentum_score
                round(smart_s, 2),          # smart_money_score
                round(conc_s, 2),           # concentration_score
                arb_f,                       # arbitrage_flag
                round(ins_s, 2),            # insider_activity
                now,                         # computed_at
            ])

        if signal_rows:
            await writer.write_composite_signals(signal_rows)

        await writer.flush_all()

        logger.info(
            "signal_compositor_complete",
            extra={
                "markets_scored": len(signal_rows),
            },
        )

    except Exception:
        logger.error("signal_compositor_error", exc_info=True)
```

---

## 5. Scheduler Updates

### 5.1 New Imports

Add to `pipeline/scheduler.py`:

```python
from pipeline.config import (
    # ...existing imports...
    ARBITRAGE_SCAN_INTERVAL,
    WALLET_ANALYZE_INTERVAL,
    SIGNAL_COMPOSITE_INTERVAL,
)
from pipeline.jobs.arbitrage_scanner import run_arbitrage_scanner
from pipeline.jobs.wallet_analyzer import run_wallet_analyzer
from pipeline.jobs.signal_compositor import run_signal_compositor
```

### 5.2 New Job Registrations

Add inside `PipelineScheduler.start()`, after the existing Phase 2 job registrations:

```python
# --- Phase 3 jobs ---
self._scheduler.add_job(
    self._job_arbitrage_scanner,
    "interval",
    seconds=ARBITRAGE_SCAN_INTERVAL,
    id="arbitrage_scanner",
    name="Arbitrage Scanner",
)
self._scheduler.add_job(
    self._job_wallet_analyzer,
    "interval",
    seconds=WALLET_ANALYZE_INTERVAL,
    id="wallet_analyzer",
    name="Wallet Analyzer",
)
self._scheduler.add_job(
    self._job_signal_compositor,
    "interval",
    seconds=SIGNAL_COMPOSITE_INTERVAL,
    id="signal_compositor",
    name="Signal Compositor",
)
```

### 5.3 New Job Wrappers

Add to the `PipelineScheduler` class:

```python
# --- Phase 3 job wrappers ---

async def _job_arbitrage_scanner(self) -> None:
    try:
        await run_arbitrage_scanner()
    except Exception:
        logger.error("arbitrage_scanner_error", exc_info=True)

async def _job_wallet_analyzer(self) -> None:
    try:
        await run_wallet_analyzer()
    except Exception:
        logger.error("wallet_analyzer_error", exc_info=True)

async def _job_signal_compositor(self) -> None:
    try:
        await run_signal_compositor()
    except Exception:
        logger.error("signal_compositor_error", exc_info=True)
```

### 5.4 Updated Health Check

Update `_health_handler` to include Phase 3 stats:

```python
async def _health_handler(self, request: web.Request) -> web.Response:
    from pipeline.jobs.leaderboard_sync import discovered_wallets
    return web.json_response({
        "status": "ok",
        "active_tokens": len(self._active_token_ids),
        "tracked_wallets": len(discovered_wallets),
        "scheduler_running": self._scheduler.running,
        "phase3_jobs": ["arbitrage_scanner", "wallet_analyzer", "signal_compositor"],
    })
```

---

## 6. Schema Migration

### 6.1 New Migration File

Create `pipeline/schema/003_phase3_analytics.sql` with all 4 CREATE TABLE statements from Section 1.

### 6.2 Migration Runner Update

Update the migration section in `main.py` to include the new schema file:

```python
schema_files = ["001_init.sql", "002_phase2_users.sql", "003_phase3_analytics.sql"]
for schema_file in schema_files:
    schema_path = Path(__file__).parent / "schema" / schema_file
    if schema_path.exists():
        writer.run_migration(schema_path.read_text())
```

---

## 7. Dashboard: TypeScript Types

Add to `dashboard/src/types/market.ts`:

```typescript
// --- Phase 3 Analytics Types ---

export interface ArbitrageOpportunity {
  condition_id: string;
  event_slug: string;
  arb_type: string;            // 'sum_to_one' | 'related_market'
  expected_sum: number;
  actual_sum: number;
  spread: number;
  related_condition_ids: string[];
  description: string;
  status: string;               // 'open' | 'closed' | 'expired'
  detected_at: string;
  resolved_at: string;
  question?: string;            // Joined from markets
}

export interface WalletCluster {
  cluster_id: string;
  wallets: string[];
  size: number;
  similarity_score: number;
  timing_corr: number;
  market_overlap: number;
  direction_agreement: number;
  common_markets: string[];
  label: string;
  created_at: string;
}

export interface InsiderAlert {
  proxy_wallet: string;
  pseudonym: string;            // Joined from trader_profiles
  profile_image: string;        // Joined from trader_profiles
  score: number;                // 0-100
  freshness_score: number;
  win_rate_score: number;
  niche_score: number;
  size_score: number;
  timing_score: number;
  computed_at: string;
}

export interface CompositeSignal {
  condition_id: string;
  question: string;             // Joined from markets
  score: number;                // -100 to +100
  confidence: number;           // 0-1
  obi_score: number;
  volume_score: number;
  trade_bias_score: number;
  momentum_score: number;
  smart_money_score: number;
  concentration_score: number;
  arbitrage_flag: number;       // 0 or 1
  insider_activity: number;     // 0-100
  computed_at: string;
}

export interface AnalyticsOverview {
  open_arbitrages: number;
  wallet_clusters: number;
  insider_alerts: number;       // Wallets with score > 50
  markets_scored: number;       // Markets with composite signals
  avg_confidence: number;
}
```

---

## 8. Dashboard: Query Functions

Add to `dashboard/src/lib/queries.ts`. Import the new types.

### 8.1 Updated Imports

```typescript
import type {
  // ...existing imports...
  ArbitrageOpportunity,
  WalletCluster,
  InsiderAlert,
  CompositeSignal,
  AnalyticsOverview,
} from "@/types/market";
```

### 8.2 `getArbitrageOpportunities`

```typescript
export async function getArbitrageOpportunities(
  limit = 50
): Promise<ArbitrageOpportunity[]> {
  return query<ArbitrageOpportunity>(
    `SELECT
      ao.condition_id,
      ao.event_slug,
      ao.arb_type,
      ao.expected_sum,
      ao.actual_sum,
      ao.spread,
      ao.related_condition_ids,
      ao.description,
      ao.status,
      ao.detected_at,
      ao.resolved_at,
      m.question
    FROM (SELECT * FROM arbitrage_opportunities FINAL) AS ao
    LEFT JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON ao.condition_id = m.condition_id
    WHERE ao.status = 'open'
    ORDER BY ao.spread DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.3 `getWalletClusters`

```typescript
export async function getWalletClusters(
  limit = 30
): Promise<WalletCluster[]> {
  return query<WalletCluster>(
    `SELECT
      cluster_id,
      wallets,
      size,
      similarity_score,
      timing_corr,
      market_overlap,
      direction_agreement,
      common_markets,
      label,
      created_at
    FROM wallet_clusters FINAL
    ORDER BY similarity_score DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.4 `getInsiderAlerts`

```typescript
export async function getInsiderAlerts(
  minScore = 30,
  limit = 50
): Promise<InsiderAlert[]> {
  return query<InsiderAlert>(
    `SELECT
      ins.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      ins.score,
      ins.freshness_score,
      ins.win_rate_score,
      ins.niche_score,
      ins.size_score,
      ins.timing_score,
      ins.computed_at
    FROM (SELECT * FROM insider_scores FINAL) AS ins
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON ins.proxy_wallet = tp.proxy_wallet
    WHERE ins.score >= {minScore:Float64}
    ORDER BY ins.score DESC
    LIMIT {limit:UInt32}`,
    { minScore, limit }
  );
}
```

### 8.5 `getCompositeSignals`

```typescript
export async function getCompositeSignals(
  limit = 50
): Promise<CompositeSignal[]> {
  return query<CompositeSignal>(
    `SELECT
      cs.condition_id,
      m.question,
      cs.score,
      cs.confidence,
      cs.obi_score,
      cs.volume_score,
      cs.trade_bias_score,
      cs.momentum_score,
      cs.smart_money_score,
      cs.concentration_score,
      cs.arbitrage_flag,
      cs.insider_activity,
      cs.computed_at
    FROM (SELECT * FROM composite_signals FINAL) AS cs
    INNER JOIN (
      SELECT condition_id, question
      FROM markets FINAL
      WHERE active = 1 AND closed = 0
    ) AS m ON cs.condition_id = m.condition_id
    ORDER BY abs(cs.score) DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.6 `getAnalyticsOverview`

```typescript
export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  const rows = await query<AnalyticsOverview>(
    `SELECT
      (
        SELECT count()
        FROM arbitrage_opportunities FINAL
        WHERE status = 'open'
      ) AS open_arbitrages,
      (
        SELECT count()
        FROM wallet_clusters FINAL
      ) AS wallet_clusters,
      (
        SELECT count()
        FROM insider_scores FINAL
        WHERE score > 50
      ) AS insider_alerts,
      (
        SELECT count()
        FROM composite_signals FINAL
      ) AS markets_scored,
      (
        SELECT avg(confidence)
        FROM composite_signals FINAL
      ) AS avg_confidence`
  );
  return rows[0] ?? {
    open_arbitrages: 0,
    wallet_clusters: 0,
    insider_alerts: 0,
    markets_scored: 0,
    avg_confidence: 0,
  };
}
```

---

## 9. Dashboard: Pages and Components

### 9.1 Page: `/analytics` (`dashboard/src/app/analytics/page.tsx`)

Server component with `export const dynamic = "force-dynamic"`. Follows the same Suspense pattern as `/signals` and `/whales`.

**Layout:**
```
[Stats Cards Row] — 5 cards: Open Arbitrages, Wallet Clusters, Insider Alerts, Markets Scored, Avg Confidence
[Tabs: Composite | Arbitrage | Clusters | Insider]
  [Composite Tab]   --> CompositeSignalsTable
  [Arbitrage Tab]   --> ArbitrageTable
  [Clusters Tab]    --> WalletClustersTable
  [Insider Tab]     --> InsiderAlertsTable
```

**Implementation:**

```typescript
import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getAnalyticsOverview,
  getCompositeSignals,
  getArbitrageOpportunities,
  getWalletClusters,
  getInsiderAlerts,
} from "@/lib/queries";
import {
  AnalyticsStatsCards,
  AnalyticsStatsCardsSkeleton,
} from "@/components/analytics-stats-cards";
import { AnalyticsTabs } from "@/components/analytics-tabs";

export const dynamic = "force-dynamic";

async function AnalyticsStatsSection() {
  const stats = await getAnalyticsOverview();
  return <AnalyticsStatsCards stats={stats} />;
}

async function AnalyticsContent() {
  const [composite, arbitrage, clusters, insider] = await Promise.all([
    getCompositeSignals(50),
    getArbitrageOpportunities(50),
    getWalletClusters(30),
    getInsiderAlerts(30, 50),
  ]);
  return (
    <AnalyticsTabs
      composite={composite}
      arbitrage={arbitrage}
      clusters={clusters}
      insider={insider}
    />
  );
}

export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Advanced analytics: arbitrage detection, wallet clustering, insider
          scoring, and composite signals
        </p>
      </div>

      <Suspense fallback={<AnalyticsStatsCardsSkeleton />}>
        <AnalyticsStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Advanced Signals</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense
            fallback={
              <div className="space-y-3">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-12 bg-[#1e1e2e] rounded animate-pulse"
                  />
                ))}
              </div>
            }
          >
            <AnalyticsContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
```

### 9.2 Component: `AnalyticsStatsCards` (`dashboard/src/components/analytics-stats-cards.tsx`)

Client component. Five stat cards.

| Card | Value | Icon (lucide-react) | Color |
|------|-------|---------------------|-------|
| Open Arbitrages | `stats.open_arbitrages` | `ArrowLeftRight` | `text-amber-400` |
| Wallet Clusters | `stats.wallet_clusters` | `Network` | `text-violet-400` |
| Insider Alerts | `stats.insider_alerts` | `ShieldAlert` | `text-red-400` |
| Markets Scored | `stats.markets_scored` | `Target` | `text-emerald-400` |
| Avg Confidence | `stats.avg_confidence` | `Gauge` | `text-blue-400` |

```typescript
"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeftRight, Network, ShieldAlert, Target, Gauge } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { AnalyticsOverview } from "@/types/market";

interface AnalyticsStatsCardsProps {
  stats: AnalyticsOverview;
}

export function AnalyticsStatsCards({ stats }: AnalyticsStatsCardsProps) {
  const cards = [
    { label: "Open Arbitrages", value: formatNumber(stats.open_arbitrages), icon: ArrowLeftRight, color: "text-amber-400" },
    { label: "Wallet Clusters", value: formatNumber(stats.wallet_clusters), icon: Network, color: "text-violet-400" },
    { label: "Insider Alerts", value: formatNumber(stats.insider_alerts), icon: ShieldAlert, color: "text-red-400" },
    { label: "Markets Scored", value: formatNumber(stats.markets_scored), icon: Target, color: "text-emerald-400" },
    { label: "Avg Confidence", value: `${(stats.avg_confidence * 100).toFixed(0)}%`, icon: Gauge, color: "text-blue-400" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {cards.map((card) => (
        <Card key={card.label} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-2 mb-1">
              <card.icon className={`h-4 w-4 ${card.color}`} />
              <span className="text-xs text-muted-foreground">{card.label}</span>
            </div>
            <p className="text-xl font-bold">{card.value}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function AnalyticsStatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse mb-2" />
            <div className="h-6 w-16 bg-[#1e1e2e] rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
```

### 9.3 Component: `AnalyticsTabs` (`dashboard/src/components/analytics-tabs.tsx`)

Client component wrapping the four table components.

```typescript
"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CompositeSignalsTable } from "./composite-signals-table";
import { ArbitrageTable } from "./arbitrage-table";
import { WalletClustersTable } from "./wallet-clusters-table";
import { InsiderAlertsTable } from "./insider-alerts-table";
import type {
  CompositeSignal,
  ArbitrageOpportunity,
  WalletCluster,
  InsiderAlert,
} from "@/types/market";

interface AnalyticsTabsProps {
  composite: CompositeSignal[];
  arbitrage: ArbitrageOpportunity[];
  clusters: WalletCluster[];
  insider: InsiderAlert[];
}

export function AnalyticsTabs({
  composite,
  arbitrage,
  clusters,
  insider,
}: AnalyticsTabsProps) {
  return (
    <Tabs defaultValue="composite">
      <TabsList>
        <TabsTrigger value="composite">
          Composite Scores ({composite.length})
        </TabsTrigger>
        <TabsTrigger value="arbitrage">
          Arbitrage ({arbitrage.length})
        </TabsTrigger>
        <TabsTrigger value="clusters">
          Clusters ({clusters.length})
        </TabsTrigger>
        <TabsTrigger value="insider">
          Insider ({insider.length})
        </TabsTrigger>
      </TabsList>
      <TabsContent value="composite">
        <CompositeSignalsTable data={composite} />
      </TabsContent>
      <TabsContent value="arbitrage">
        <ArbitrageTable data={arbitrage} />
      </TabsContent>
      <TabsContent value="clusters">
        <WalletClustersTable data={clusters} />
      </TabsContent>
      <TabsContent value="insider">
        <InsiderAlertsTable data={insider} />
      </TabsContent>
    </Tabs>
  );
}
```

### 9.4 Component: `CompositeSignalsTable` (`dashboard/src/components/composite-signals-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Score | `score` | Colored bar: green (+), red (-), width = abs(score)% |
| Confidence | `confidence` | Progress bar 0-100% |
| OBI | `obi_score` | Small colored number |
| Volume | `volume_score` | Small colored number |
| Trades | `trade_bias_score` | Small colored number |
| Momentum | `momentum_score` | Small colored number |
| Smart Money | `smart_money_score` | Small colored number |
| Arb | `arbitrage_flag` | Badge: "Yes" (amber) or empty |
| Updated | `computed_at` | Relative time |

### 9.5 Component: `ArbitrageTable` (`dashboard/src/components/arbitrage-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Type | `arb_type` | Badge: "Sum" (blue) or "Related" (purple) |
| Expected | `expected_sum` | Fixed decimal |
| Actual | `actual_sum` | Fixed decimal |
| Spread | `spread` | Percentage (red if > 0.05) |
| Related Markets | `related_condition_ids.length` | Count badge |
| Description | `description` | Truncated text |
| Detected | `detected_at` | Relative time |

### 9.6 Component: `WalletClustersTable` (`dashboard/src/components/wallet-clusters-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Cluster | `cluster_id` | Truncated UUID |
| Wallets | `size` | Count badge |
| Similarity | `similarity_score` | Progress bar 0-100% |
| Timing | `timing_corr` | Percentage |
| Market Overlap | `market_overlap` | Percentage |
| Direction | `direction_agreement` | Percentage |
| Common Markets | `common_markets.length` | Count badge |
| Detected | `created_at` | Relative time |

### 9.7 Component: `InsiderAlertsTable` (`dashboard/src/components/insider-alerts-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Wallet | `pseudonym` + `profile_image` or truncated `proxy_wallet` | Avatar + name |
| Score | `score` | Colored progress bar (red > 70, amber > 50, yellow > 30) |
| Freshness | `freshness_score` | Small progress bar |
| Win Rate | `win_rate_score` | Small progress bar |
| Niche | `niche_score` | Small progress bar |
| Size | `size_score` | Small progress bar |
| Updated | `computed_at` | Relative time |

### 9.8 Enhanced `/signals` Page

Add a composite signal score column to the existing signals page. In `dashboard/src/app/signals/page.tsx`, fetch composite signals alongside existing data and pass to `SignalsTabs`:

**Changes to `SignalsContent`:**

```typescript
async function SignalsContent() {
  const [obi, volume, largeTrades, compositeSignals] = await Promise.all([
    getOrderBookImbalance(50),
    getVolumeAnomalies(30),
    getLargeTrades(1000, 50),
    getCompositeSignals(50),
  ]);
  return (
    <SignalsTabs
      obi={obi}
      volumeAnomalies={volume}
      largeTrades={largeTrades}
      compositeSignals={compositeSignals}
    />
  );
}
```

**Changes to `SignalsTabs` props:**

Add an optional `compositeSignals` prop and a new "Composite" tab as the first tab:

```typescript
interface SignalsTabsProps {
  obi: OBISignal[];
  volumeAnomalies: VolumeAnomaly[];
  largeTrades: LargeTrade[];
  compositeSignals?: CompositeSignal[];
}
```

The Composite tab reuses the `CompositeSignalsTable` component from the analytics page.

---

## 10. Navigation Update

Update `dashboard/src/app/layout.tsx` to add the `/analytics` route.

```typescript
import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Zap,
  Users,
  Brain,     // NEW
} from "lucide-react";

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/#markets", label: "Markets", icon: BarChart3 },
  { href: "/#trending", label: "Trending", icon: TrendingUp },
  { href: "/signals", label: "Signals", icon: Zap },
  { href: "/whales", label: "Whales", icon: Users },
  { href: "/analytics", label: "Analytics", icon: Brain },    // NEW
];
```

---

## 11. File-by-File Change List

### New Files (create)

| File | Type | Description |
|------|------|-------------|
| `pipeline/schema/003_phase3_analytics.sql` | SQL DDL | 4 new ClickHouse tables |
| `pipeline/jobs/arbitrage_scanner.py` | Python Job | 2-min arbitrage detection |
| `pipeline/jobs/wallet_analyzer.py` | Python Job | 30-min clustering + insider scoring |
| `pipeline/jobs/signal_compositor.py` | Python Job | 5-min composite signal computation |
| `dashboard/src/app/analytics/page.tsx` | Server Component | Analytics page |
| `dashboard/src/components/analytics-stats-cards.tsx` | Client Component | Stats cards |
| `dashboard/src/components/analytics-tabs.tsx` | Client Component | Tab wrapper |
| `dashboard/src/components/composite-signals-table.tsx` | Client Component | Composite signals table |
| `dashboard/src/components/arbitrage-table.tsx` | Client Component | Arbitrage opportunities table |
| `dashboard/src/components/wallet-clusters-table.tsx` | Client Component | Wallet clusters table |
| `dashboard/src/components/insider-alerts-table.tsx` | Client Component | Insider alerts table |

### Modified Files (edit)

| File | Changes |
|------|---------|
| `pipeline/config.py` | Add 10 new constants (3 intervals + 7 tuning params) |
| `pipeline/clickhouse_writer.py` | Add 4 new TABLE_COLUMNS entries + 4 convenience methods |
| `pipeline/scheduler.py` | Add 3 new job imports, 3 job registrations, 3 job wrappers, health check update |
| `pipeline/main.py` | Update migration runner to include `003_phase3_analytics.sql` |
| `dashboard/src/types/market.ts` | Add 5 new interfaces |
| `dashboard/src/lib/queries.ts` | Add 6 new query functions + import updates |
| `dashboard/src/app/layout.tsx` | Add `Brain` icon import + `/analytics` nav item |
| `dashboard/src/app/signals/page.tsx` | Add composite signals fetch + pass to SignalsTabs |
| `dashboard/src/components/signals-tabs.tsx` | Add optional compositeSignals prop + Composite tab |

### Unchanged Files

| File | Notes |
|------|-------|
| `pipeline/schema/001_init.sql` | No changes to existing tables |
| `pipeline/schema/002_phase2_users.sql` | No changes to Phase 2 tables |
| `pipeline/api/data_client.py` | No new API endpoints needed |
| `pipeline/api/clob_client.py` | No changes |
| `pipeline/api/gamma_client.py` | No changes |
| `pipeline/api/ws_client.py` | No changes |
| All Phase 1 pipeline jobs | No changes |
| All Phase 2 pipeline jobs | No changes |
| `dashboard/src/lib/clickhouse.ts` | Client singleton unchanged |
| `dashboard/src/lib/format.ts` | Existing formatters sufficient |
| All Phase 1 and Phase 2 components | No modifications needed (except signals-tabs.tsx) |

---

## 12. Design Decisions & Rationale

1. **Pipeline jobs with direct ClickHouse reads**: The Phase 3 jobs (arbitrage_scanner, wallet_analyzer, signal_compositor) read from ClickHouse tables populated by Phase 1 and Phase 2 jobs. They use `clickhouse_connect` to query data, compute analytics in Python, and write results back via the existing `ClickHouseWriter`. This avoids complex ClickHouse materialized views for multi-step computations.

2. **`_get_read_client()` pattern**: Phase 3 jobs need to READ from ClickHouse (unlike Phase 1/2 jobs that only WRITE). A module-level function creates a read client separate from the writer singleton. This is wrapped in `asyncio.to_thread()` since `clickhouse_connect` is synchronous.

3. **Greedy clustering, not DBSCAN**: The wallet clustering uses a simple pairwise greedy approach (O(n^2) capped at 200 wallets) rather than DBSCAN because: (a) the tracked wallet set is small (~500), (b) we want interpretable pair-level similarity scores, (c) DBSCAN requires tuning epsilon and min_samples. The cap at 200 wallets keeps comparisons under 20K pairs.

4. **Insider timing_score as placeholder (0.0)**: Full pre-announcement timing analysis requires correlating trade timestamps with market resolution events or external news events. This data is not yet available. The timing_score field and weight are reserved for future enhancement.

5. **Composite signal weights**: Smart money (0.25) and OBI (0.20) are weighted highest because they are the most differentiated signals. Volume anomaly (0.10) and insider (0.05) are weighted lowest because they are noisier. These weights can be tuned based on backtesting.

6. **ReplacingMergeTree for all Phase 3 tables**: All four tables store "current state" that gets updated periodically. Deduplication via FINAL is consistent with the existing pattern for `markets`, `trader_rankings`, etc.

7. **TTL on tables**: `arbitrage_opportunities` has 30-day TTL (arbs are short-lived). `wallet_clusters` and `insider_scores` have 90-day TTL. `composite_signals` has no TTL (current state only, one row per market).

8. **Enhanced /signals page**: Rather than moving all composite signals to /analytics only, we add a "Composite" tab to the existing /signals page. This keeps all signal types accessible from one location while /analytics provides the deep-dive views for arbitrage, clustering, and insider detection.

9. **`Brain` icon for Analytics**: Distinguishes from Signals (Zap) and Whales (Users). The Brain icon from lucide-react conveys "advanced analysis" without being too niche.

10. **No new API endpoints**: Phase 3 is entirely self-contained within the pipeline. All computations use data already being ingested by Phase 1 (market_trades, orderbook_snapshots, market_prices) and Phase 2 (wallet_activity, wallet_positions, trader_rankings, market_holders, trader_profiles). No new Polymarket API calls are needed.

---

## 13. Acceptance Criteria

- [ ] `003_phase3_analytics.sql` creates all 4 tables without errors on ClickHouse Cloud
- [ ] `docker compose up -d --build` starts pipeline with all 11 jobs (4 Phase 1 + 4 Phase 2 + 3 Phase 3)
- [ ] `curl http://localhost:8080/health` returns `phase3_jobs` in the response
- [ ] Arbitrage scanner detects sum-to-one deviations for binary markets
- [ ] Arbitrage scanner detects related-market inconsistencies for multi-outcome events
- [ ] Wallet analyzer produces clusters for wallets with synchronized trading
- [ ] Wallet analyzer produces insider scores for all tracked wallets
- [ ] Signal compositor produces composite scores for top 500 markets
- [ ] `/analytics` page loads with 5 stats cards showing live counts
- [ ] Composite Scores tab shows markets sorted by absolute score
- [ ] Arbitrage tab shows open opportunities sorted by spread
- [ ] Clusters tab shows detected wallet clusters with similarity metrics
- [ ] Insider tab shows wallets with score > 30 sorted by risk
- [ ] `/signals` page has new "Composite" tab with composite scores
- [ ] Sidebar navigation includes "Analytics" link pointing to `/analytics`
- [ ] `npm run build` succeeds with no type errors
- [ ] All existing pages (overview, signals, whales, market detail) are unaffected
- [ ] Pipeline handles empty data gracefully (no crashes when Phase 2 tables are empty)
