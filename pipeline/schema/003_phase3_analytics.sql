-- ============================================================
-- Polymarket ClickHouse Schema Migration 003 — Phase 3 Advanced Analytics
-- Run automatically on startup after 002_phase2_users.sql
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Arbitrage Opportunities (detected pricing inconsistencies)
-- ============================================================
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
TTL toDateTime(detected_at) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Wallet Clusters (grouped wallets with synchronized behavior)
-- ============================================================
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
TTL toDateTime(created_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. Insider Scores (per-wallet insider risk score)
-- ============================================================
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
TTL toDateTime(computed_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 4. Composite Signals (per-market multi-factor signal)
-- ============================================================
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
