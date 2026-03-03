-- ============================================================
-- Polymarket ClickHouse Schema Migration 011 — Insider Trading Detection
-- Run automatically on startup after 010_online_learning.sql
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Insider Trade Signals (per-trade insider signal scoring)
-- ============================================================
CREATE TABLE IF NOT EXISTS insider_trade_signals
(
    -- Identity
    trade_id               String,                               -- Unique trade identifier
    condition_id           LowCardinality(String),               -- Market condition ID
    proxy_wallet           String,                               -- Wallet address

    -- Trade details
    side                   LowCardinality(String),               -- 'BUY' or 'SELL'
    size                   Float64 DEFAULT 0,                    -- Trade size in shares
    usdc_size              Float64 DEFAULT 0,                    -- Trade size in USDC
    price                  Float64 DEFAULT 0,                    -- Execution price
    trade_timestamp        DateTime64(3) CODEC(DoubleDelta, LZ4),

    -- Signal scores (0-100)
    pre_news_score         Float64 DEFAULT 0,                    -- Proximity to major price move / resolution
    statistical_score      Float64 DEFAULT 0,                    -- Z-score deviation from population norms
    profitability_score    Float64 DEFAULT 0,                    -- Risk-adjusted return anomaly
    coordination_score     Float64 DEFAULT 0,                    -- Coordinated trading pattern match
    composite_score        Float64 DEFAULT 0,                    -- Weighted composite (0-100)

    -- Context
    category               LowCardinality(String) DEFAULT '',    -- Market category (e.g., 'Middle East')
    event_slug             String DEFAULT '',                    -- Event group slug
    direction_correct      UInt8 DEFAULT 0,                      -- 1 if trade direction matched outcome
    hours_before_move      Float64 DEFAULT 0,                    -- Hours before major price move
    price_move_pct         Float64 DEFAULT 0,                    -- Magnitude of subsequent price move

    -- Timestamps
    scored_at              DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX composite_idx composite_score TYPE minmax GRANULARITY 4,
    INDEX category_idx category TYPE set(50) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_timestamp)
ORDER BY (condition_id, proxy_wallet, trade_timestamp)
TTL toDateTime(trade_timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Trader Suspicion Profiles (per-trader suspicion scoring)
-- ============================================================
CREATE TABLE IF NOT EXISTS trader_suspicion_profiles
(
    -- Identity
    proxy_wallet           String,                               -- Wallet address

    -- Composite suspicion score (0-100)
    suspicion_score        Float64 DEFAULT 0,                    -- Weighted composite
    suspicion_tier         LowCardinality(String) DEFAULT 'low', -- 'low', 'medium', 'high', 'critical'

    -- Component scores (0-100 each)
    pre_news_score         Float64 DEFAULT 0,                    -- Pre-news trading pattern score
    statistical_score      Float64 DEFAULT 0,                    -- Multi-metric z-score anomaly
    profitability_score    Float64 DEFAULT 0,                    -- Risk-adjusted return anomaly
    coordination_score     Float64 DEFAULT 0,                    -- Coordinated trading pattern score
    category_focus_score   Float64 DEFAULT 0,                    -- Concentration in sensitive categories

    -- Detailed breakdown (JSON for flexibility)
    factors                String DEFAULT '{}' CODEC(ZSTD(3)),   -- JSON breakdown of all factor details

    -- Statistics snapshot
    total_trades           UInt32 DEFAULT 0,                     -- Total trades analyzed
    win_rate               Float64 DEFAULT 0,                    -- Overall win rate
    avg_roi                Float64 DEFAULT 0,                    -- Average ROI per trade
    mideast_trade_pct      Float64 DEFAULT 0,                    -- % of trades in Middle East markets
    flagged_trade_count    UInt32 DEFAULT 0,                     -- Number of individually flagged trades
    total_pnl              Float64 DEFAULT 0,                    -- Total PnL across all markets

    -- Timestamps
    first_flagged_at       DateTime64(3) DEFAULT now64(3),
    computed_at            DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX score_idx suspicion_score TYPE minmax GRANULARITY 4,
    INDEX tier_idx suspicion_tier TYPE set(5) GRANULARITY 4
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (proxy_wallet)
TTL toDateTime(computed_at) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. Pre-News Events (major price moves and resolution events)
-- ============================================================
CREATE TABLE IF NOT EXISTS pre_news_events
(
    -- Identity
    event_id               String,                               -- UUID for the event
    condition_id           LowCardinality(String),               -- Market condition ID

    -- Event details
    event_type             LowCardinality(String),               -- 'price_move', 'resolution', 'volume_spike'
    magnitude              Float64 DEFAULT 0,                    -- Price change % or volume spike ratio
    direction              LowCardinality(String) DEFAULT '',    -- 'up', 'down'

    -- Price context
    price_before           Float64 DEFAULT 0,                    -- Price before event
    price_after            Float64 DEFAULT 0,                    -- Price after event
    volume_during          Float64 DEFAULT 0,                    -- Volume during event window

    -- Market context
    category               LowCardinality(String) DEFAULT '',    -- Market category
    event_slug             String DEFAULT '',                    -- Event group slug
    question               String DEFAULT '' CODEC(ZSTD(3)),     -- Market question text

    -- Window
    window_start           DateTime64(3) CODEC(DoubleDelta, LZ4),
    window_end             DateTime64(3) CODEC(DoubleDelta, LZ4),

    -- Timestamps
    detected_at            DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX type_idx event_type TYPE set(10) GRANULARITY 4,
    INDEX category_idx category TYPE set(50) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(detected_at)
ORDER BY (condition_id, event_type, detected_at)
TTL toDateTime(detected_at) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 4. Coordinated Trading Groups (wallet coordination clusters)
-- ============================================================
CREATE TABLE IF NOT EXISTS coordinated_trading_groups
(
    -- Identity
    group_id               String,                               -- UUID for the group

    -- Members
    wallets                Array(String),                        -- Array of proxy_wallet addresses
    size                   UInt32 DEFAULT 0,                     -- Number of wallets in group

    -- Coordination metrics
    correlation_score      Float64 DEFAULT 0,                    -- 0-1 coordination strength
    timing_correlation     Float64 DEFAULT 0,                    -- Trades within 5-min windows
    market_overlap         Float64 DEFAULT 0,                    -- Fraction of shared markets
    direction_agreement    Float64 DEFAULT 0,                    -- Fraction of same-direction trades
    size_similarity        Float64 DEFAULT 0,                    -- Trade size ratio similarity

    -- Context
    common_markets         Array(String),                        -- Markets where group trades together
    common_categories      Array(String),                        -- Categories group focuses on
    total_volume           Float64 DEFAULT 0,                    -- Combined group volume (USDC)
    avg_suspicion          Float64 DEFAULT 0,                    -- Average suspicion score of members
    label                  String DEFAULT '',                    -- Optional label (e.g., 'wash_trading')

    -- Timestamps
    detected_at            DateTime64(3) DEFAULT now64(3),
    updated_at             DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX group_idx group_id TYPE bloom_filter GRANULARITY 4,
    INDEX correlation_idx correlation_score TYPE minmax GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (group_id)
TTL toDateTime(detected_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;
