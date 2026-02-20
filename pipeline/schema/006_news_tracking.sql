-- ============================================================
-- Polymarket ClickHouse Schema Migration 006 — News & OSINT Tracking
-- Multi-source news ingestion for conflict market intelligence
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. News Articles (raw ingested articles from all sources)
-- ============================================================
CREATE TABLE IF NOT EXISTS news_articles
(
    -- Identity
    article_id         String,                               -- SHA256(url) or source-specific ID
    source             LowCardinality(String),               -- 'isw', 'deepstate', 'ukrinform', 'reuters', 'telegram', 'liveuamap', 'twitter'
    source_url         String CODEC(ZSTD(3)),                -- Original URL

    -- Content
    title              String CODEC(ZSTD(3)),
    body               String CODEC(ZSTD(3)),                -- Full text (or summary for paywalled)
    language           LowCardinality(String) DEFAULT 'en',

    -- Classification
    category           LowCardinality(String) DEFAULT '',     -- 'frontline', 'logistics', 'politics', 'weapons', 'casualties', 'negotiations'
    region             LowCardinality(String) DEFAULT '',     -- 'donbas', 'south', 'north', 'crimea', 'kursk', 'global'

    -- NLP-extracted signals
    sentiment          Float32 DEFAULT 0,                     -- -1 (bad for UA) to +1 (good for UA)
    urgency            Float32 DEFAULT 0,                     -- 0-1, how time-sensitive
    confidence         Float32 DEFAULT 0,                     -- 0-1, source reliability

    -- Entity extraction
    settlements_mentioned  Array(LowCardinality(String)),     -- Settlement IDs mentioned
    markets_mentioned      Array(LowCardinality(String)),     -- Polymarket condition_ids mentioned
    actors                 Array(LowCardinality(String)),     -- 'ua_army', 'ru_army', 'wagner', 'diplomats'

    -- Control change signals
    control_changes    String DEFAULT '{}' CODEC(ZSTD(3)),    -- JSON: [{settlement_id, old_control, new_control, confidence}]

    -- Timestamps
    published_at       DateTime64(3) CODEC(DoubleDelta, LZ4),
    ingested_at        DateTime64(3) DEFAULT now64(3),
    ts_date            Date MATERIALIZED toDate(published_at),

    -- Indexes
    INDEX source_idx source TYPE set(20) GRANULARITY 4,
    INDEX article_idx article_id TYPE bloom_filter GRANULARITY 4,
    INDEX category_idx category TYPE set(20) GRANULARITY 4,
    INDEX settlement_idx settlements_mentioned TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(published_at)
ORDER BY (source, published_at, article_id)
TTL toDateTime(published_at) + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Frontline State (timestamped settlement control snapshots)
-- ============================================================
CREATE TABLE IF NOT EXISTS frontline_state
(
    -- Identity
    settlement_id      LowCardinality(String),

    -- State
    control            LowCardinality(String),                -- 'UA', 'RU', 'CONTESTED'
    assault_intensity  Float32 DEFAULT 0,                     -- 0-1
    shelling_intensity Float32 DEFAULT 0,                     -- 0-1
    supply_disruption  Float32 DEFAULT 0,                     -- 0-1
    frontline_distance_km Float32 DEFAULT 50.0,

    -- Source
    source             LowCardinality(String) DEFAULT '',     -- 'deepstate', 'isw', 'manual', 'model'
    confidence         Float32 DEFAULT 0.5,

    -- Timestamp
    observed_at        DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX settlement_idx settlement_id TYPE bloom_filter GRANULARITY 4,
    INDEX control_idx control TYPE set(5) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(observed_at)
ORDER BY (settlement_id, observed_at)
TTL toDateTime(observed_at) + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. News Sentiment Aggregate (per-settlement, per-hour rollup)
-- ============================================================
CREATE TABLE IF NOT EXISTS news_sentiment_hourly
(
    settlement_id      LowCardinality(String),
    hour               DateTime CODEC(DoubleDelta, LZ4),

    -- Aggregates
    article_count      UInt32 DEFAULT 0,
    avg_sentiment      Float32 DEFAULT 0,
    max_urgency        Float32 DEFAULT 0,
    source_diversity   UInt8 DEFAULT 0,                       -- Number of distinct sources

    -- Weighted sentiment (source reliability × sentiment)
    weighted_sentiment Float32 DEFAULT 0,

    -- Velocity
    news_velocity      Float32 DEFAULT 0,                     -- articles_this_hour / avg_articles_per_hour

    INDEX settlement_idx settlement_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = SummingMergeTree()
ORDER BY (settlement_id, hour)
TTL hour + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 4. Market Microstructure Snapshots (enhanced tick data)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_microstructure
(
    condition_id       LowCardinality(String),

    -- Spread dynamics
    bid_ask_spread     Float64 DEFAULT 0,
    effective_spread   Float64 DEFAULT 0,                     -- Actual execution spread vs mid
    realized_spread    Float64 DEFAULT 0,                     -- Post-trade price impact

    -- Depth metrics
    bid_depth_1        Float64 DEFAULT 0,                     -- Top-of-book bid depth
    ask_depth_1        Float64 DEFAULT 0,
    bid_depth_5        Float64 DEFAULT 0,                     -- Top-5 levels
    ask_depth_5        Float64 DEFAULT 0,
    obi                Float64 DEFAULT 0,                     -- Order book imbalance
    depth_ratio        Float64 DEFAULT 0,                     -- Spoof detection: top5/top1

    -- Trade flow
    buy_volume_5m      Float64 DEFAULT 0,                     -- 5-min buy volume
    sell_volume_5m     Float64 DEFAULT 0,
    trade_count_5m     UInt32 DEFAULT 0,
    large_trade_count_5m UInt32 DEFAULT 0,                    -- Trades > $1K
    vwap_5m            Float64 DEFAULT 0,
    kyle_lambda        Float64 DEFAULT 0,                     -- Price impact per $1 traded

    -- Toxicity (adverse selection)
    toxic_flow_ratio   Float64 DEFAULT 0,                     -- Informed vs uninformed order ratio
    price_impact_1m    Float64 DEFAULT 0,                     -- Price change 1 min after trade

    -- Liquidity resilience
    spread_after_trade Float64 DEFAULT 0,                     -- Spread recovery after large trade
    depth_recovery_sec Float64 DEFAULT 0,                     -- Time to restore depth after fill

    -- Timestamp
    snapshot_time      DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(snapshot_time)
ORDER BY (condition_id, snapshot_time)
TTL toDateTime(snapshot_time) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity = 8192;
