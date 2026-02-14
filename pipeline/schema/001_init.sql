-- ============================================================
-- Polymarket ClickHouse Schema Migration 001
-- Run against ClickHouse Cloud instance
-- ============================================================

CREATE DATABASE IF NOT EXISTS polymarket;

USE polymarket;

-- ============================================================
-- 1. Markets (metadata, ReplacingMergeTree)
-- ============================================================
CREATE TABLE IF NOT EXISTS markets
(
    -- Identity
    condition_id       String,                          -- Polymarket condition ID (primary identifier)
    market_slug        String,                          -- URL-friendly slug
    question           String,                          -- Full market question text
    description        String CODEC(ZSTD(3)),           -- Longer description, high compression

    -- Event grouping
    event_id           String DEFAULT '',               -- Polymarket event group ID
    event_title        String DEFAULT '',               -- Event group title
    event_slug         String DEFAULT '',               -- Event group slug
    neg_risk           UInt8 DEFAULT 0,                 -- 1 = negative-risk market

    -- Classification
    category           LowCardinality(String),          -- e.g., 'politics', 'sports', 'crypto'
    tags               Array(LowCardinality(String)),   -- Searchable tags

    -- Outcomes
    outcomes           Array(String),                   -- ['Yes', 'No'] or multi-outcome
    outcome_prices     Array(Float64),                  -- Current prices (probabilities)
    token_ids          Array(String),                   -- CLOB token IDs per outcome

    -- Status
    active             UInt8,                           -- 1 = open for trading
    closed             UInt8,                           -- 1 = market closed
    resolved           UInt8,                           -- 1 = resolved with winner
    resolution_source  String DEFAULT '',               -- URL or description of resolution source
    winning_outcome    String DEFAULT '',               -- Which outcome won

    -- Liquidity & volume
    volume_24h         Float64 DEFAULT 0,
    volume_total       Float64 DEFAULT 0,
    liquidity          Float64 DEFAULT 0,
    volume_1wk         Float64 DEFAULT 0,               -- Rolling 1-week volume
    volume_1mo         Float64 DEFAULT 0,               -- Rolling 1-month volume

    -- Competition & price changes
    competitive_score  Float64 DEFAULT 0,               -- How competitive the market is (0-1)
    one_day_price_change  Float64 DEFAULT 0,            -- 24h price change
    one_week_price_change Float64 DEFAULT 0,            -- 7d price change

    -- Timestamps
    start_date         DateTime64(3) DEFAULT now64(3),
    end_date           DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),
    created_at         DateTime64(3) DEFAULT now64(3),
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX question_idx question TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 4,
    INDEX slug_idx     market_slug TYPE bloom_filter GRANULARITY 4,
    INDEX event_id_idx event_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id)
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Market Prices (tick-level time series)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_prices
(
    -- Identifiers
    condition_id       LowCardinality(String),          -- FK to markets
    token_id           String,                          -- Specific outcome token
    outcome            LowCardinality(String),          -- 'Yes' / 'No' / named outcome

    -- Price data
    price              Float64 CODEC(Gorilla, LZ4),     -- 0.0 to 1.0 (probability)
    bid                Float64 DEFAULT 0 CODEC(Gorilla, LZ4),
    ask                Float64 DEFAULT 0 CODEC(Gorilla, LZ4),
    spread             Float64 MATERIALIZED ask - bid,  -- Computed spread

    -- Volume
    volume             Float64 DEFAULT 0 CODEC(Gorilla, LZ4),

    -- Timestamp
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, outcome, timestamp)
TTL timestamp + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. Market Trades
-- ============================================================
CREATE TABLE IF NOT EXISTS market_trades
(
    -- Identifiers
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),

    -- Trade data
    price              Float64 CODEC(Gorilla, LZ4),
    size               Float64 CODEC(Gorilla, LZ4),
    side               Enum8('buy' = 1, 'sell' = 2),
    trade_id           String,

    -- Timestamp
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX trade_idx     trade_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, outcome, timestamp, trade_id)
TTL timestamp + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 4. Orderbook Snapshots
-- ============================================================
CREATE TABLE IF NOT EXISTS orderbook_snapshots
(
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),

    -- L2 order book
    bid_prices         Array(Float64),
    bid_sizes          Array(Float64),
    ask_prices         Array(Float64),
    ask_sizes          Array(Float64),

    -- Derived
    best_bid           Float64 MATERIALIZED if(length(bid_prices) > 0, bid_prices[1], 0),
    best_ask           Float64 MATERIALIZED if(length(ask_prices) > 0, ask_prices[1], 0),
    mid_price          Float64 MATERIALIZED (best_bid + best_ask) / 2,
    total_bid_depth    Float64 MATERIALIZED arraySum(bid_sizes),
    total_ask_depth    Float64 MATERIALIZED arraySum(ask_sizes),

    -- Timestamp
    snapshot_time      DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(snapshot_time)
ORDER BY (condition_id, outcome, snapshot_time)
TTL snapshot_time + INTERVAL 7 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 5. Market Events (resolution, status changes)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_events
(
    condition_id       LowCardinality(String),
    event_type         LowCardinality(String),          -- 'created', 'resolved', 'closed', 'liquidity_change'
    event_data         String CODEC(ZSTD(1)),            -- JSON payload with event details
    event_time         DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX event_type_idx event_type TYPE set(20) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (condition_id, event_type, event_time)
SETTINGS index_granularity = 8192;

-- ============================================================
-- 6. OHLCV 1-Minute (AggregatingMergeTree target)
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_1m
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    bar_time           DateTime CODEC(DoubleDelta, LZ4),

    -- OHLCV as aggregate function states
    open               AggregateFunction(argMin, Float64, DateTime64(3)),
    high               AggregateFunction(max, Float64),
    low                AggregateFunction(min, Float64),
    close              AggregateFunction(argMax, Float64, DateTime64(3)),
    volume             AggregateFunction(sum, Float64),
    trade_count        AggregateFunction(count),
    buy_volume         AggregateFunction(sumIf, Float64, UInt8),
    sell_volume        AggregateFunction(sumIf, Float64, UInt8)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(bar_time)
ORDER BY (condition_id, outcome, bar_time);

-- ============================================================
-- 7. OHLCV 1-Minute Materialized View (from market_trades)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m_mv
TO ohlcv_1m
AS SELECT
    condition_id,
    outcome,
    toStartOfMinute(timestamp) AS bar_time,
    argMinState(price, timestamp) AS open,
    maxState(price) AS high,
    minState(price) AS low,
    argMaxState(price, timestamp) AS close,
    sumState(size) AS volume,
    countState() AS trade_count,
    sumIfState(size, side = 'buy') AS buy_volume,
    sumIfState(size, side = 'sell') AS sell_volume
FROM market_trades
GROUP BY condition_id, outcome, bar_time;

-- ============================================================
-- 8. OHLCV 1-Hour (AggregatingMergeTree target)
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_1h
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    bar_time           DateTime CODEC(DoubleDelta, LZ4),

    open               AggregateFunction(argMin, Float64, DateTime64(3)),
    high               AggregateFunction(max, Float64),
    low                AggregateFunction(min, Float64),
    close              AggregateFunction(argMax, Float64, DateTime64(3)),
    volume             AggregateFunction(sum, Float64),
    trade_count        AggregateFunction(count)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(bar_time)
ORDER BY (condition_id, outcome, bar_time);

-- ============================================================
-- 9. OHLCV 1-Hour Materialized View (directly from market_trades)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h_mv
TO ohlcv_1h
AS SELECT
    condition_id,
    outcome,
    toStartOfHour(timestamp) AS bar_time,
    argMinState(price, timestamp) AS open,
    maxState(price) AS high,
    minState(price) AS low,
    argMaxState(price, timestamp) AS close,
    sumState(size) AS volume,
    countState() AS trade_count
FROM market_trades
GROUP BY condition_id, outcome, bar_time;

-- ============================================================
-- 10. Daily Volume Rollup (SummingMergeTree)
-- ============================================================
CREATE TABLE IF NOT EXISTS volume_daily
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    trade_date         Date,
    total_volume       Float64,
    trade_count        UInt64,
    buy_volume         Float64,
    sell_volume        Float64,
    vwap               Float64                          -- Volume-weighted average price
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (condition_id, outcome, trade_date);

-- ============================================================
-- 11. Daily Volume Materialized View
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS volume_daily_mv
TO volume_daily
AS SELECT
    condition_id,
    outcome,
    toDate(timestamp) AS trade_date,
    sum(size) AS total_volume,
    count() AS trade_count,
    sumIf(size, side = 'buy') AS buy_volume,
    sumIf(size, side = 'sell') AS sell_volume,
    sum(price * size) / sum(size) AS vwap
FROM market_trades
GROUP BY condition_id, outcome, trade_date;

-- ============================================================
-- 12. Latest Price Tracker (ReplacingMergeTree)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_latest_price
(
    condition_id       String,
    outcome            LowCardinality(String),
    price              Float64 CODEC(Gorilla, LZ4),
    volume_24h         Float64,
    updated_at         DateTime64(3) CODEC(DoubleDelta, LZ4)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id, outcome);

-- ============================================================
-- 13. Latest Price Materialized View
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS market_latest_price_mv
TO market_latest_price
AS SELECT
    condition_id,
    outcome,
    price,
    0 AS volume_24h,    -- Updated separately or via scheduled query
    timestamp AS updated_at
FROM market_prices;
