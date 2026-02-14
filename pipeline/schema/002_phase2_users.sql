-- ============================================================
-- Polymarket ClickHouse Schema Migration 002 — Phase 2 User/Wallet Data
-- Run automatically on startup after 001_init.sql
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Trader Rankings (leaderboard snapshots)
-- ============================================================
CREATE TABLE IF NOT EXISTS trader_rankings
(
    -- Identity
    proxy_wallet       String,                            -- 0x-prefixed wallet address
    user_name          String DEFAULT '',                 -- Display name (pseudonym)
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),  -- Profile image URL

    -- Ranking
    rank               UInt32,                            -- Leaderboard position
    category           LowCardinality(String),            -- OVERALL, POLITICS, SPORTS, CRYPTO, etc.
    time_period        LowCardinality(String),            -- DAY, WEEK, MONTH, ALL
    order_by           LowCardinality(String),            -- PNL or VOL

    -- Metrics
    pnl                Float64 DEFAULT 0,                 -- Profit/loss for the period
    volume             Float64 DEFAULT 0,                 -- Trading volume for the period

    -- Profile flags
    verified_badge     UInt8 DEFAULT 0,                   -- 1 = verified
    x_username         String DEFAULT '',                 -- X/Twitter handle

    -- Timestamp
    snapshot_time      DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX category_idx category TYPE set(20) GRANULARITY 4
)
ENGINE = ReplacingMergeTree(snapshot_time)
ORDER BY (proxy_wallet, category, time_period, order_by)
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Market Holders (top holders per market)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_holders
(
    -- Market identity
    condition_id       LowCardinality(String),            -- FK to markets
    token_id           String DEFAULT '',                  -- Token ID

    -- Holder identity
    proxy_wallet       String,                             -- Holder wallet address
    pseudonym          String DEFAULT '',                  -- Display name
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),   -- Profile image URL
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Holdings
    amount             Float64 DEFAULT 0,                  -- Token balance

    -- Timestamp
    snapshot_time      DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(snapshot_time)
ORDER BY (condition_id, proxy_wallet, outcome_index)
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. Wallet Positions (tracked wallet open positions)
-- ============================================================
CREATE TABLE IF NOT EXISTS wallet_positions
(
    -- Identity
    proxy_wallet       String,                             -- Wallet address
    condition_id       LowCardinality(String),             -- Market condition ID
    asset              String DEFAULT '',                   -- Token ID
    outcome            LowCardinality(String),             -- 'Yes' / 'No'
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Position data
    size               Float64 DEFAULT 0,                  -- Number of tokens held
    avg_price          Float64 DEFAULT 0,                  -- Average entry price
    initial_value      Float64 DEFAULT 0,                  -- Cost basis (USDC)
    current_value      Float64 DEFAULT 0,                  -- Mark-to-market value (USDC)
    cur_price          Float64 DEFAULT 0,                  -- Current market price

    -- PnL
    cash_pnl           Float64 DEFAULT 0,                  -- Unrealized PnL (USDC)
    percent_pnl        Float64 DEFAULT 0,                  -- Percent return
    realized_pnl       Float64 DEFAULT 0,                  -- Realized PnL (USDC)

    -- Market metadata (denormalized for query convenience)
    title              String DEFAULT '',                   -- Market question
    market_slug        String DEFAULT '',                   -- URL slug
    end_date           DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),

    -- Timestamp
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (proxy_wallet, condition_id, outcome)
SETTINGS index_granularity = 8192;

-- ============================================================
-- 4. Wallet Activity (trade/activity history, append-only)
-- ============================================================
CREATE TABLE IF NOT EXISTS wallet_activity
(
    -- Identity
    proxy_wallet       String,                             -- Wallet address
    condition_id       LowCardinality(String),             -- Market condition ID
    asset              String DEFAULT '',                   -- Token ID

    -- Activity data
    activity_type      LowCardinality(String),             -- TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE
    side               LowCardinality(String) DEFAULT '',  -- BUY or SELL (for trades)
    outcome            LowCardinality(String) DEFAULT '',  -- Outcome name
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Amounts
    size               Float64 DEFAULT 0,                  -- Token amount
    usdc_size          Float64 DEFAULT 0,                  -- USDC equivalent
    price              Float64 DEFAULT 0,                  -- Execution price

    -- On-chain
    transaction_hash   String DEFAULT '',                   -- Polygon tx hash

    -- Market metadata (denormalized)
    title              String DEFAULT '',                   -- Market question
    market_slug        String DEFAULT '',                   -- URL slug

    -- Timestamp
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX tx_idx transaction_hash TYPE bloom_filter GRANULARITY 4,
    INDEX type_idx activity_type TYPE set(10) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (proxy_wallet, timestamp, condition_id)
TTL timestamp + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 5. Trader Profiles (wallet profile data)
-- ============================================================
CREATE TABLE IF NOT EXISTS trader_profiles
(
    -- Identity
    proxy_wallet       String,                             -- Primary key
    pseudonym          String DEFAULT '',                   -- Display name
    name               String DEFAULT '',                   -- Real name (if set)
    bio                String DEFAULT '' CODEC(ZSTD(3)),    -- User bio
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),    -- Profile image URL

    -- Social
    x_username         String DEFAULT '',                   -- X/Twitter handle
    verified_badge     UInt8 DEFAULT 0,                    -- 1 = verified

    -- Metadata
    display_username_public UInt8 DEFAULT 0,               -- 1 = username is public
    profile_created_at DateTime64(3) DEFAULT toDateTime64('1970-01-01', 3),

    -- Tracking metadata
    discovered_via     LowCardinality(String) DEFAULT '',  -- 'leaderboard', 'holders', 'manual'
    first_seen_at      DateTime64(3) DEFAULT now64(3),     -- When we first discovered this wallet
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX pseudonym_idx pseudonym TYPE tokenbf_v1(1024, 3, 0) GRANULARITY 4,
    INDEX x_idx x_username TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (proxy_wallet)
SETTINGS index_granularity = 8192;
