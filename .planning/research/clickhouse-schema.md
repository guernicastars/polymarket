# ClickHouse Cloud Schema Design for Prediction Market Time-Series Data

## Table of Contents
1. [Table Engine Selection](#1-table-engine-selection)
2. [Partitioning Strategy](#2-partitioning-strategy)
3. [Core Schema: CREATE TABLE Statements](#3-core-schema-create-table-statements)
4. [Materialized Views for Real-Time Aggregations](#4-materialized-views-for-real-time-aggregations)
5. [ClickHouse Cloud Connection Patterns](#5-clickhouse-cloud-connection-patterns)
6. [Python Pipeline Insert Patterns](#6-python-pipeline-insert-patterns)
7. [TTL Policies for Data Retention](#7-ttl-policies-for-data-retention)
8. [Dictionary / Metadata Tables](#8-dictionary--metadata-tables)
9. [Dashboard Query Patterns](#9-dashboard-query-patterns)
10. [Codec & Compression Recommendations](#10-codec--compression-recommendations)

---

## 1. Table Engine Selection

### Recommended Engines for Prediction Market Data

| Table | Engine | Rationale |
|-------|--------|-----------|
| `market_prices` (tick-level) | **MergeTree** | Raw append-only price observations; highest write throughput |
| `markets` (metadata) | **ReplacingMergeTree** | Market metadata changes over time (status, description); deduplicates on merge keeping latest version |
| `ohlcv_1m` / `ohlcv_1h` | **AggregatingMergeTree** | Pre-aggregated candle data from materialized views; merges partial aggregates correctly |
| `volume_daily` | **SummingMergeTree** | Daily volume rollups; auto-sums numeric columns on merge |
| `market_events` | **MergeTree** | Append-only event log (resolutions, new markets, liquidity changes) |
| `orderbook_snapshots` | **MergeTree** with TTL | Short-lived L2 data; auto-expires after retention window |

### Why MergeTree Family

- **Columnar storage** with 10:1 to 30:1 compression on market data
- **Sparse primary index** enables skipping entire granules during scans
- **Background merges** handle deduplication and aggregation asynchronously
- **Partition pruning** eliminates irrelevant time ranges instantly
- ClickHouse Cloud manages replication automatically (no need for ReplicatedMergeTree prefix)

### ReplacingMergeTree for Market Metadata

```sql
-- Deduplicates rows with same ORDER BY key, keeping highest `version`
-- Use FINAL in queries or enable `do_not_merge_across_partitions_select_final = 1`
ENGINE = ReplacingMergeTree(updated_at)
```

Key gotcha: deduplication happens during **background merges**, not at insert time. Always use `FINAL` modifier or `argMax` pattern in queries to get the latest state.

### AggregatingMergeTree for Pre-Aggregated Views

```sql
-- Stores partial aggregate states (AggregateFunction types)
-- Background merges combine partial states correctly
ENGINE = AggregatingMergeTree()
```

The State/Merge pattern:
- **Insert**: Use `-State` suffix functions (e.g., `avgState(price)`)
- **Query**: Use `-Merge` suffix functions (e.g., `avgMerge(avg_price)`)

---

## 2. Partitioning Strategy

### Guidelines

- Partition size target: **1-300 GB per partition** (for MergeTree)
- For smaller tables (< 10 GB): **skip partitioning entirely**
- Keep partition count in the **dozens to low hundreds**, not thousands
- Align partition boundaries with TTL retention for efficient `DROP PARTITION`
- ClickHouse can only prune partitions when the partition key appears in WHERE

### Recommended Partitioning

| Table | Partition Key | Rationale |
|-------|--------------|-----------|
| `market_prices` | `toYYYYMM(timestamp)` | Monthly; balances partition count vs. size for years of tick data |
| `markets` | None | Small table (thousands of rows); no benefit from partitioning |
| `ohlcv_1m` | `toYYYYMM(bar_time)` | Monthly candles |
| `orderbook_snapshots` | `toYYYYMMDD(snapshot_time)` | Daily; short retention (7 days) matches daily drop |
| `market_events` | `toYYYYMM(event_time)` | Monthly |

### Anti-Patterns to Avoid

- **Too many partitions** (e.g., per-hour for low-volume data): causes excessive parts, slower inserts
- **High-cardinality partition keys** (e.g., `market_id`): thousands of partitions
- **Cross-partition queries**: scanning many partitions can be slower than no partitioning

---

## 3. Core Schema: CREATE TABLE Statements

### 3.1 Markets (Metadata)

```sql
CREATE TABLE IF NOT EXISTS markets
(
    -- Identity
    condition_id       String,                          -- Polymarket condition ID (primary identifier)
    market_slug        String,                          -- URL-friendly slug
    question           String,                          -- Full market question text
    description        String CODEC(ZSTD(3)),           -- Longer description, high compression

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

    -- Timestamps
    start_date         DateTime64(3) DEFAULT now64(3),
    end_date           DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),
    created_at         DateTime64(3) DEFAULT now64(3),
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX question_idx question TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 4,
    INDEX slug_idx     market_slug TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id)
SETTINGS index_granularity = 8192;
```

### 3.2 Market Prices (Tick-Level Time Series)

```sql
CREATE TABLE IF NOT EXISTS market_prices
(
    -- Identifiers
    condition_id       LowCardinality(String),          -- FK to markets
    token_id           String,                          -- Specific outcome token
    outcome            LowCardinality(String),          -- 'Yes' / 'No' / named outcome

    -- Price data
    price              Float64,                         -- 0.0 to 1.0 (probability)
    bid                Float64 DEFAULT 0,               -- Best bid
    ask                Float64 DEFAULT 0,               -- Best ask
    spread             Float64 MATERIALIZED ask - bid,  -- Computed spread

    -- Volume
    volume             Float64 DEFAULT 0,               -- Volume in this tick/interval

    -- Timestamp
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),  -- Millisecond precision
    ts_date            Date MATERIALIZED toDate(timestamp),     -- For partition pruning

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, outcome, timestamp)
TTL timestamp + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;
```

### 3.3 Market Trades

```sql
CREATE TABLE IF NOT EXISTS market_trades
(
    -- Identifiers
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),

    -- Trade data
    price              Float64,
    size               Float64,                          -- Trade size in tokens
    side               Enum8('buy' = 1, 'sell' = 2),    -- Buy or sell
    trade_id           String,                           -- Unique trade identifier

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
```

### 3.4 Orderbook Snapshots

```sql
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
```

### 3.5 Market Events (Resolution, Status Changes)

```sql
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
```

---

## 4. Materialized Views for Real-Time Aggregations

### 4.1 OHLCV 1-Minute Candles

#### Target Table

```sql
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
```

#### Materialized View (from market_trades)

```sql
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
```

#### Query Pattern

```sql
SELECT
    condition_id,
    outcome,
    bar_time,
    argMinMerge(open) AS open,
    maxMerge(high) AS high,
    minMerge(low) AS low,
    argMaxMerge(close) AS close,
    sumMerge(volume) AS volume,
    countMerge(trade_count) AS trade_count
FROM ohlcv_1m
WHERE condition_id = '{condition_id}'
  AND outcome = 'Yes'
  AND bar_time >= now() - INTERVAL 24 HOUR
GROUP BY condition_id, outcome, bar_time
ORDER BY bar_time;
```

### 4.2 OHLCV 1-Hour Candles

#### Target Table

```sql
CREATE TABLE IF NOT EXISTS ohlcv_1h
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    bar_time           DateTime CODEC(DoubleDelta, LZ4),

    open               AggregateFunction(argMin, Float64, DateTime),
    high               AggregateFunction(max, Float64),
    low                AggregateFunction(min, Float64),
    close              AggregateFunction(argMax, Float64, DateTime),
    volume             AggregateFunction(sum, Float64),
    trade_count        AggregateFunction(sum, UInt64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(bar_time)
ORDER BY (condition_id, outcome, bar_time);
```

#### Materialized View (cascading from 1m candles)

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h_mv
TO ohlcv_1h
AS SELECT
    condition_id,
    outcome,
    toStartOfHour(bar_time) AS bar_time,
    argMinState(argMinMerge(open), bar_time) AS open,
    maxState(maxMerge(high)) AS high,
    minState(minMerge(low)) AS low,
    argMaxState(argMaxMerge(close), bar_time) AS close,
    sumState(sumMerge(volume)) AS volume,
    sumState(countMerge(trade_count)) AS trade_count
FROM ohlcv_1m
GROUP BY condition_id, outcome, toStartOfHour(bar_time);
```

**Note**: Cascading materialized views (1m -> 1h) may have edge cases. An alternative is to create the 1h view directly from `market_trades`:

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h_direct_mv
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
```

### 4.3 Daily Volume Rollup

```sql
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
```

### 4.4 Price Change Tracker (Latest Price per Market)

```sql
CREATE TABLE IF NOT EXISTS market_latest_price
(
    condition_id       String,
    outcome            LowCardinality(String),
    price              Float64,
    volume_24h         Float64,
    updated_at         DateTime64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id, outcome);

CREATE MATERIALIZED VIEW IF NOT EXISTS market_latest_price_mv
TO market_latest_price
AS SELECT
    condition_id,
    outcome,
    price,
    0 AS volume_24h,    -- Updated separately or via scheduled query
    timestamp AS updated_at
FROM market_prices;
```

---

## 5. ClickHouse Cloud Connection Patterns

### HTTP Interface (Recommended for Cloud)

ClickHouse Cloud exposes **HTTPS on port 8443**. The `clickhouse-connect` library uses HTTP under the hood, making it the natural choice for Cloud deployments.

```python
import clickhouse_connect

# ClickHouse Cloud connection
client = clickhouse_connect.get_client(
    host='your-instance.clickhouse.cloud',
    port=8443,
    username='default',
    password='your-password',
    secure=True,                    # Required for Cloud (TLS)
    connect_timeout=30,
    send_receive_timeout=300,       # 5 min for large queries
    compress='lz4',                 # Compress data in transit
)
```

### Native Protocol (Alternative)

For highest raw throughput, `clickhouse-driver` uses the native TCP protocol (port 9440 on Cloud with TLS):

```python
from clickhouse_driver import Client

client = Client(
    host='your-instance.clickhouse.cloud',
    port=9440,
    user='default',
    password='your-password',
    secure=True,
    verify=True,
    compression='lz4',
)
```

### Recommendation for This Project

Use **`clickhouse-connect`** (HTTP interface) because:
- Official ClickHouse-maintained driver
- Works through firewalls and proxies (port 443/8443)
- Built-in DataFrame support (`query_df()`, `insert_df()`)
- Async wrapper available for concurrent operations
- Connection pooling works automatically
- ClickHouse Cloud primarily optimized for HTTP interface

### Environment Variables Pattern

```python
import os
import clickhouse_connect

def get_clickhouse_client():
    """Create a ClickHouse Cloud client from environment variables."""
    return clickhouse_connect.get_client(
        host=os.environ['CLICKHOUSE_HOST'],
        port=int(os.environ.get('CLICKHOUSE_PORT', '8443')),
        username=os.environ.get('CLICKHOUSE_USER', 'default'),
        password=os.environ['CLICKHOUSE_PASSWORD'],
        database=os.environ.get('CLICKHOUSE_DATABASE', 'polymarket'),
        secure=True,
        compress='lz4',
        connect_timeout=30,
        send_receive_timeout=300,
    )
```

---

## 6. Python Pipeline Insert Patterns

### Batch Insert (Recommended)

```python
import clickhouse_connect
from datetime import datetime

client = get_clickhouse_client()

# Collect rows in memory, insert in batches
BATCH_SIZE = 50_000  # 10K-100K rows optimal

def insert_price_batch(rows: list[dict]):
    """Insert a batch of price observations.

    Each row: {condition_id, token_id, outcome, price, bid, ask, volume, timestamp}
    """
    data = [
        [
            r['condition_id'],
            r['token_id'],
            r['outcome'],
            r['price'],
            r.get('bid', 0.0),
            r.get('ask', 0.0),
            r.get('volume', 0.0),
            r['timestamp'],
        ]
        for r in rows
    ]

    client.insert(
        'market_prices',
        data,
        column_names=[
            'condition_id', 'token_id', 'outcome',
            'price', 'bid', 'ask', 'volume', 'timestamp',
        ],
    )
```

### DataFrame Insert (for Pandas Users)

```python
import pandas as pd

def insert_prices_df(df: pd.DataFrame):
    """Insert a DataFrame of price data.

    DataFrame columns must match table columns.
    """
    client.insert_df(
        'market_prices',
        df,
        column_names=[
            'condition_id', 'token_id', 'outcome',
            'price', 'bid', 'ask', 'volume', 'timestamp',
        ],
    )
```

### Async Insert Pattern (for Real-Time Pipeline)

```python
import asyncio
import clickhouse_connect

async def async_insert_prices(rows: list[dict]):
    """Non-blocking insert for async pipelines."""
    client = await clickhouse_connect.get_async_client(
        host=os.environ['CLICKHOUSE_HOST'],
        port=8443,
        username='default',
        password=os.environ['CLICKHOUSE_PASSWORD'],
        secure=True,
    )

    data = [[r['condition_id'], r['outcome'], r['price'], r['timestamp']] for r in rows]
    await client.insert('market_prices', data,
                        column_names=['condition_id', 'outcome', 'price', 'timestamp'])
```

### Buffer Table Pattern (for High-Frequency Inserts)

If the pipeline sends very frequent small batches (< 1000 rows), use a Buffer table to accumulate inserts and flush to the main table:

```sql
CREATE TABLE market_prices_buffer AS market_prices
ENGINE = Buffer(
    polymarket,           -- database
    market_prices,        -- destination table
    16,                   -- num_layers (parallelism)
    10, 100,              -- min/max time (seconds) before flush
    10000, 100000,        -- min/max rows before flush
    10000000, 100000000   -- min/max bytes before flush
);
```

The Python pipeline writes to `market_prices_buffer`; ClickHouse automatically flushes to `market_prices`.

### Insert Best Practices Summary

| Practice | Recommendation |
|----------|---------------|
| Batch size | 10,000 - 100,000 rows per INSERT |
| Insert frequency | No more than 1 insert/second per table |
| Compression | LZ4 for local, ZSTD for remote/cloud |
| Format | Native columnar (default in clickhouse-connect) |
| Async | Use async client for concurrent pipeline steps |
| Buffer tables | For sub-1000 row micro-batches |
| Deduplication | ClickHouse deduplicates identical inserts within 100 blocks by default |

---

## 7. TTL Policies for Data Retention

### Strategy for Prediction Market Data

| Table | Retention | Rationale |
|-------|-----------|-----------|
| `market_prices` | 2 years | Core historical data; needed for backtesting |
| `market_trades` | 2 years | Trade-level history |
| `orderbook_snapshots` | 7 days | High volume, low long-term value |
| `ohlcv_1m` | 6 months | Granular candles; roll up to 1h after 6 months |
| `ohlcv_1h` | 2 years | Coarser candles for long-term charts |
| `volume_daily` | Forever | Small table, high analytical value |
| `markets` | Forever | Metadata; small table |
| `market_events` | Forever | Audit trail |

### TTL Implementation

TTL is declared in the CREATE TABLE (shown above in schema definitions). To modify existing tables:

```sql
-- Add TTL to existing table
ALTER TABLE market_prices MODIFY TTL timestamp + INTERVAL 2 YEAR DELETE;

-- Conditional TTL: keep resolved market data longer
ALTER TABLE market_events
    MODIFY TTL event_time + INTERVAL 1 YEAR WHERE event_type = 'liquidity_change',
                event_time + INTERVAL 5 YEAR WHERE event_type IN ('resolved', 'created');
```

### TTL with Rollup (Downsample Before Delete)

```sql
-- Aggregate 1-minute candles into hourly before deleting
CREATE TABLE ohlcv_1m_with_rollup
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    bar_time           DateTime,
    open_price         Float64,
    high_price         Float64,
    low_price          Float64,
    close_price        Float64,
    volume             Float64,
    trade_count        UInt64 DEFAULT 1
)
ENGINE = SummingMergeTree()
ORDER BY (condition_id, outcome, toStartOfHour(bar_time))
TTL bar_time + INTERVAL 6 MONTH
    GROUP BY condition_id, outcome, toStartOfHour(bar_time)
    SET open_price = min(open_price),
        high_price = max(high_price),
        low_price = min(low_price),
        close_price = max(close_price),
        volume = sum(volume),
        trade_count = sum(trade_count),
        bar_time = toStartOfHour(any(bar_time));
```

### Performance Settings

```sql
-- Enable fast TTL cleanup by dropping whole parts
ALTER TABLE market_prices MODIFY SETTING ttl_only_drop_parts = 1;

-- Control merge frequency for TTL processing
ALTER TABLE market_prices MODIFY SETTING merge_with_ttl_timeout = 600;  -- 10 minutes
```

---

## 8. Dictionary / Metadata Tables

### LowCardinality vs. Dictionary Tables

For prediction market data, **LowCardinality columns** are preferred over Dictionary tables because:
- Market categories, outcomes, and event types have < 10,000 distinct values
- Simpler architecture (no external dictionary source to manage)
- Automatic dictionary encoding within the column itself
- Faster GROUP BY and WHERE operations via integer comparisons

Use LowCardinality for:
- `condition_id` in fact tables (thousands of markets)
- `outcome` ('Yes', 'No', named outcomes)
- `category` (politics, sports, crypto, etc.)
- `event_type` (created, resolved, closed)

### When to Use a Dictionary Table

If you need to **join** metadata at query time without denormalizing, use a Dictionary:

```sql
-- Source table for the dictionary
CREATE TABLE market_metadata_source
(
    condition_id       String,
    question           String,
    category           String,
    active             UInt8,
    end_date           DateTime64(3)
)
ENGINE = ReplacingMergeTree()
ORDER BY condition_id;

-- Dictionary definition
CREATE DICTIONARY market_dict
(
    condition_id       String,
    question           String,
    category           String,
    active             UInt8,
    end_date           DateTime64(3)
)
PRIMARY KEY condition_id
SOURCE(CLICKHOUSE(
    TABLE 'market_metadata_source'
    DB 'polymarket'
))
LIFETIME(MIN 60 MAX 300)           -- Refresh every 1-5 minutes
LAYOUT(FLAT());                    -- Best for < 500K keys
```

#### Using the Dictionary in Queries

```sql
-- Fast dictionary lookup (no JOIN needed)
SELECT
    condition_id,
    dictGet('market_dict', 'question', condition_id) AS question,
    dictGet('market_dict', 'category', condition_id) AS category,
    sum(volume) AS total_volume
FROM market_trades
WHERE timestamp >= today() - 7
GROUP BY condition_id
ORDER BY total_volume DESC
LIMIT 20;
```

### Recommended Approach

For this project, **denormalize market metadata into fact tables** using LowCardinality columns. The `markets` ReplacingMergeTree table serves as the source of truth, and the pipeline copies relevant fields (category, question) into trade/price rows at insert time. This avoids JOIN overhead in dashboard queries.

---

## 9. Dashboard Query Patterns

### 9.1 Top Movers (Biggest Price Changes)

```sql
-- Markets with largest 24h price change
WITH current_prices AS (
    SELECT
        condition_id,
        outcome,
        argMax(price, timestamp) AS current_price
    FROM market_prices
    WHERE timestamp >= now() - INTERVAL 1 HOUR
    GROUP BY condition_id, outcome
),
previous_prices AS (
    SELECT
        condition_id,
        outcome,
        argMax(price, timestamp) AS prev_price
    FROM market_prices
    WHERE timestamp BETWEEN now() - INTERVAL 25 HOUR AND now() - INTERVAL 24 HOUR
    GROUP BY condition_id, outcome
)
SELECT
    c.condition_id,
    c.outcome,
    c.current_price,
    p.prev_price,
    c.current_price - p.prev_price AS price_change,
    (c.current_price - p.prev_price) / p.prev_price * 100 AS pct_change
FROM current_prices c
JOIN previous_prices p ON c.condition_id = p.condition_id AND c.outcome = p.outcome
WHERE p.prev_price > 0.05  -- Exclude near-zero markets
ORDER BY abs(pct_change) DESC
LIMIT 20;
```

### 9.2 Trending Markets (Volume Spike Detection)

```sql
-- Markets where recent volume significantly exceeds historical average
WITH recent AS (
    SELECT
        condition_id,
        sum(size) AS volume_1h
    FROM market_trades
    WHERE timestamp >= now() - INTERVAL 1 HOUR
    GROUP BY condition_id
),
baseline AS (
    SELECT
        condition_id,
        sum(size) / 24 AS avg_hourly_volume  -- 24h average
    FROM market_trades
    WHERE timestamp BETWEEN now() - INTERVAL 25 HOUR AND now() - INTERVAL 1 HOUR
    GROUP BY condition_id
)
SELECT
    r.condition_id,
    r.volume_1h,
    b.avg_hourly_volume,
    r.volume_1h / greatest(b.avg_hourly_volume, 0.01) AS volume_ratio
FROM recent r
JOIN baseline b ON r.condition_id = b.condition_id
WHERE b.avg_hourly_volume > 0
ORDER BY volume_ratio DESC
LIMIT 20;
```

### 9.3 Volume Leaders

```sql
-- Top markets by 24h trading volume
SELECT
    condition_id,
    sum(size) AS volume_24h,
    count() AS trade_count,
    uniq(outcome) AS num_outcomes
FROM market_trades
WHERE timestamp >= now() - INTERVAL 24 HOUR
GROUP BY condition_id
ORDER BY volume_24h DESC
LIMIT 50;
```

### 9.4 Market Price History (Chart Data)

```sql
-- Price history for a specific market, suitable for charting
SELECT
    bar_time,
    argMinMerge(open) AS open,
    maxMerge(high) AS high,
    minMerge(low) AS low,
    argMaxMerge(close) AS close,
    sumMerge(volume) AS volume
FROM ohlcv_1m
WHERE condition_id = '{condition_id}'
  AND outcome = 'Yes'
  AND bar_time >= now() - INTERVAL 7 DAY
GROUP BY bar_time
ORDER BY bar_time;
```

### 9.5 Market Overview (Active Markets Summary)

```sql
SELECT
    m.condition_id,
    m.question,
    m.category,
    m.outcomes,
    m.outcome_prices,
    m.volume_total,
    m.liquidity,
    m.end_date
FROM markets FINAL AS m
WHERE m.active = 1
  AND m.closed = 0
ORDER BY m.volume_total DESC
LIMIT 100;
```

### 9.6 Category Breakdown

```sql
SELECT
    category,
    count() AS market_count,
    sum(volume_total) AS total_volume,
    avg(liquidity) AS avg_liquidity
FROM markets FINAL
WHERE active = 1
GROUP BY category
ORDER BY total_volume DESC;
```

### 9.7 Recently Resolved Markets

```sql
SELECT
    condition_id,
    question,
    winning_outcome,
    outcome_prices,
    volume_total,
    end_date
FROM markets FINAL
WHERE resolved = 1
ORDER BY end_date DESC
LIMIT 20;
```

### Dashboard Query Optimization Tips

1. **Use materialized views** for all frequently-queried aggregations (OHLCV, daily volume)
2. **Avoid `SELECT *`** -- only select columns needed for the dashboard widget
3. **Use `FINAL`** on ReplacingMergeTree tables to get deduplicated results
4. **Pre-filter with partition key** (time-based WHERE clauses) to enable partition pruning
5. **Use `PREWHERE`** for conditions on columns not in ORDER BY (ClickHouse often does this automatically)
6. **Cache hot queries** at the application layer (Next.js ISR or API route caching)

---

## 10. Codec & Compression Recommendations

### Column-Specific Codecs

| Column Type | Codec | Rationale |
|-------------|-------|-----------|
| Timestamps (DateTime64) | `DoubleDelta, LZ4` | Sequential timestamps compress extremely well with delta-of-delta encoding |
| Prices (Float64) | `Gorilla, LZ4` | XOR-based encoding optimal for slowly-changing float values |
| Volume/Size (Float64) | `Gorilla, LZ4` | Also slowly-changing numeric |
| Counters (UInt64) | `T64, LZ4` | Integer-optimized codec |
| Short strings (LowCardinality) | Default (LZ4) | Dictionary encoding handles compression |
| Long text (description) | `ZSTD(3)` | Higher compression ratio for text blobs |
| JSON payloads | `ZSTD(1)` | Good compression, moderate CPU |

### Compression Settings

```sql
-- Apply codecs at column level in CREATE TABLE
price Float64 CODEC(Gorilla, LZ4),
timestamp DateTime64(3) CODEC(DoubleDelta, LZ4),
description String CODEC(ZSTD(3)),
volume Float64 CODEC(Gorilla, LZ4),
trade_count UInt64 CODEC(T64, LZ4)
```

### Expected Compression Ratios

For prediction market data (prices between 0-1, timestamps at second/millisecond granularity):
- **Timestamps**: 10-20x compression with DoubleDelta
- **Prices**: 5-10x compression with Gorilla (prices change slowly)
- **String columns**: 10-30x with LZ4/ZSTD + LowCardinality
- **Overall**: Expect 10-15x total compression ratio

---

## Migration Script

A complete migration script that creates all tables and views in order:

```sql
-- ============================================================
-- Polymarket ClickHouse Schema Migration
-- Run against ClickHouse Cloud instance
-- ============================================================

CREATE DATABASE IF NOT EXISTS polymarket;
USE polymarket;

-- 1. Markets (metadata, ReplacingMergeTree)
CREATE TABLE IF NOT EXISTS markets
(
    condition_id       String,
    market_slug        String,
    question           String,
    description        String CODEC(ZSTD(3)),
    category           LowCardinality(String),
    tags               Array(LowCardinality(String)),
    outcomes           Array(String),
    outcome_prices     Array(Float64),
    token_ids          Array(String),
    active             UInt8,
    closed             UInt8,
    resolved           UInt8,
    resolution_source  String DEFAULT '',
    winning_outcome    String DEFAULT '',
    volume_24h         Float64 DEFAULT 0,
    volume_total       Float64 DEFAULT 0,
    liquidity          Float64 DEFAULT 0,
    start_date         DateTime64(3) DEFAULT now64(3),
    end_date           DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),
    created_at         DateTime64(3) DEFAULT now64(3),
    updated_at         DateTime64(3) DEFAULT now64(3),
    INDEX question_idx question TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 4,
    INDEX slug_idx     market_slug TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id)
SETTINGS index_granularity = 8192;

-- 2. Market Prices (tick-level time series)
CREATE TABLE IF NOT EXISTS market_prices
(
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),
    price              Float64,
    bid                Float64 DEFAULT 0,
    ask                Float64 DEFAULT 0,
    spread             Float64 MATERIALIZED ask - bid,
    volume             Float64 DEFAULT 0,
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, outcome, timestamp)
TTL timestamp + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;

-- 3. Market Trades
CREATE TABLE IF NOT EXISTS market_trades
(
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),
    price              Float64,
    size               Float64,
    side               Enum8('buy' = 1, 'sell' = 2),
    trade_id           String,
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX trade_idx     trade_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, outcome, timestamp, trade_id)
TTL timestamp + INTERVAL 2 YEAR DELETE
SETTINGS index_granularity = 8192;

-- 4. Orderbook Snapshots
CREATE TABLE IF NOT EXISTS orderbook_snapshots
(
    condition_id       LowCardinality(String),
    token_id           String,
    outcome            LowCardinality(String),
    bid_prices         Array(Float64),
    bid_sizes          Array(Float64),
    ask_prices         Array(Float64),
    ask_sizes          Array(Float64),
    best_bid           Float64 MATERIALIZED if(length(bid_prices) > 0, bid_prices[1], 0),
    best_ask           Float64 MATERIALIZED if(length(ask_prices) > 0, ask_prices[1], 0),
    mid_price          Float64 MATERIALIZED (best_bid + best_ask) / 2,
    total_bid_depth    Float64 MATERIALIZED arraySum(bid_sizes),
    total_ask_depth    Float64 MATERIALIZED arraySum(ask_sizes),
    snapshot_time      DateTime64(3) CODEC(DoubleDelta, LZ4),
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(snapshot_time)
ORDER BY (condition_id, outcome, snapshot_time)
TTL snapshot_time + INTERVAL 7 DAY DELETE
SETTINGS index_granularity = 8192;

-- 5. Market Events
CREATE TABLE IF NOT EXISTS market_events
(
    condition_id       LowCardinality(String),
    event_type         LowCardinality(String),
    event_data         String CODEC(ZSTD(1)),
    event_time         DateTime64(3) CODEC(DoubleDelta, LZ4),
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX event_type_idx event_type TYPE set(20) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (condition_id, event_type, event_time)
SETTINGS index_granularity = 8192;

-- 6. OHLCV 1-Minute (AggregatingMergeTree target)
CREATE TABLE IF NOT EXISTS ohlcv_1m
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    bar_time           DateTime CODEC(DoubleDelta, LZ4),
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

-- 7. OHLCV 1-Minute Materialized View
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

-- 8. OHLCV 1-Hour (AggregatingMergeTree target)
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

-- 9. OHLCV 1-Hour Materialized View (directly from trades)
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

-- 10. Daily Volume Rollup
CREATE TABLE IF NOT EXISTS volume_daily
(
    condition_id       LowCardinality(String),
    outcome            LowCardinality(String),
    trade_date         Date,
    total_volume       Float64,
    trade_count        UInt64,
    buy_volume         Float64,
    sell_volume        Float64,
    vwap               Float64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (condition_id, outcome, trade_date);

-- 11. Daily Volume Materialized View
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

-- 12. Latest Price Tracker
CREATE TABLE IF NOT EXISTS market_latest_price
(
    condition_id       String,
    outcome            LowCardinality(String),
    price              Float64,
    volume_24h         Float64,
    updated_at         DateTime64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (condition_id, outcome);

CREATE MATERIALIZED VIEW IF NOT EXISTS market_latest_price_mv
TO market_latest_price
AS SELECT
    condition_id,
    outcome,
    price,
    0 AS volume_24h,
    timestamp AS updated_at
FROM market_prices;
```

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python driver | `clickhouse-connect` | Official, HTTP-based, Cloud-compatible, DataFrame support |
| Primary key for prices | `(condition_id, outcome, timestamp)` | Most queries filter by market + outcome, then scan time range |
| Partitioning | Monthly (`toYYYYMM`) | Balances partition count vs. size; aligns with 2-year TTL |
| Candle aggregation | AggregatingMergeTree + State/Merge | Correct incremental aggregation of OHLCV across merges |
| Market metadata | ReplacingMergeTree | Handles upserts naturally; use FINAL for latest state |
| Orderbook retention | 7-day TTL with daily partitions | High volume, low long-term value; daily partitions enable fast DROP |
| String encoding | LowCardinality everywhere | < 10K distinct values per column; 2-3x faster than plain String |
| Compression | DoubleDelta for timestamps, Gorilla for prices | Domain-specific codecs for 10-20x compression |
| Dashboard queries | Pre-aggregated materialized views | Sub-second response for OHLCV charts, volume leaders, top movers |
