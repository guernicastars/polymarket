# Polymarket Data Infrastructure — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Linux VM (Docker)                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            polymarket-pipeline (Python)              │    │
│  │                                                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐    │    │
│  │  │ Market   │  │ Price    │  │ WebSocket      │    │    │
│  │  │ Sync     │  │ Poller   │  │ Listener       │    │    │
│  │  │ (5 min)  │  │ (30 sec) │  │ (real-time)    │    │    │
│  │  └────┬─────┘  └────┬─────┘  └──────┬─────────┘    │    │
│  │       │              │               │              │    │
│  │       ▼              ▼               ▼              │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │              ClickHouse Writer               │   │    │
│  │  │         (batch insert, buffer)               │   │    │
│  │  └───────────────────┬──────────────────────────┘   │    │
│  └──────────────────────┼──────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTPS (port 8443)
                          ▼
              ┌───────────────────────┐
              │  ClickHouse Cloud     │
              │  (polymarket DB)      │
              └───────────┬───────────┘
                          │ HTTPS
                          ▼
              ┌───────────────────────┐
              │  Vercel               │
              │  Next.js Dashboard    │
              │  (Server Components)  │
              └───────────────────────┘
```

## Components

### 1. Python Pipeline (`polymarket/pipeline/`)
- **market_sync.py** — Syncs market metadata from Gamma API every 5 min
- **price_poller.py** — Polls prices/spreads from CLOB API every 30 sec
- **trade_collector.py** — Collects recent trades from Data API every 60 sec
- **ws_listener.py** — WebSocket connection for real-time trade/price events
- **orderbook_snapshot.py** — Snapshots orderbook depth every 60 sec
- **clickhouse_writer.py** — Batched writer with buffer for all data types
- **scheduler.py** — APScheduler orchestrates all polling jobs
- **config.py** — Environment-based configuration

### 2. ClickHouse Schema (`polymarket/pipeline/schema/`)
- 5 core tables: markets, market_prices, market_trades, orderbook_snapshots, market_events
- 4 materialized views: ohlcv_1m, ohlcv_1h, volume_daily, market_latest_price
- ReplacingMergeTree for metadata, MergeTree for time-series, AggregatingMergeTree for rollups

### 3. Next.js Dashboard (`polymarket/dashboard/`)
- Server Components for initial data load
- Route handlers for client-side polling (10s refresh)
- TradingView Lightweight Charts for price candles
- shadcn/ui components + Tailwind CSS
- Key pages: Overview, Market Detail, Trending, Signals

### 4. Docker (`polymarket/pipeline/`)
- Single `docker-compose.yml` with pipeline service
- Environment variables for ClickHouse Cloud connection
- Health checks, restart policies, log rotation

## Data Flow

1. **Market Discovery** (every 5 min): Gamma API → markets table
2. **Price Snapshots** (every 30 sec): CLOB API → market_prices table
3. **Trade Collection** (every 60 sec): Data API → market_trades table → triggers OHLCV materialized views
4. **Orderbook Depth** (every 60 sec): CLOB API → orderbook_snapshots table
5. **Real-Time** (continuous): WebSocket → market_prices + market_trades (supplements polling)

## Environment Variables

### Pipeline (Docker)
```
CLICKHOUSE_HOST=xxx.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=xxx
CLICKHOUSE_DATABASE=polymarket
```

### Dashboard (Vercel)
```
CLICKHOUSE_URL=https://xxx.clickhouse.cloud:8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=xxx
CLICKHOUSE_DB=polymarket
```
