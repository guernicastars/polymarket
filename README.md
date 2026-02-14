# Polymarket Data Infrastructure

Continuous data pipeline and analytics dashboard for Polymarket prediction markets. Collects market metadata, prices, trades, and orderbook depth from Polymarket's APIs into ClickHouse Cloud, then serves it through a Next.js dashboard deployed on Vercel.

## Architecture

```
+-----------------------------------------------------------+
|                    Linux VM (Docker)                        |
|                                                            |
|  +------------------------------------------------------+  |
|  |            polymarket-pipeline (Python)               |  |
|  |                                                       |  |
|  |  +------------+  +------------+  +----------------+   |  |
|  |  | Market     |  | Price      |  | Trade          |   |  |
|  |  | Sync       |  | Poller     |  | Collector      |   |  |
|  |  | (5 min)    |  | (30 sec)   |  | (60 sec)       |   |  |
|  |  +-----+------+  +-----+------+  +-------+--------+   |  |
|  |        |               |                 |             |  |
|  |  +-----+------+  +----+-------+          |             |  |
|  |  | Orderbook  |  | WebSocket  |          |             |  |
|  |  | Snapshot   |  | Listener   |          |             |  |
|  |  | (60 sec)   |  | (realtime) |          |             |  |
|  |  +-----+------+  +----+-------+          |             |  |
|  |        |               |                 |             |  |
|  |        v               v                 v             |  |
|  |  +------------------------------------------------+   |  |
|  |  |         ClickHouse Writer (batch buffer)        |   |  |
|  |  +------------------------+-----------------------+   |  |
|  +------------------------------------------------------+  |
+----------------------------+-------------------------------+
                             | HTTPS (port 8443)
                             v
               +-------------------------+
               |   ClickHouse Cloud      |
               |   (polymarket DB)       |
               +------------+------------+
                            | HTTPS
                            v
               +-------------------------+
               |   Vercel                |
               |   Next.js Dashboard     |
               |   (Server Components)   |
               +-------------------------+
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ and npm
- A ClickHouse Cloud account ([cloud.clickhouse.com](https://cloud.clickhouse.com))

### 1. ClickHouse Cloud Setup

1. Create a ClickHouse Cloud service (any region; `iad1` recommended for Vercel pairing).
2. Note the **host**, **port** (8443), **username** (default), and **password**.
3. Run the schema migration against your instance:

```bash
cd polymarket/pipeline
# Option A: use the clickhouse-client CLI
clickhouse-client \
  --host your-instance.clickhouse.cloud \
  --port 9440 \
  --user default \
  --password 'your-password' \
  --secure \
  --multiquery < schema/001_init.sql

# Option B: the pipeline runs migrations on startup automatically
```

This creates the `polymarket` database with 5 core tables and 4 materialized views.

### 2. Pipeline Setup (Data Ingestion)

```bash
cd polymarket/pipeline

# Configure credentials
cp .env.example .env
# Edit .env with your ClickHouse Cloud credentials

# Start the pipeline
docker compose up -d

# Verify it's running
docker compose logs -f

# Check health
curl http://localhost:8080/health
```

The pipeline immediately starts collecting data:
- **Market metadata** from the Gamma API (every 5 min)
- **Price snapshots** from the CLOB API (every 30 sec)
- **Trades** from the Data API (every 60 sec)
- **Orderbook depth** from the CLOB API (every 60 sec)
- **Real-time events** via WebSocket (continuous)

### 3. Dashboard Setup (Analytics UI)

```bash
cd polymarket/dashboard

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your ClickHouse Cloud credentials

# Start development server
npm run dev
# Open http://localhost:3000

# Production build
npm run build
```

## Environment Variables

### Pipeline (`pipeline/.env`)

| Variable | Description | Example |
|---|---|---|
| `CLICKHOUSE_HOST` | ClickHouse Cloud hostname | `abc123.us-east-1.aws.clickhouse.cloud` |
| `CLICKHOUSE_PORT` | ClickHouse Cloud HTTPS port | `8443` |
| `CLICKHOUSE_USER` | ClickHouse username | `default` |
| `CLICKHOUSE_PASSWORD` | ClickHouse password | `your-password` |
| `CLICKHOUSE_DATABASE` | Target database name | `polymarket` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HEALTH_CHECK_PORT` | Health endpoint port | `8080` |

### Dashboard (`dashboard/.env.local`)

| Variable | Description | Example |
|---|---|---|
| `CLICKHOUSE_URL` | ClickHouse Cloud HTTPS URL | `https://abc123.clickhouse.cloud:8443` |
| `CLICKHOUSE_USER` | ClickHouse username | `default` |
| `CLICKHOUSE_PASSWORD` | ClickHouse password | `your-password` |
| `CLICKHOUSE_DB` | Target database name | `polymarket` |

## Data Flow

### Collection Pipeline

1. **Market Discovery** (every 5 min): Gamma API `/events?active=true` with pagination. Fetches all active events with nested markets, flattens into the `markets` table via `ReplacingMergeTree` upserts.

2. **Price Snapshots** (every 30 sec): CLOB API `POST /prices` in batches of 50 tokens. Captures best bid, best ask, mid price, and volume for each active outcome token. Written to `market_prices`.

3. **Trade Collection** (every 60 sec): Data API `/trades` for recent trades across all active markets. Deduplicates by `trade_id`. Inserts into `market_trades`, which triggers materialized views for OHLCV candle generation.

4. **Orderbook Depth** (every 60 sec): CLOB API `POST /books` for top 100 markets by volume. Captures full L2 orderbook (bid/ask arrays). Written to `orderbook_snapshots` with 7-day TTL.

5. **Real-time Events** (continuous): WebSocket connection to `wss://ws-subscriptions-clob.polymarket.com/ws/market`. Processes `last_trade_price` and `price_change` events. Supplements polling data for lower latency.

### ClickHouse Schema

**Core Tables:**

| Table | Engine | Purpose | Retention |
|---|---|---|---|
| `markets` | ReplacingMergeTree | Market metadata, prices, volume | Forever |
| `market_prices` | MergeTree | Tick-level price snapshots | 2 years |
| `market_trades` | MergeTree | Individual trades | 2 years |
| `orderbook_snapshots` | MergeTree | L2 orderbook depth | 7 days |
| `market_events` | MergeTree | Status changes, resolutions | Forever |

**Materialized Views (auto-generated from inserts):**

| View | Source | Purpose |
|---|---|---|
| `ohlcv_1m` | `market_trades` | 1-minute OHLCV candles with buy/sell volume |
| `ohlcv_1h` | `market_trades` | 1-hour OHLCV candles |
| `volume_daily` | `market_trades` | Daily volume rollups with VWAP |
| `market_latest_price` | `market_prices` | Latest price per market/outcome |

### Dashboard Queries

The dashboard queries ClickHouse Cloud directly via server components and route handlers:

- **Overview**: Active market count, total 24h volume, trending count
- **Top Markets**: Markets ranked by 24h volume with current prices
- **Top Movers**: Markets with largest 24h price changes (percent)
- **Trending**: Volume spike detection (current hour vs. 24h average)
- **Market Detail**: Full metadata, OHLCV price history, recent trades
- **Category Breakdown**: Volume and liquidity aggregated by category
- **Recently Resolved**: Markets that have resolved with winning outcomes

## API Endpoints Collected

The pipeline ingests data from three Polymarket APIs:

| API | Base URL | Auth | Data Collected |
|---|---|---|---|
| Gamma API | `gamma-api.polymarket.com` | None | Events, markets, metadata, volume, tags |
| CLOB API | `clob.polymarket.com` | None (read) | Prices, spreads, orderbooks, price history |
| Data API | `data-api.polymarket.com` | None | Trades (public, no wallet required) |
| WebSocket | `ws-subscriptions-clob.polymarket.com` | None | Real-time trades and price changes |

## Dashboard Features

- **Server-rendered initial load** via Next.js Server Components querying ClickHouse
- **Client-side polling** via route handlers with SWR (10-second refresh)
- **Dark theme** with shadcn/ui components and Tailwind CSS
- **Market overview** with KPI cards (volume, active markets, movers)
- **Market detail pages** with OHLCV price charts and trade history
- **Trending detection** based on volume spike ratios
- **Category filtering** across all market views

## Project Structure

```
polymarket/
  pipeline/                          # Python data pipeline (Docker)
    config.py                        # Environment config + structured logging
    clickhouse_writer.py             # Batched writer with buffer and retry
    api/
      gamma_client.py                # Gamma API client (market discovery)
      clob_client.py                 # CLOB API client (prices, orderbooks)
      data_client.py                 # Data API client (trades)
      ws_client.py                   # WebSocket client (real-time events)
    jobs/
      market_sync.py                 # Sync market metadata (every 5 min)
      price_poller.py                # Poll prices/spreads (every 30 sec)
      trade_collector.py             # Collect trades (every 60 sec)
      orderbook_snapshot.py          # Snapshot orderbooks (every 60 sec)
    schema/
      001_init.sql                   # ClickHouse DDL (tables + views)
      migrate.py                     # Migration runner
    Dockerfile                       # Python 3.12-slim container
    docker-compose.yml               # Single-service compose
    .env.example                     # Environment variable template

  dashboard/                         # Next.js app (deploy to Vercel)
    src/
      app/
        layout.tsx                   # Root layout (dark theme)
        page.tsx                     # Dashboard home page
        api/
          markets/route.ts           # GET /api/markets
          markets/[id]/route.ts      # GET /api/markets/:id
          prices/[id]/route.ts       # GET /api/prices/:id
      lib/
        clickhouse.ts                # ClickHouse client singleton
        queries.ts                   # Parameterized query functions
        format.ts                    # Number/date formatters
        utils.ts                     # cn() helper
      components/
        ui/                          # shadcn/ui components
      types/
        market.ts                    # TypeScript interfaces
    next.config.ts
    vercel.json                      # Vercel deployment config
    .env.example                     # Environment variable template

  .planning/
    ARCHITECTURE.md                  # System architecture document
    research/                        # API and technology research
      data-model.md                  # Polymarket data hierarchy
      clickhouse-schema.md           # ClickHouse schema design
      gamma-api.md                   # Gamma API reference
      clob-api.md                    # CLOB + Data + WS API reference
      dashboard-stack.md             # Dashboard technology stack
```

## Deployment

### Pipeline: Linux VM with Docker

1. Provision a Linux VM (any cloud provider; 1 vCPU / 512 MB RAM is sufficient).
2. Install Docker and Docker Compose.
3. Clone the repository and navigate to `polymarket/pipeline`.
4. Copy `.env.example` to `.env` and fill in ClickHouse Cloud credentials.
5. Start the pipeline:

```bash
docker compose up -d
```

6. Monitor logs:

```bash
docker compose logs -f
```

7. The pipeline auto-restarts on failure (`restart: unless-stopped`). Log rotation is configured to 10 MB / 3 files.

### Dashboard: Vercel

1. Push the repository to GitHub.
2. Import the project in [Vercel](https://vercel.com).
3. Set the **Root Directory** to `polymarket/dashboard`.
4. Add environment variables in Vercel project settings:
   - `CLICKHOUSE_URL`
   - `CLICKHOUSE_USER`
   - `CLICKHOUSE_PASSWORD`
   - `CLICKHOUSE_DB`
5. Deploy:

```bash
cd polymarket/dashboard
npx vercel --prod
```

The dashboard uses Node.js runtime (not Edge) for all ClickHouse queries. API routes have a 30-second max duration configured in `vercel.json`.

## Monitoring and Troubleshooting

### Pipeline Health

- **Health endpoint**: `GET http://localhost:8080/health`
- **Docker health check**: Configured with 30-second intervals and 3 retries
- **Structured JSON logs**: All log output is JSON-formatted for parsing

### Common Issues

| Issue | Diagnosis | Fix |
|---|---|---|
| Pipeline not inserting data | Check `docker compose logs` for connection errors | Verify `.env` credentials and that ClickHouse Cloud service is running |
| Dashboard shows no data | Check browser console for API errors | Verify `.env.local` credentials; ensure pipeline has been running long enough to populate tables |
| ClickHouse connection timeout | Network/firewall blocking port 8443 | Ensure outbound HTTPS on port 8443 is allowed |
| High memory usage in pipeline | Buffer accumulating too many rows | Reduce `BUFFER_FLUSH_SIZE` in config or increase flush frequency |
| Stale prices on dashboard | SWR polling not refreshing | Check that route handlers have `dynamic = "force-dynamic"` set |

### Log Inspection

```bash
# Pipeline logs (Docker)
docker compose logs -f --tail 100

# Filter for errors
docker compose logs 2>&1 | grep '"levelname": "ERROR"'

# ClickHouse query log (run from clickhouse-client)
SELECT query, read_rows, elapsed
FROM system.query_log
WHERE type = 'QueryFinish'
  AND query_duration_ms > 1000
ORDER BY event_time DESC
LIMIT 20;
```

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| Pipeline runtime | Python | 3.12 |
| Pipeline HTTP client | httpx | Latest |
| Pipeline DB driver | clickhouse-connect | Latest |
| Pipeline scheduler | APScheduler | Latest |
| Pipeline container | Docker + Compose | Latest |
| Database | ClickHouse Cloud | Latest |
| Dashboard framework | Next.js | 16 |
| Dashboard UI | React 19 + Tailwind CSS 4 + shadcn/ui | Latest |
| Dashboard DB driver | @clickhouse/client | Latest |
| Dashboard deployment | Vercel | Latest |
