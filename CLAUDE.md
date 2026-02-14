# CLAUDE.md — Polymarket Signals

Prediction market data infrastructure: continuous pipeline ingesting Polymarket data into ClickHouse Cloud, with a Next.js analytics dashboard deployed on Vercel.

## Quick Start

### Pipeline (Python/Docker) — Continuous data ingestion
```bash
cd pipeline
cp .env.example .env              # Configure ClickHouse Cloud credentials
docker compose up -d --build      # Build and start
docker compose logs -f            # Monitor
docker compose logs --tail 50 2>&1 | grep -E 'flush_ok|error|complete'  # Key events
curl http://localhost:8080/health # Health check
```

### Dashboard (Next.js/TypeScript) — Vercel-deployed analytics UI
```bash
cd dashboard && npm install
cp .env.example .env.local        # Configure ClickHouse Cloud credentials
npm run dev     # http://localhost:3000
npm run build   # Production build (validate before deploy)
# Deploy: push to main (Vercel auto-deploys) or `npx vercel --prod`
```

## Architecture

### Pipeline (pipeline/)
Continuous Python pipeline polling Polymarket APIs (Gamma, CLOB, Data, WebSocket) and writing to ClickHouse Cloud. Runs in Docker on a Linux VM (Azure: `ivekchik@172.205.218.70`).

**Jobs:**
- **Market Sync** (every 5 min): Gamma API `/events` → `markets` table (45K+ markets, sorted by volume)
- **Price Poller** (every 30 sec): CLOB API `/prices` → `market_prices` table (top 1,000 tokens by volume)
- **Trade Collector** (every 60 sec): Data API `/trades` → `market_trades` table → triggers OHLCV materialized views
- **Orderbook Snapshots** (every 60 sec): CLOB API `/books` → `orderbook_snapshots` table (top 100, 7-day TTL)
- **WebSocket Listener** (continuous): Real-time trade/price events for top 5,000 tokens (10 connections)
- **Batched Writer**: In-memory buffer (10K rows / 10s flush), 3x retry with exponential backoff

**Performance tuning (config.py):**
- `PRICE_POLL_MAX_TOKENS = 1000` — only poll top tokens (by volume) to finish within 30s cycle
- `WS_MAX_TOTAL_TOKENS = 5000` — limit WebSocket to 10 connections (500 tokens each)
- `PRICE_BATCH_SIZE = 100` — tokens per API call, 5 concurrent requests via asyncio.Semaphore
- Markets sorted by `volume_24h` descending in market_sync so top-N slicing picks highest activity

**Key files:**
- `pipeline/config.py` — All intervals, batch sizes, API URLs, buffer settings
- `pipeline/scheduler.py` — APScheduler orchestration, WS lifecycle, health endpoint (:8080)
- `pipeline/api/gamma_client.py` — Gamma API client; categories derived from `event.tags[0].label`
- `pipeline/api/clob_client.py` — CLOB API with concurrent batch fetching
- `pipeline/api/ws_client.py` — WebSocket with auto-reconnect (exponential backoff)
- `pipeline/clickhouse_writer.py` — Batched writer with per-table buffers
- `pipeline/schema/001_init.sql` — Full ClickHouse DDL (run automatically on startup)

**Known behaviors:**
- WebSocket connections 0-3 reconnect every ~10s (Polymarket server-side idle timeout) — not a bug
- Price Poller may show "skipped: max instances reached" if previous run is slow — harmless

### Dashboard (dashboard/)
Next.js 16 + React 19 server components querying ClickHouse Cloud directly, deployed on Vercel.

**Pages:**
- **Overview** (`/`): Stats cards (24h volume, active markets, trending count), Top Markets table, Top Movers, Trending (volume spikes), Category breakdown
- **Markets** (`/markets`): Full market table with search
- **Market Detail** (`/market/[id]`): TradingView candlestick chart (Lightweight Charts v5), orderbook
- **Trending** (`/trending`): Volume spike detection
- **Signals** (`/signals`): Betting signals dashboard

**Key architecture decisions:**
- Server components for SSR with `export const dynamic = "force-dynamic"` (no caching, fresh ClickHouse queries)
- SWR for client-side data fetching with `revalidateOnFocus: false` (manual reload only, no polling)
- Top Movers query uses `markets.one_day_price_change` from Gamma API (not computed from price history)
- Trending query uses `volume_24h / (volume_1wk / 7)` spike ratio from Gamma API (not computed from trade history)
- Categories derived from Gamma API event tags (first tag label), not a dedicated category field
- TradingView Lightweight Charts **v5** API: `chart.addSeries(CandlestickSeries, {...})` (NOT v4's `addCandlestickSeries()`)
- ClickHouse `FINAL` cannot be used directly in JOINs — must wrap in subquery: `JOIN (SELECT ... FROM markets FINAL) AS m`

**Key files:**
- `dashboard/src/lib/queries.ts` — All ClickHouse queries (10 named functions)
- `dashboard/src/lib/clickhouse.ts` — Singleton ClickHouse client
- `dashboard/src/components/price-chart.tsx` — TradingView chart with candlestick + volume
- `dashboard/vercel.json` — Region iad1, maxDuration 30s (do NOT add `runtime` field — breaks Next.js builds)

### ClickHouse Schema (pipeline/schema/)
5 core tables + 4 materialized views:
- `markets` (ReplacingMergeTree): Market metadata with Gamma API fields (volume_24h, volume_1wk, volume_1mo, one_day_price_change, one_week_price_change, category, tags)
- `market_prices` (MergeTree): Tick-level price snapshots, partitioned monthly, 2-year TTL
- `market_trades` (MergeTree): Individual trades with buy/sell side
- `orderbook_snapshots` (MergeTree): L2 depth, daily partitions, 7-day TTL
- `market_events` (MergeTree): Status changes and resolutions
- `ohlcv_1m`, `ohlcv_1h` (AggregatingMergeTree): OHLCV candles via materialized views from trades
- `volume_daily` (SummingMergeTree): Daily volume rollups with VWAP
- `market_latest_price` (ReplacingMergeTree): Latest price tracker

## Environment Variables

### Pipeline (pipeline/.env)
```
CLICKHOUSE_HOST=<host>
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=<password>
CLICKHOUSE_DATABASE=polymarket
```

### Dashboard (dashboard/.env.local) + Vercel env vars
```
CLICKHOUSE_URL=https://<host>:8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=<password>
CLICKHOUSE_DB=polymarket
```

## Deployment

### Pipeline → Azure VM
```bash
ssh -i /Users/ivrejchik/Downloads/max.pem ivekchik@172.205.218.70
cd ~/polymarket/pipeline
# Update: git pull && docker compose up -d --build
# Logs: docker compose logs -f
# Health: curl http://localhost:8080/health
```

### Dashboard → Vercel
- Auto-deploys on push to `main` branch
- Manual: `cd dashboard && npx vercel --prod`
- Project URL: https://polymarket-f9bnfijfb-eugene-ss-projects.vercel.app
- GitHub: https://github.com/guernicastars/polymarket

## Known Issues
- WebSocket connections 0-3 reconnect every ~10s (Polymarket server-side idle timeout)
- `vercel.json` must NOT have a `runtime` field (causes build error)
- Dashboard queries use Gamma API pre-computed fields (one_day_price_change, volume_24h/1wk/1mo) — accumulated pipeline data will supplement as history builds
