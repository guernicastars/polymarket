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

**Phase 1 Jobs (market data):**
- **Market Sync** (every 5 min): Gamma API `/events` → `markets` table (45K+ markets, sorted by volume)
- **Price Poller** (every 30 sec): CLOB API `/prices` → `market_prices` table (top 1,000 tokens by volume)
- **Trade Collector** (every 60 sec): Data API `/trades` → `market_trades` table → triggers OHLCV materialized views
- **Orderbook Snapshots** (every 60 sec): CLOB API `/books` → `orderbook_snapshots` table (top 100, 7-day TTL)
- **WebSocket Listener** (continuous): Real-time trade/price events for top 5,000 tokens (10 connections)
- **Batched Writer**: In-memory buffer (10K rows / 10s flush), 3x retry with exponential backoff

**Phase 2 Jobs (user/wallet data):**
- **Leaderboard Sync** (every 1 hour): Data API `/v1/leaderboard` → `trader_rankings` table (top 200 per category/period/order combo across 10 categories, 4 periods, 2 order types). Discovers wallets for downstream tracking.
- **Holder Sync** (every 15 min): Data API `/holders` → `market_holders` table (top 20 holders for top 50 markets by volume)
- **Position Sync** (every 5 min): Data API `/positions` + `/activity` → `wallet_positions` + `wallet_activity` tables (up to 500 tracked wallets from leaderboard discovery)
- **Profile Enricher** (every 10 min): Gamma API `/public-profile` → `trader_profiles` table (20 wallets per batch, eventually enriches all discovered wallets)

**Phase 3 Jobs (advanced analytics):**
- **Arbitrage Scanner** (every 2 min): Reads `markets` prices, detects sum-to-one deviations (binary markets) and related-market inconsistencies (multi-outcome events) → `arbitrage_opportunities` table. Closes resolved opportunities automatically.
- **Wallet Analyzer** (every 30 min): Reads `wallet_activity`, `trader_profiles`, `wallet_positions`, `markets` for two tasks: (1) Pairwise wallet clustering based on timing correlation, market overlap, and direction agreement → `wallet_clusters` table; (2) Insider scoring based on freshness, win rate, niche focus, position size vs liquidity → `insider_scores` table.
- **Signal Compositor** (every 5 min): Combines 8 signal sources (OBI, volume anomaly, large trade bias, momentum, smart money direction, concentration risk, arbitrage flag, insider activity) into weighted composite score per market → `composite_signals` table. Top 500 markets by volume scored.

**Performance tuning (config.py):**
- `PRICE_POLL_MAX_TOKENS = 1000` — only poll top tokens (by volume) to finish within 30s cycle
- `WS_MAX_TOTAL_TOKENS = 5000` — limit WebSocket to 10 connections (500 tokens each)
- `PRICE_BATCH_SIZE = 100` — tokens per API call, 5 concurrent requests via asyncio.Semaphore
- Markets sorted by `volume_24h` descending in market_sync so top-N slicing picks highest activity

**Key files:**
- `pipeline/config.py` — All intervals, batch sizes, API URLs, buffer settings
- `pipeline/scheduler.py` — APScheduler orchestration, WS lifecycle, health endpoint (:8080), all 11 jobs registered
- `pipeline/api/gamma_client.py` — Gamma API client; categories derived from `event.tags[0].label`
- `pipeline/api/clob_client.py` — CLOB API with concurrent batch fetching
- `pipeline/api/data_client.py` — Data API client with leaderboard, positions, activity, holders, value, profile methods + parse helpers
- `pipeline/api/ws_client.py` — WebSocket with auto-reconnect (exponential backoff)
- `pipeline/clickhouse_writer.py` — Batched writer with per-table buffers (14 tables)
- `pipeline/jobs/leaderboard_sync.py` — Leaderboard scraper, populates `discovered_wallets` shared set
- `pipeline/jobs/holder_sync.py` — Market holder tracker (uses `active_condition_ids` from market_sync)
- `pipeline/jobs/position_sync.py` — Wallet position/activity poller (uses `discovered_wallets` from leaderboard_sync)
- `pipeline/jobs/profile_enricher.py` — Profile enrichment (Gamma API, batch processing)
- `pipeline/jobs/arbitrage_scanner.py` — Cross-market arbitrage detection (sum-to-one + related-market)
- `pipeline/jobs/wallet_analyzer.py` — Wallet clustering + insider scoring
- `pipeline/jobs/signal_compositor.py` — Multi-factor composite signal computation (8 signal sources)
- `pipeline/migrate.py` — Schema migration runner (executes `001_init.sql` + `002_phase2_users.sql` + `003_phase3_analytics.sql`)
- `pipeline/schema/001_init.sql` — Phase 1 ClickHouse DDL (run automatically on startup)
- `pipeline/schema/002_phase2_users.sql` — Phase 2 ClickHouse DDL (5 new tables, run automatically on startup)
- `pipeline/schema/003_phase3_analytics.sql` — Phase 3 ClickHouse DDL (4 new tables: arbitrage_opportunities, wallet_clusters, insider_scores, composite_signals)

**Known behaviors:**
- WebSocket connections 0-3 reconnect every ~10s (Polymarket server-side idle timeout) — not a bug
- Price Poller may show "skipped: max instances reached" if previous run is slow — harmless

### Dashboard (dashboard/)
Next.js 16 + React 19 server components querying ClickHouse Cloud directly, deployed on Vercel.

**Pages:**
- **Overview** (`/`): Stats cards (24h volume, active markets, trending count), Top Markets table, Top Movers, Trending (volume spikes), Category breakdown
- **Markets** (`/markets`): Full market table with search
- **Market Detail** (`/market/[id]`): TradingView candlestick chart (Lightweight Charts v5), orderbook, top holders section
- **Trending** (`/trending`): Volume spike detection
- **Signals** (`/signals`): Betting signals dashboard — Composite scores (Phase 3), OBI, volume anomalies, large trades
- **Whales** (`/whales`): Whale intelligence dashboard — leaderboard rankings, whale activity feed, smart money positions, position concentration analysis
- **Analytics** (`/analytics`): Advanced analytics — composite signal scores, arbitrage opportunities, wallet clusters, insider alerts (4 tabbed views with stats cards)

**Key architecture decisions:**
- Server components for SSR with `export const dynamic = "force-dynamic"` (no caching, fresh ClickHouse queries)
- SWR for client-side data fetching with `revalidateOnFocus: false` (manual reload only, no polling)
- Top Movers query uses `markets.one_day_price_change` from Gamma API (not computed from price history)
- Trending query uses `volume_24h / (volume_1wk / 7)` spike ratio from Gamma API (not computed from trade history)
- Categories derived from Gamma API event tags (first tag label), not a dedicated category field
- TradingView Lightweight Charts **v5** API: `chart.addSeries(CandlestickSeries, {...})` (NOT v4's `addCandlestickSeries()`)
- ClickHouse `FINAL` cannot be used directly in JOINs — must wrap in subquery: `JOIN (SELECT ... FROM markets FINAL) AS m`

**Key files:**
- `dashboard/src/lib/queries.ts` — All ClickHouse queries (29 named functions: 10 market, 5 signal, 8 whale, 6 analytics)
- `dashboard/src/lib/clickhouse.ts` — Singleton ClickHouse client
- `dashboard/src/types/market.ts` — All TypeScript interfaces (9 Phase 1 + 9 Phase 2 + 5 Phase 3 types)
- `dashboard/src/components/price-chart.tsx` — TradingView chart with candlestick + volume
- `dashboard/src/components/signals-stats-cards.tsx` — Signal count stat cards
- `dashboard/src/components/signals-tabs.tsx` — Tab wrapper for Composite / OBI / Volume / Large Trades tables
- `dashboard/src/components/whales-stats-cards.tsx` — Whale stats cards (tracked wallets, trades, positions, markets)
- `dashboard/src/components/whales-tabs.tsx` — Tab wrapper for Leaderboard / Activity / Smart Money / Concentration
- `dashboard/src/components/leaderboard-table.tsx` — Trader leaderboard with rank, PnL, volume, verification
- `dashboard/src/components/whale-activity-table.tsx` — Whale activity feed with type badges, relative timestamps
- `dashboard/src/components/smart-money-table.tsx` — Smart money positions with PnL and returns
- `dashboard/src/components/concentration-table.tsx` — Position concentration with top-5 share progress bars
- `dashboard/src/components/top-holders-table.tsx` — Market detail holders table
- `dashboard/src/components/analytics-stats-cards.tsx` — Analytics stats cards (5 cards: arbitrages, clusters, insider alerts, markets scored, avg confidence)
- `dashboard/src/components/analytics-tabs.tsx` — Tab wrapper for Composite / Arbitrage / Clusters / Insider views
- `dashboard/src/components/composite-signals-table.tsx` — Composite signal scores with score bars and component breakdown
- `dashboard/src/components/arbitrage-table.tsx` — Arbitrage opportunities with spread, type badges
- `dashboard/src/components/wallet-clusters-table.tsx` — Wallet clusters with similarity metrics
- `dashboard/src/components/insider-alerts-table.tsx` — Insider alerts with factor breakdown bars
- `dashboard/vercel.json` — Region iad1, maxDuration 30s (do NOT add `runtime` field — breaks Next.js builds)

**Signal queries (in queries.ts):**
- `getOrderBookImbalance(limit)` — OBI from latest `orderbook_snapshots`, bid/ask ratio sorted by extremity
- `getVolumeAnomalies(limit)` — Markets with 4h volume > 2x daily average from `market_trades` + `markets`
- `getLargeTrades(minSize, limit)` — Individual trades above USD threshold from `market_trades`
- `getMarketTechnicals(conditionId, outcome)` — RSI-14, VWAP, momentum from `ohlcv_1h` (per-market)
- `getSignalsOverview()` — Aggregate signal counts for stats cards

**Whale queries (in queries.ts):**
- `getLeaderboard(category, timePeriod, orderBy, limit)` — Trader rankings from `trader_rankings FINAL`
- `getWhaleActivity(limit)` — Large wallet trades (>=$500 USDC) from `wallet_activity` JOIN `trader_profiles`
- `getSmartMoneyPositions(limit)` — Top-ranked trader positions from `wallet_positions` JOIN `trader_rankings`
- `getTopHolders(conditionId, limit)` — Top holders per market from `market_holders FINAL`
- `getTraderProfile(wallet)` — Single wallet profile from `trader_profiles FINAL`
- `getPositionConcentration(limit)` — Top-5 holder share per market from `market_holders` with CTE aggregation
- `getWhalesOverview()` — Aggregate stats (tracked wallets, whale trades, positions, markets held)

**Analytics queries (in queries.ts):**
- `getArbitrageOpportunities(limit)` — Open arbitrage opportunities from `arbitrage_opportunities FINAL` JOIN `markets`
- `getWalletClusters(limit)` — Detected wallet clusters sorted by similarity from `wallet_clusters FINAL`
- `getInsiderAlerts(minScore, limit)` — Wallets with insider score above threshold from `insider_scores FINAL` JOIN `trader_profiles`
- `getCompositeSignals(limit)` — Composite signal scores sorted by absolute score from `composite_signals FINAL` JOIN `markets`
- `getAnalyticsOverview()` — Aggregate stats (open arbitrages, clusters, insider alerts, markets scored, avg confidence)

**Composite signal weights:** OBI (0.20), volume anomaly (0.10), large trade bias (0.15), momentum (0.15), smart money (0.25), concentration risk (0.10), insider (0.05). Each component normalized to -100..+100; composite clamped to -100..+100.

### ClickHouse Schema (pipeline/schema/)
**Phase 1** (`001_init.sql`) — 5 core tables + 4 materialized views:
- `markets` (ReplacingMergeTree): Market metadata with Gamma API fields (volume_24h, volume_1wk, volume_1mo, one_day_price_change, one_week_price_change, category, tags)
- `market_prices` (MergeTree): Tick-level price snapshots, partitioned monthly, 2-year TTL
- `market_trades` (MergeTree): Individual trades with buy/sell side
- `orderbook_snapshots` (MergeTree): L2 depth, daily partitions, 7-day TTL
- `market_events` (MergeTree): Status changes and resolutions
- `ohlcv_1m`, `ohlcv_1h` (AggregatingMergeTree): OHLCV candles via materialized views from trades
- `volume_daily` (SummingMergeTree): Daily volume rollups with VWAP
- `market_latest_price` (ReplacingMergeTree): Latest price tracker

**Phase 2** (`002_phase2_users.sql`) — 5 user/wallet tables:
- `trader_rankings` (ReplacingMergeTree): Leaderboard snapshots by wallet+category+time_period+order_by
- `market_holders` (ReplacingMergeTree): Top holders per market by condition_id+wallet+outcome_index
- `wallet_positions` (ReplacingMergeTree): Tracked wallet open positions by wallet+condition_id+outcome
- `wallet_activity` (MergeTree): Append-only wallet trade/activity history, partitioned monthly, 1-year TTL
- `trader_profiles` (ReplacingMergeTree): Wallet profile data (pseudonym, bio, X handle, verification status)

**Phase 3** (`003_phase3_analytics.sql`) — 4 analytics tables:
- `arbitrage_opportunities` (ReplacingMergeTree): Detected pricing inconsistencies (sum-to-one + related-market), 30-day TTL
- `wallet_clusters` (ReplacingMergeTree): Grouped wallets with synchronized trading behavior, 90-day TTL
- `insider_scores` (ReplacingMergeTree): Per-wallet insider risk score (0-100) with factor breakdown, 90-day TTL
- `composite_signals` (ReplacingMergeTree): Per-market multi-factor signal (-100 to +100) with 8 component scores

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

## Signals Roadmap

### Phase 1 (COMPLETE): Signals from Existing Data
Four signal types computed at query time from existing pipeline tables (no new schema or pipeline jobs):
- **Order Book Imbalance (OBI)** — bid/ask volume ratio from `orderbook_snapshots`. OBI > 0.6 = bullish, < 0.4 = bearish.
- **Volume Anomaly Detection** — 4h volume vs rolling daily average from `market_trades` + `markets`. Ratio > 2.0 = anomalous.
- **Large Trade Detection** — individual trades above USD threshold from `market_trades`. Default >= $1,000.
- **Technical Indicators (RSI-14, VWAP, Momentum)** — computed from `ohlcv_1h` per-market (available for market detail page).

Dashboard `/signals` page live with stats cards, tabbed OBI/Volume/Large Trades tables, and sidebar navigation.

### Phase 2 (COMPLETE): User & Wallet Data
New pipeline jobs, schema migration, Data API client extensions, and Whales dashboard page:
- **Leaderboard Sync** (hourly) — Fetches top 200 traders per category/period/order combo across 10 categories, 4 time periods, 2 order types → `trader_rankings` table. Discovers wallets for downstream tracking via shared `discovered_wallets` set.
- **Holder Sync** (15 min) — Fetches top 20 holders for top 50 markets by volume → `market_holders` table.
- **Position Sync** (5 min) — Fetches positions + recent activity for up to 500 tracked wallets → `wallet_positions` + `wallet_activity` tables. Dedup via per-wallet timestamp watermarks.
- **Profile Enricher** (10 min) — Fetches Gamma API public profiles for newly discovered wallets (20 per batch) → `trader_profiles` table.

New tables: `trader_rankings`, `market_holders`, `wallet_positions`, `wallet_activity`, `trader_profiles` (schema in `002_phase2_users.sql`).

Dashboard `/whales` page with 4 stats cards (tracked wallets, whale trades 24h, total positions, markets held) and 4 tabbed views (Leaderboard, Activity Feed, Smart Money Positions, Position Concentration). Market detail page enhanced with Top Holders section.

### Phase 3 (COMPLETE): Advanced Analytics
New pipeline jobs, schema migration, and Analytics dashboard page:
- **Arbitrage Scanner** (2 min) — Detects sum-to-one deviations in binary markets and pricing inconsistencies across related multi-outcome events → `arbitrage_opportunities` table. Auto-closes resolved opportunities.
- **Wallet Analyzer** (30 min) — Two tasks: (1) Pairwise wallet clustering (timing correlation, market overlap, direction agreement) → `wallet_clusters` table; (2) Insider scoring (freshness, win rate, niche focus, size vs liquidity) → `insider_scores` table.
- **Signal Compositor** (5 min) — Combines 8 signal sources into weighted composite score (-100 to +100) per market for top 500 markets → `composite_signals` table. Components: OBI (0.20), volume anomaly (0.10), large trade bias (0.15), momentum (0.15), smart money (0.25), concentration risk (0.10), insider (0.05). Arbitrage flag and insider activity as additional metadata.

New tables: `arbitrage_opportunities`, `wallet_clusters`, `insider_scores`, `composite_signals` (schema in `003_phase3_analytics.sql`).

Dashboard `/analytics` page with 5 stats cards (open arbitrages, wallet clusters, insider alerts, markets scored, avg confidence) and 4 tabbed views (Composite Scores, Arbitrage, Clusters, Insider). `/signals` page enhanced with Composite tab. Sidebar navigation includes Analytics link.

**System totals:** 11 pipeline jobs (4 Phase 1 + 4 Phase 2 + 3 Phase 3), 3 schema files (14 tables + 4 materialized views), 4 dashboard pages (/, /signals, /whales, /analytics), 29 query functions.

## Known Issues
- WebSocket connections 0-3 reconnect every ~10s (Polymarket server-side idle timeout)
- `vercel.json` must NOT have a `runtime` field (causes build error)
- Dashboard queries use Gamma API pre-computed fields (one_day_price_change, volume_24h/1wk/1mo) — accumulated pipeline data will supplement as history builds
