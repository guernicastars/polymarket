# Phase 1 Signals Spec — Polymarket Signals

> Single source of truth for all Phase 1 implementation. Every DDL statement, query function, TypeScript type, and component is defined here.

## Overview

Phase 1 builds four signal types from **existing pipeline data only** (no new API integrations):

1. **Order Book Imbalance (OBI)** — from `orderbook_snapshots`
2. **Volume Anomaly Detection** — from `market_trades` / OHLCV views
3. **Large Trade Detection** — from `market_trades`
4. **Technical Indicators (RSI, VWAP, Momentum)** — from `ohlcv_1h`

All signals are computed via ClickHouse queries at read time (no new materialized views needed for Phase 1). The dashboard exposes them on a new `/signals` page.

---

## 1. ClickHouse Schema Changes

**No new DDL is required.** All four signal types are computed as analytical queries over existing tables (`orderbook_snapshots`, `market_trades`, `ohlcv_1h`, `markets`). This avoids schema migration coordination and keeps Phase 1 self-contained in the dashboard.

### Tables Used

| Signal | Source Table(s) | Key Columns |
|--------|----------------|-------------|
| OBI | `orderbook_snapshots` | `bid_sizes`, `ask_sizes`, `condition_id`, `outcome`, `snapshot_time` |
| Volume Anomaly | `market_trades`, `markets` | `size`, `timestamp`, `condition_id`, `volume_24h`, `volume_1wk` |
| Large Trades | `market_trades`, `markets` | `size`, `price`, `side`, `timestamp`, `condition_id` |
| RSI / VWAP / Momentum | `ohlcv_1h` (AggregatingMergeTree) | `open`, `high`, `low`, `close`, `volume`, `bar_time` |

### Why No Materialized Views

- OBI is a point-in-time snapshot query (latest orderbook state per market) — a materialized view would duplicate `orderbook_snapshots` data with minimal benefit given the 7-day TTL and 60s snapshot frequency.
- Volume anomalies compare recent vs historical volume — this is a ratio computed at query time.
- Large trades are a simple threshold filter on `market_trades`.
- RSI/VWAP/Momentum use window functions over `ohlcv_1h` — ClickHouse handles these efficiently on the existing AggregatingMergeTree.

If query latency becomes an issue at scale, Phase 2 can introduce materialized views for pre-aggregation.

---

## 2. ClickHouse Queries

All queries go in `dashboard/src/lib/queries.ts`. Each query is a named export function. Below are the exact SQL and TypeScript signatures.

### 2.1 Order Book Imbalance (OBI)

Computes bid/ask volume ratio from the most recent orderbook snapshot per market. OBI > 0.6 = bullish pressure, OBI < 0.4 = bearish pressure.

```sql
-- getOrderBookImbalance(limit = 50)
SELECT
  os.condition_id,
  m.question,
  os.outcome,
  arraySum(os.bid_sizes) AS total_bid,
  arraySum(os.ask_sizes) AS total_ask,
  arraySum(os.bid_sizes) / greatest(arraySum(os.bid_sizes) + arraySum(os.ask_sizes), 0.001) AS obi,
  if(length(os.bid_prices) > 0, os.bid_prices[1], 0) AS best_bid,
  if(length(os.ask_prices) > 0, os.ask_prices[1], 0) AS best_ask,
  os.snapshot_time
FROM orderbook_snapshots os
INNER JOIN (
  SELECT condition_id, outcome, max(snapshot_time) AS max_time
  FROM orderbook_snapshots
  WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
  GROUP BY condition_id, outcome
) latest ON os.condition_id = latest.condition_id
  AND os.outcome = latest.outcome
  AND os.snapshot_time = latest.max_time
INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
  ON os.condition_id = m.condition_id
WHERE (arraySum(os.bid_sizes) + arraySum(os.ask_sizes)) > 0
ORDER BY abs(obi - 0.5) DESC
LIMIT {limit:UInt32}
```

**TypeScript signature:**
```typescript
export async function getOrderBookImbalance(limit = 50): Promise<OBISignal[]>
```

### 2.2 Volume Anomalies

Detects markets where recent trade volume (last 4 hours) significantly exceeds their rolling daily average volume. Uses `market_trades` aggregated against `markets.volume_1wk` baseline.

```sql
-- getVolumeAnomalies(limit = 30)
SELECT
  t.condition_id,
  m.question,
  m.outcome_prices[1] AS current_price,
  sum(t.size) AS volume_4h,
  m.volume_1wk / 7 AS avg_daily_volume,
  sum(t.size) / greatest(m.volume_1wk / 7 / 6, 0.01) AS volume_ratio,
  count() AS trade_count
FROM market_trades t
INNER JOIN (
  SELECT condition_id, question, outcome_prices, volume_1wk
  FROM markets FINAL
  WHERE active = 1 AND closed = 0
) AS m ON t.condition_id = m.condition_id
WHERE t.timestamp >= now() - INTERVAL 4 HOUR
GROUP BY t.condition_id, m.question, m.outcome_prices, m.volume_1wk
HAVING volume_ratio > 2.0 AND volume_4h > 100
ORDER BY volume_ratio DESC
LIMIT {limit:UInt32}
```

**TypeScript signature:**
```typescript
export async function getVolumeAnomalies(limit = 30): Promise<VolumeAnomaly[]>
```

### 2.3 Large Trade Detection

Surfaces individual trades above a dollar-size threshold (price * size). Joins with `markets` for context.

```sql
-- getLargeTrades(minSize = 1000, limit = 50)
SELECT
  t.condition_id,
  m.question,
  t.outcome,
  t.price,
  t.size,
  t.price * t.size AS usd_size,
  t.side,
  t.trade_id,
  t.timestamp
FROM market_trades t
INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
  ON t.condition_id = m.condition_id
WHERE t.timestamp >= now() - INTERVAL 24 HOUR
  AND t.price * t.size >= {minSize:Float64}
ORDER BY usd_size DESC
LIMIT {limit:UInt32}
```

**TypeScript signature:**
```typescript
export async function getLargeTrades(minSize = 1000, limit = 50): Promise<LargeTrade[]>
```

### 2.4 Technical Indicators (RSI, VWAP, Momentum)

Computes RSI-14, VWAP, and momentum (24h price change %) from hourly OHLCV candles for a given market. This is a per-market query used on the market detail page and the signals overview.

```sql
-- getMarketTechnicals(conditionId, outcome = 'Yes')
WITH hourly AS (
  SELECT
    bar_time,
    argMinMerge(open) AS o,
    maxMerge(high) AS h,
    minMerge(low) AS l,
    argMaxMerge(close) AS c,
    sumMerge(volume) AS v
  FROM ohlcv_1h
  WHERE condition_id = {conditionId:String}
    AND outcome = {outcome:String}
    AND bar_time >= now() - INTERVAL 7 DAY
  GROUP BY bar_time
  ORDER BY bar_time
),
deltas AS (
  SELECT
    bar_time, o, h, l, c, v,
    c - lagInFrame(c, 1, c) OVER (ORDER BY bar_time) AS delta
  FROM hourly
),
gains_losses AS (
  SELECT
    bar_time, o, h, l, c, v, delta,
    if(delta > 0, delta, 0) AS gain,
    if(delta < 0, abs(delta), 0) AS loss
  FROM deltas
)
SELECT
  bar_time,
  o AS open,
  h AS high,
  l AS low,
  c AS close,
  v AS volume,
  -- RSI-14: use avg gain/loss over trailing 14 bars
  100 - (100 / (1 + (
    avg(gain) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
    / greatest(avg(loss) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0.0001)
  ))) AS rsi,
  -- VWAP: cumulative price*volume / cumulative volume within the day
  sum(c * v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time) /
    greatest(sum(v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time), 0.0001) AS vwap,
  -- Momentum: % change over 24 bars (24h)
  if(lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) > 0,
    (c - lagInFrame(c, 24, 0) OVER (ORDER BY bar_time)) / lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) * 100,
    0) AS momentum
FROM gains_losses
ORDER BY bar_time
```

**TypeScript signature:**
```typescript
export async function getMarketTechnicals(
  conditionId: string,
  outcome?: string  // default 'Yes'
): Promise<TechnicalBar[]>
```

### 2.5 Signals Overview (Aggregated Summary)

A lightweight query to power the stats cards at the top of the /signals page. Counts active signals across all types.

```sql
-- getSignalsOverview()
SELECT
  -- OBI extremes (>0.65 or <0.35) from latest snapshots
  (
    SELECT count(DISTINCT condition_id)
    FROM orderbook_snapshots
    WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
      AND (
        arraySum(bid_sizes) / greatest(arraySum(bid_sizes) + arraySum(ask_sizes), 0.001) > 0.65
        OR arraySum(bid_sizes) / greatest(arraySum(bid_sizes) + arraySum(ask_sizes), 0.001) < 0.35
      )
  ) AS obi_signals,
  -- Volume anomalies (4h volume > 2x daily avg)
  (
    SELECT count(DISTINCT t.condition_id)
    FROM market_trades t
    INNER JOIN (SELECT condition_id, volume_1wk FROM markets FINAL WHERE active = 1 AND closed = 0) AS m
      ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 4 HOUR
    GROUP BY t.condition_id, m.volume_1wk
    HAVING sum(t.size) / greatest(m.volume_1wk / 7 / 6, 0.01) > 2.0 AND sum(t.size) > 100
  ) AS volume_anomalies,
  -- Large trades (>$1K in last 24h)
  (
    SELECT count()
    FROM market_trades
    WHERE timestamp >= now() - INTERVAL 24 HOUR
      AND price * size >= 1000
  ) AS large_trades_24h,
  -- Total active markets for context
  (
    SELECT countIf(active = 1 AND closed = 0)
    FROM markets FINAL
  ) AS active_markets
```

**Note:** The volume_anomalies subquery uses a pattern that may need adjustment since ClickHouse doesn't support HAVING in scalar subqueries cleanly. Use this simpler alternative:

```sql
-- getSignalsOverview() — simplified
SELECT
  (
    SELECT count(DISTINCT condition_id)
    FROM orderbook_snapshots
    WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
      AND arraySum(bid_sizes) + arraySum(ask_sizes) > 0
      AND abs(arraySum(bid_sizes) / (arraySum(bid_sizes) + arraySum(ask_sizes)) - 0.5) > 0.15
  ) AS obi_signals,
  (
    SELECT uniq(condition_id)
    FROM market_trades
    WHERE timestamp >= now() - INTERVAL 4 HOUR
  ) AS volume_active_markets,
  (
    SELECT count()
    FROM market_trades
    WHERE timestamp >= now() - INTERVAL 24 HOUR
      AND price * size >= 1000
  ) AS large_trades_24h,
  (
    SELECT countIf(active = 1 AND closed = 0)
    FROM markets FINAL
  ) AS active_markets
```

**TypeScript signature:**
```typescript
export async function getSignalsOverview(): Promise<SignalsOverview>
```

---

## 3. TypeScript Types

Add to `dashboard/src/types/market.ts`:

```typescript
// --- Phase 1 Signal Types ---

export interface OBISignal {
  condition_id: string;
  question: string;
  outcome: string;
  total_bid: number;
  total_ask: number;
  obi: number;            // 0-1, >0.6 bullish, <0.4 bearish
  best_bid: number;
  best_ask: number;
  snapshot_time: string;
}

export interface VolumeAnomaly {
  condition_id: string;
  question: string;
  current_price: number;
  volume_4h: number;
  avg_daily_volume: number;
  volume_ratio: number;   // >2.0 = anomalous
  trade_count: number;
}

export interface LargeTrade {
  condition_id: string;
  question: string;
  outcome: string;
  price: number;
  size: number;
  usd_size: number;
  side: string;
  trade_id: string;
  timestamp: string;
}

export interface TechnicalBar {
  bar_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi: number;           // 0-100
  vwap: number;
  momentum: number;      // % change over 24h
}

export interface SignalsOverview {
  obi_signals: number;
  volume_active_markets: number;
  large_trades_24h: number;
  active_markets: number;
}
```

---

## 4. Query Functions — Exact Implementation

Add these 5 functions to `dashboard/src/lib/queries.ts`. Import the new types.

### 4.1 Updated Imports (top of queries.ts)

```typescript
import type {
  // ...existing imports...
  OBISignal,
  VolumeAnomaly,
  LargeTrade,
  TechnicalBar,
  SignalsOverview,
} from "@/types/market";
```

### 4.2 getOrderBookImbalance

```typescript
export async function getOrderBookImbalance(limit = 50): Promise<OBISignal[]> {
  return query<OBISignal>(
    `SELECT
      os.condition_id,
      m.question,
      os.outcome,
      arraySum(os.bid_sizes) AS total_bid,
      arraySum(os.ask_sizes) AS total_ask,
      arraySum(os.bid_sizes) / greatest(arraySum(os.bid_sizes) + arraySum(os.ask_sizes), 0.001) AS obi,
      if(length(os.bid_prices) > 0, os.bid_prices[1], 0) AS best_bid,
      if(length(os.ask_prices) > 0, os.ask_prices[1], 0) AS best_ask,
      os.snapshot_time
    FROM orderbook_snapshots os
    INNER JOIN (
      SELECT condition_id, outcome, max(snapshot_time) AS max_time
      FROM orderbook_snapshots
      WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
      GROUP BY condition_id, outcome
    ) latest ON os.condition_id = latest.condition_id
      AND os.outcome = latest.outcome
      AND os.snapshot_time = latest.max_time
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON os.condition_id = m.condition_id
    WHERE (arraySum(os.bid_sizes) + arraySum(os.ask_sizes)) > 0
    ORDER BY abs(obi - 0.5) DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 4.3 getVolumeAnomalies

```typescript
export async function getVolumeAnomalies(limit = 30): Promise<VolumeAnomaly[]> {
  return query<VolumeAnomaly>(
    `SELECT
      t.condition_id,
      m.question,
      m.outcome_prices[1] AS current_price,
      sum(t.size) AS volume_4h,
      m.volume_1wk / 7 AS avg_daily_volume,
      sum(t.size) / greatest(m.volume_1wk / 7 / 6, 0.01) AS volume_ratio,
      count() AS trade_count
    FROM market_trades t
    INNER JOIN (
      SELECT condition_id, question, outcome_prices, volume_1wk
      FROM markets FINAL
      WHERE active = 1 AND closed = 0
    ) AS m ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 4 HOUR
    GROUP BY t.condition_id, m.question, m.outcome_prices, m.volume_1wk
    HAVING volume_ratio > 2.0 AND volume_4h > 100
    ORDER BY volume_ratio DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 4.4 getLargeTrades

```typescript
export async function getLargeTrades(
  minSize = 1000,
  limit = 50
): Promise<LargeTrade[]> {
  return query<LargeTrade>(
    `SELECT
      t.condition_id,
      m.question,
      t.outcome,
      t.price,
      t.size,
      t.price * t.size AS usd_size,
      t.side,
      t.trade_id,
      t.timestamp
    FROM market_trades t
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 24 HOUR
      AND t.price * t.size >= {minSize:Float64}
    ORDER BY usd_size DESC
    LIMIT {limit:UInt32}`,
    { minSize, limit }
  );
}
```

### 4.5 getMarketTechnicals

```typescript
export async function getMarketTechnicals(
  conditionId: string,
  outcome = "Yes"
): Promise<TechnicalBar[]> {
  return query<TechnicalBar>(
    `WITH hourly AS (
      SELECT
        bar_time,
        argMinMerge(open) AS o,
        maxMerge(high) AS h,
        minMerge(low) AS l,
        argMaxMerge(close) AS c,
        sumMerge(volume) AS v
      FROM ohlcv_1h
      WHERE condition_id = {conditionId:String}
        AND outcome = {outcome:String}
        AND bar_time >= now() - INTERVAL 7 DAY
      GROUP BY bar_time
      ORDER BY bar_time
    ),
    deltas AS (
      SELECT
        bar_time, o, h, l, c, v,
        c - lagInFrame(c, 1, c) OVER (ORDER BY bar_time) AS delta
      FROM hourly
    ),
    gains_losses AS (
      SELECT
        bar_time, o, h, l, c, v, delta,
        if(delta > 0, delta, 0) AS gain,
        if(delta < 0, abs(delta), 0) AS loss
      FROM deltas
    )
    SELECT
      bar_time,
      o AS open,
      h AS high,
      l AS low,
      c AS close,
      v AS volume,
      100 - (100 / (1 + (
        avg(gain) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
        / greatest(avg(loss) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0.0001)
      ))) AS rsi,
      sum(c * v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time) /
        greatest(sum(v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time), 0.0001) AS vwap,
      if(lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) > 0,
        (c - lagInFrame(c, 24, 0) OVER (ORDER BY bar_time)) / lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) * 100,
        0) AS momentum
    FROM gains_losses
    ORDER BY bar_time`,
    { conditionId, outcome }
  );
}
```

### 4.6 getSignalsOverview

```typescript
export async function getSignalsOverview(): Promise<SignalsOverview> {
  const rows = await query<SignalsOverview>(
    `SELECT
      (
        SELECT count(DISTINCT condition_id)
        FROM orderbook_snapshots
        WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
          AND arraySum(bid_sizes) + arraySum(ask_sizes) > 0
          AND abs(arraySum(bid_sizes) / (arraySum(bid_sizes) + arraySum(ask_sizes)) - 0.5) > 0.15
      ) AS obi_signals,
      (
        SELECT uniq(condition_id)
        FROM market_trades
        WHERE timestamp >= now() - INTERVAL 4 HOUR
      ) AS volume_active_markets,
      (
        SELECT count()
        FROM market_trades
        WHERE timestamp >= now() - INTERVAL 24 HOUR
          AND price * size >= 1000
      ) AS large_trades_24h,
      (
        SELECT countIf(active = 1 AND closed = 0)
        FROM markets FINAL
      ) AS active_markets`
  );
  return rows[0] ?? {
    obi_signals: 0,
    volume_active_markets: 0,
    large_trades_24h: 0,
    active_markets: 0,
  };
}
```

---

## 5. Dashboard Components & Page Structure

### 5.1 Page: `/signals` (`dashboard/src/app/signals/page.tsx`)

Server component with `export const dynamic = "force-dynamic"`. Follows the same Suspense pattern as the overview page.

**Layout:**
```
[Stats Cards Row] — 4 cards: OBI Signals, Volume Anomalies, Large Trades (24h), Active Markets
[Tabs: OBI | Volume | Large Trades]
  [OBI Tab]         → OBISignalsTable
  [Volume Tab]      → VolumeAnomaliesTable
  [Large Trades Tab] → LargeTradesTable
```

**Server sections (async functions inside page):**
```typescript
// Fetches
async function SignalsStatsSection() {
  const stats = await getSignalsOverview();
  return <SignalsStatsCards stats={stats} />;
}

async function OBISection() {
  const data = await getOrderBookImbalance(50);
  return <OBISignalsTable data={data} />;
}

async function VolumeSection() {
  const data = await getVolumeAnomalies(30);
  return <VolumeAnomaliesTable data={data} />;
}

async function LargeTradesSection() {
  const data = await getLargeTrades(1000, 50);
  return <LargeTradesTable data={data} />;
}
```

### 5.2 Component: `SignalsStatsCards` (`dashboard/src/components/signals-stats-cards.tsx`)

Client component ("use client"). Four stat cards in a row, same visual style as existing `StatsCards`.

| Card | Value | Icon (lucide-react) | Color |
|------|-------|---------------------|-------|
| OBI Signals | `stats.obi_signals` | `BookOpen` | `text-violet-400` |
| Volume Anomalies | `stats.volume_active_markets` | `TrendingUp` | `text-amber-400` |
| Large Trades (24h) | `stats.large_trades_24h` | `Zap` | `text-red-400` |
| Active Markets | `stats.active_markets` | `Activity` | `text-emerald-400` |

Include `SignalsStatsCardsSkeleton` export (same pattern as existing `StatsCardsSkeleton`).

### 5.3 Component: `OBISignalsTable` (`dashboard/src/components/obi-signals-table.tsx`)

Client component. Table with columns:

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Outcome | `outcome` | Badge |
| OBI | `obi` | Progress bar + percentage (green >0.6, red <0.4, gray neutral) |
| Bid Depth | `total_bid` | `formatUSD()` |
| Ask Depth | `total_ask` | `formatUSD()` |
| Spread | `best_ask - best_bid` | Formatted as cents |
| Signal | Derived from OBI | Badge: "Bullish" (green, OBI>0.6), "Bearish" (red, OBI<0.4), "Neutral" |
| Time | `snapshot_time` | Relative time (e.g., "2m ago") |

### 5.4 Component: `VolumeAnomaliesTable` (`dashboard/src/components/volume-anomalies-table.tsx`)

Client component. Table with columns:

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Price | `current_price` | `formatPrice()` |
| 4h Volume | `volume_4h` | `formatUSD()` |
| Avg Daily | `avg_daily_volume` | `formatUSD()` |
| Spike Ratio | `volume_ratio` | `Nx` format with color coding (>5x red, >3x amber, >2x yellow) |
| Trades | `trade_count` | `formatNumber()` |

### 5.5 Component: `LargeTradesTable` (`dashboard/src/components/large-trades-table.tsx`)

Client component. Table with columns:

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Outcome | `outcome` | Badge |
| Side | `side` | Badge: green "Buy", red "Sell" |
| Price | `price` | `formatPrice()` |
| Size | `usd_size` | `formatUSD()` |
| Tokens | `size` | `formatNumber()` |
| Time | `timestamp` | Relative time (e.g., "15m ago") |

### 5.6 Tab Navigation

Use the existing `@/components/ui/tabs` (Tabs, TabsList, TabsTrigger, TabsContent) from shadcn. The tabs component is a client component wrapping the server-fetched data. The page itself remains a server component.

**Pattern:** The page pre-fetches all three datasets and passes them as props. A client-side `SignalsTabs` wrapper handles tab switching without re-fetching.

```typescript
// In page.tsx
async function SignalsContent() {
  const [obi, volume, largeTrades] = await Promise.all([
    getOrderBookImbalance(50),
    getVolumeAnomalies(30),
    getLargeTrades(1000, 50),
  ]);
  return (
    <SignalsTabs
      obi={obi}
      volumeAnomalies={volume}
      largeTrades={largeTrades}
    />
  );
}
```

### 5.7 Component: `SignalsTabs` (`dashboard/src/components/signals-tabs.tsx`)

Client component. Wraps the three table components in tabs.

```typescript
"use client";

interface SignalsTabsProps {
  obi: OBISignal[];
  volumeAnomalies: VolumeAnomaly[];
  largeTrades: LargeTrade[];
}

export function SignalsTabs({ obi, volumeAnomalies, largeTrades }: SignalsTabsProps) {
  return (
    <Tabs defaultValue="obi">
      <TabsList>
        <TabsTrigger value="obi">Order Book ({obi.length})</TabsTrigger>
        <TabsTrigger value="volume">Volume Spikes ({volumeAnomalies.length})</TabsTrigger>
        <TabsTrigger value="large">Large Trades ({largeTrades.length})</TabsTrigger>
      </TabsList>
      <TabsContent value="obi"><OBISignalsTable data={obi} /></TabsContent>
      <TabsContent value="volume"><VolumeAnomaliesTable data={volumeAnomalies} /></TabsContent>
      <TabsContent value="large"><LargeTradesTable data={largeTrades} /></TabsContent>
    </Tabs>
  );
}
```

---

## 6. Navigation Update

Update `dashboard/src/app/layout.tsx` to add the `/signals` route to the sidebar nav.

**Change the `navItems` array:**

```typescript
const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/#markets", label: "Markets", icon: BarChart3 },
  { href: "/#trending", label: "Trending", icon: TrendingUp },
  { href: "/signals", label: "Signals", icon: Zap },
];
```

This replaces the old `/#movers` "Signals" link (which just scrolled to movers on the overview) with a proper `/signals` route.

---

## 7. File-by-File Change List

### New Files (create)

| File | Type | Description |
|------|------|-------------|
| `dashboard/src/app/signals/page.tsx` | Server Component | Signals page with stats + tabs |
| `dashboard/src/components/signals-stats-cards.tsx` | Client Component | Stats cards for signal counts |
| `dashboard/src/components/signals-tabs.tsx` | Client Component | Tab wrapper for 3 signal tables |
| `dashboard/src/components/obi-signals-table.tsx` | Client Component | OBI signals table |
| `dashboard/src/components/volume-anomalies-table.tsx` | Client Component | Volume anomaly table |
| `dashboard/src/components/large-trades-table.tsx` | Client Component | Large trades table |

### Modified Files (edit)

| File | Changes |
|------|---------|
| `dashboard/src/types/market.ts` | Add 5 new interfaces: `OBISignal`, `VolumeAnomaly`, `LargeTrade`, `TechnicalBar`, `SignalsOverview` |
| `dashboard/src/lib/queries.ts` | Add 5 new query functions: `getOrderBookImbalance`, `getVolumeAnomalies`, `getLargeTrades`, `getMarketTechnicals`, `getSignalsOverview`. Update import statement. |
| `dashboard/src/app/layout.tsx` | Update `navItems` to replace `/#movers` with `/signals` |

### Unchanged Files

| File | Notes |
|------|-------|
| `pipeline/schema/001_init.sql` | No DDL changes needed |
| `pipeline/config.py` | No pipeline changes needed |
| `dashboard/src/lib/clickhouse.ts` | Client singleton unchanged |
| `dashboard/src/lib/format.ts` | Existing formatters are sufficient |
| All existing components | No modifications needed |

---

## 8. Design Decisions & Rationale

1. **No materialized views for Phase 1**: Keeps deployment simple (no schema migration needed on the Azure VM). All signals are computed at query time. ClickHouse's columnar engine handles these analytical queries efficiently for the data volumes involved (top 100 orderbook markets, thousands of trades/day).

2. **All data from existing tables**: Phase 1 uses only `orderbook_snapshots`, `market_trades`, `ohlcv_1h`, and `markets`. No new pipeline jobs or API integrations.

3. **Server components with Suspense**: Matches the existing page pattern. Data is fetched server-side with `force-dynamic`, so every page load gets fresh ClickHouse results.

4. **Tabs instead of separate pages**: All three signal types on one page reduces navigation overhead. Users can quickly scan all active signals. The tabs client component receives pre-fetched data from the server, so tab switching is instant (no waterfall).

5. **`getMarketTechnicals` for per-market use**: RSI/VWAP/momentum are per-market time series — they are not displayed on the /signals overview (which shows cross-market aggregates). They will be used in Phase 1 on the market detail page or reserved for Phase 2 enhancements. Including the query function now establishes the interface.

6. **Signal thresholds**: OBI > 0.65 / < 0.35 for directional signal. Volume ratio > 2.0x for anomaly. Trade size >= $1,000 for large trade. These are configurable in the query parameters but start with sensible defaults.

7. **Relative timestamps**: Use JavaScript `Intl.RelativeTimeFormat` or a simple helper to show "2m ago", "1h ago" etc. in the trade/snapshot time columns.

---

## 9. Acceptance Criteria

- [ ] `/signals` page loads with 4 stats cards showing live counts
- [ ] OBI tab shows orderbook imbalance sorted by extremity, with bullish/bearish signal badges
- [ ] Volume Anomalies tab shows markets with >2x volume spike in last 4 hours
- [ ] Large Trades tab shows trades >=$1,000 in last 24 hours sorted by size
- [ ] All market names link to `/market/[condition_id]`
- [ ] Sidebar navigation includes "Signals" link pointing to `/signals`
- [ ] `npm run build` succeeds with no type errors
- [ ] Page handles empty data gracefully (empty state messages in tables)
- [ ] Existing pages (overview, market detail) are unaffected
