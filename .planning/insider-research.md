# Insider Trading Detection — Codebase Research

Comprehensive analysis of existing patterns, data, and conventions to guide the implementation of enhanced insider trading detection for Polymarket.

---

## 1. Schema Patterns & Conventions

### Naming Conventions
- **Table names**: `snake_case`, plural nouns (e.g., `insider_scores`, `wallet_clusters`, `composite_signals`)
- **Column names**: `snake_case` throughout
- **Primary keys**: use domain identifiers (`condition_id`, `proxy_wallet`, `cluster_id`), not auto-increment
- **Timestamps**: `DateTime64(3)` with codec `CODEC(DoubleDelta, LZ4)` for time-series, `DEFAULT now64(3)` for tracking columns
- **Materialized dates**: `ts_date Date MATERIALIZED toDate(timestamp)` pattern for partition alignment

### Engine Choices
| Pattern | Engine | When to Use |
|---------|--------|-------------|
| Dedup by key | `ReplacingMergeTree(version_col)` | Most analytics tables — latest state wins (insider_scores, composite_signals, wallet_clusters) |
| Append-only time-series | `MergeTree()` | Trade history, predictions, activity logs (market_trades, wallet_activity, gnn_predictions) |
| Pre-aggregated rollups | `AggregatingMergeTree()` / `SummingMergeTree()` | OHLCV, daily volumes, sentiment rollups |

### Partition & TTL Patterns
- **Monthly partitions**: `PARTITION BY toYYYYMM(timestamp)` for time-series
- **Daily partitions**: `PARTITION BY toYYYYMMDD(snapshot_time)` for high-churn short-TTL data (orderbook, microstructure)
- **No partitions**: Small ReplacingMergeTree tables (insider_scores, trader_profiles)
- **TTL ranges**: 7 days (orderbooks), 30 days (arbitrage, microstructure, similarity), 90 days (clusters, insider_scores, execution), 1 year (wallet_activity, gnn_predictions), 2 years (market_prices, trades)

### ORDER BY Conventions
- Always starts with the primary lookup key(s)
- ReplacingMergeTree: ORDER BY = dedup key (e.g., `ORDER BY (proxy_wallet)` for insider_scores)
- Time-series MergeTree: ORDER BY includes timestamp last (e.g., `ORDER BY (condition_id, outcome, timestamp)`)

### Index Patterns
- `bloom_filter GRANULARITY 4` for lookup columns (condition_id, proxy_wallet)
- `set(N) GRANULARITY 4` for low-cardinality enum-like columns (status, category, activity_type)
- `minmax GRANULARITY 4` for numeric range queries (score)
- `tokenbf_v1(N, 3, 0) GRANULARITY 4` for text search (question, pseudonym)

### Column Type Patterns
- `LowCardinality(String)` for repeated string values (condition_id as FK, category, status, side)
- `String` for unique identifiers (proxy_wallet, cluster_id, trade_id)
- `Float64` for all numeric metrics with CODEC `CODEC(Gorilla, LZ4)` for price data
- `UInt8` for boolean flags (active, closed, resolved, verified_badge)
- `String CODEC(ZSTD(3))` for JSON blobs and long text (factors, components, description)
- `Array(String)` for wallet lists, market lists

### JSON Column Convention
- Store flexible data as JSON string with `CODEC(ZSTD(3))`
- ALSO denormalize key fields as separate typed columns for query/sort
- Example from `insider_scores`: `factors String DEFAULT '{}' CODEC(ZSTD(3))` PLUS individual `freshness_score Float64`, `win_rate_score Float64`, etc.

---

## 2. Pipeline Job Patterns

### Job Structure (async function)
Every Phase 3+ job follows this exact pattern:

```python
"""Job: <description>."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    # ... job-specific config constants
)

logger = logging.getLogger(__name__)


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    """Create a read-only ClickHouse client for queries."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


async def run_<job_name>() -> None:
    """<Docstring explaining the job steps>."""
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # ... query logic using:
        #   result = await asyncio.to_thread(client.query, "SQL")
        #   for row in result.result_rows: ...

        # ... computation logic ...

        # ... write results:
        #   rows: list[list] = [...]
        #   await writer.write_<table>(rows)
        #   await writer.flush_all()

        logger.info(
            "<job_name>_complete",
            extra={"metric1": val1, "metric2": val2},
        )

    except Exception:
        logger.error("<job_name>_error", exc_info=True)
```

### Key Patterns
1. **Read client**: Uses `clickhouse_connect` synchronous client wrapped in `asyncio.to_thread()` for reads
2. **Write client**: Uses singleton `ClickHouseWriter.get_instance()` for buffered writes
3. **Row format**: Rows are `list[list[Any]]` — positional, matching TABLE_COLUMNS order in clickhouse_writer.py
4. **Error handling**: Single `try/except` wrapping the entire job, logging with `exc_info=True`
5. **Logging**: Structured JSON logging via `logger.info("event_name", extra={...})`
6. **ClickHouse FINAL**: Must wrap in subquery for JOINs: `FROM (SELECT * FROM table FINAL) AS alias`

### Scheduler Registration Pattern (`scheduler.py`)

```python
# In __init__ imports:
from pipeline.jobs.<module> import run_<job_name>

# In start() method:
self._scheduler.add_job(
    self._job_<name>,
    "interval",
    seconds=<INTERVAL_CONSTANT>,
    id="<job_id>",
    name="<Human Name>",
)

# Job wrapper method:
async def _job_<name>(self) -> None:
    try:
        await run_<job_name>()
    except Exception:
        logger.error("<job_id>_error", exc_info=True)
```

### ClickHouseWriter Registration Pattern

In `clickhouse_writer.py`:
1. Add column list to `TABLE_COLUMNS` dict
2. Add convenience method: `async def write_<table>(self, rows) -> None: await self.write("<table>", rows)`

### Config Constants Pattern (`config.py`)

```python
# Grouped by phase in comment-delimited sections
JOB_NAME_INTERVAL = <seconds>     # Comment with human-readable time
JOB_TUNING_PARAM = <value>        # Comment explaining purpose
```

### Migration Pattern (`migrate.py`)

Add the new SQL filename to the `SCHEMA_FILES` list in order.

---

## 3. Existing Insider Detection — Coverage & Gaps

### Current `insider_scores` Table (003_phase3_analytics.sql)

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS insider_scores (
    proxy_wallet       String,
    score              Float64 DEFAULT 0,       -- 0-100 composite
    factors            String DEFAULT '{}',     -- JSON breakdown
    freshness_score    Float64 DEFAULT 0,       -- 0-100
    win_rate_score     Float64 DEFAULT 0,       -- 0-100
    niche_score        Float64 DEFAULT 0,       -- 0-100
    size_score         Float64 DEFAULT 0,       -- 0-100
    timing_score       Float64 DEFAULT 0,       -- 0-100 (PLACEHOLDER = 0)
    computed_at        DateTime64(3)
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (proxy_wallet)
TTL 90 DAY
```

### Current Scoring (wallet_analyzer.py)

| Factor | Weight | Implementation | Quality |
|--------|--------|----------------|---------|
| Freshness | 0.20 | Wallet age vs INSIDER_FRESHNESS_DAYS (30d) | Working but coarse |
| Win Rate | 0.30 | Wins / resolved positions (min 3 resolved) | Working, uses wallet_positions + markets |
| Niche Focus | 0.15 | Count of unique markets (<=3 = 80, <=5 = 40) | Very simplistic discrete buckets |
| Size vs Liquidity | 0.25 | avg(current_value / liquidity) | Working |
| Timing | 0.10 | **PLACEHOLDER = 0.0 always** | Not implemented |

### Critical Gaps

1. **Timing score is 0.0** — The most important insider signal (trading before announcements/resolutions) is not computed. Comment in code: "requires event resolution timestamps which we do not yet track."

2. **No pre-event trading analysis** — No tracking of trades that occur shortly before market resolution or major price movements.

3. **No trade-level granularity** — Current scoring is wallet-level only; doesn't analyze individual suspicious trades.

4. **No market-specific insider detection** — Current system scores wallets globally, not per-market. A wallet might be "insider" in one market category but legitimate in another.

5. **No temporal patterns** — No detection of timing patterns (e.g., consistently trading 1-2 hours before major moves).

6. **No cross-reference with market events** — The `market_events` table tracks resolutions but isn't used by the insider scorer.

7. **No alert/notification system** — Insider scores are computed but there's no threshold-based alerting or trend detection.

8. **No historical insider score tracking** — ReplacingMergeTree overwrites per wallet. No history of how scores evolve over time.

9. **No "early mover" detection** — Not analyzing whether wallets consistently enter positions before the market moves.

10. **Existing composite signal integration is minimal** — `signal_compositor.py` uses insider_scores only as `avg(score)` per market, ignoring per-trade timing data.

---

## 4. Available Data for Insider Detection

### Data Sources in ClickHouse

| Table | Relevant Columns | Insider Signal Value |
|-------|-------------------|---------------------|
| `wallet_activity` | proxy_wallet, condition_id, side, usdc_size, price, timestamp, activity_type | **HIGH** — trade-level timing data |
| `market_trades` | condition_id, price, size, side, timestamp | **HIGH** — market-level trade flow |
| `markets` | condition_id, resolved, winning_outcome, end_date, volume_24h, liquidity | **HIGH** — resolution events |
| `market_events` | condition_id, event_type, event_data, event_time | **HIGH** — resolution timestamps |
| `wallet_positions` | proxy_wallet, condition_id, outcome, size, avg_price, cash_pnl, percent_pnl | **MEDIUM** — current state of holdings |
| `trader_rankings` | proxy_wallet, rank, category, pnl, volume | **MEDIUM** — reputation context |
| `trader_profiles` | proxy_wallet, pseudonym, first_seen_at, profile_created_at | **MEDIUM** — wallet age/identity |
| `market_holders` | condition_id, proxy_wallet, amount, outcome_index | **MEDIUM** — position concentration |
| `insider_scores` | proxy_wallet, score, factor breakdown | **LOW** — existing scores to enhance |
| `composite_signals` | condition_id, smart_money_score, insider_activity | **LOW** — downstream consumer |
| `ohlcv_1h` | condition_id, bar_time, OHLCV aggregates | **MEDIUM** — price movement context |
| `market_microstructure` | condition_id, toxic_flow_ratio, kyle_lambda, price_impact_1m | **HIGH** — informed trading signals |
| `news_articles` | settlements_mentioned, markets_mentioned, published_at | **MEDIUM** — news timing correlation |

### Key Queries Available

**1. Pre-resolution trading (trade timing vs resolution):**
```sql
-- Trades by wallet in the 24h before market resolution
SELECT wa.proxy_wallet, wa.condition_id, wa.side, wa.usdc_size, wa.timestamp,
       me.event_time AS resolution_time,
       dateDiff('hour', wa.timestamp, me.event_time) AS hours_before_resolution
FROM wallet_activity wa
INNER JOIN market_events me ON wa.condition_id = me.condition_id
WHERE me.event_type = 'resolved'
  AND wa.timestamp >= me.event_time - INTERVAL 24 HOUR
  AND wa.timestamp < me.event_time
  AND wa.activity_type = 'TRADE'
```

**2. Correctness of pre-resolution trades:**
```sql
-- Did they bet on the winning outcome?
SELECT wa.proxy_wallet, wa.condition_id, wa.side, wa.outcome,
       m.winning_outcome,
       wa.outcome = m.winning_outcome AS correct_bet
FROM wallet_activity wa
INNER JOIN (SELECT * FROM markets FINAL WHERE resolved = 1) m
  ON wa.condition_id = m.condition_id
WHERE wa.activity_type = 'TRADE'
  AND wa.timestamp >= m.end_date - INTERVAL 24 HOUR
```

**3. Early mover detection (trading before price moves):**
```sql
-- Compare wallet entry price to subsequent price movement
SELECT wa.proxy_wallet, wa.condition_id, wa.price AS entry_price,
       last_price.price AS final_price,
       abs(last_price.price - wa.price) AS price_move
FROM wallet_activity wa
INNER JOIN (
    SELECT condition_id, argMax(price, timestamp) AS price
    FROM market_prices
    GROUP BY condition_id
) last_price ON wa.condition_id = last_price.condition_id
```

**4. Toxic flow from microstructure:**
```sql
-- Markets with high toxic flow and insider scores
SELECT mm.condition_id, mm.toxic_flow_ratio, mm.kyle_lambda,
       avg(ins.score) AS avg_insider_score
FROM market_microstructure mm
INNER JOIN wallet_activity wa ON mm.condition_id = wa.condition_id
INNER JOIN (SELECT * FROM insider_scores FINAL) ins ON wa.proxy_wallet = ins.proxy_wallet
GROUP BY mm.condition_id, mm.toxic_flow_ratio, mm.kyle_lambda
```

### Data Volume Estimates (from config.py)
- **Tracked wallets**: up to 500 (`TRACKED_WALLET_MAX`)
- **Position sync**: every 5 minutes
- **Wallet activity**: append-only, 1-year TTL
- **Markets scored**: top 500 by volume
- **Leaderboard wallets**: 200 per category/period/order combo

---

## 5. Dashboard Component Patterns

### Page Structure Pattern

```tsx
// app/<page>/page.tsx
import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getOverviewQuery, getDataQuery } from "@/lib/queries";
import { StatsCards, StatsCardsSkeleton } from "@/components/<page>-stats-cards";
import { TabComponent } from "@/components/<page>-tabs";

export const dynamic = "force-dynamic";

async function StatsSection() {
  const stats = await getOverviewQuery();
  return <StatsCards stats={stats} />;
}

async function ContentSection() {
  const [data1, data2] = await Promise.all([
    getDataQuery1(),
    getDataQuery2(),
  ]);
  return <TabComponent data1={data1} data2={data2} />;
}

export default function Page() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Title</h1>
        <p className="text-muted-foreground text-sm mt-1">Description</p>
      </div>
      <Suspense fallback={<StatsCardsSkeleton />}>
        <StatsSection />
      </Suspense>
      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Section Title</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<SkeletonLoader />}>
            <ContentSection />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
```

### Stats Cards Pattern

```tsx
// components/<page>-stats-cards.tsx
"use client";
import { Card, CardContent } from "@/components/ui/card";
import { IconName } from "lucide-react";
import { formatNumber } from "@/lib/format";

export function StatsCards({ stats }) {
  const cards = [
    { label: "Label", value: formatNumber(stats.field), icon: Icon, color: "text-<color>-400" },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-<N> gap-4">
      {cards.map((card) => (
        <Card key={card.label} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-2 mb-1">
              <card.icon className={`h-4 w-4 ${card.color}`} />
              <span className="text-xs text-muted-foreground">{card.label}</span>
            </div>
            <p className="text-xl font-bold">{card.value}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function StatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-<N> gap-4">
      {Array.from({ length: N }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse mb-2" />
            <div className="h-6 w-16 bg-[#1e1e2e] rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
```

### Tabs Pattern

```tsx
// components/<page>-tabs.tsx
"use client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table1 } from "./table1";
import { Table2 } from "./table2";

export function PageTabs({ data1, data2 }) {
  return (
    <Tabs defaultValue="tab1">
      <TabsList>
        <TabsTrigger value="tab1">Tab 1 ({data1.length})</TabsTrigger>
        <TabsTrigger value="tab2">Tab 2 ({data2.length})</TabsTrigger>
      </TabsList>
      <TabsContent value="tab1"><Table1 data={data1} /></TabsContent>
      <TabsContent value="tab2"><Table2 data={data2} /></TabsContent>
    </Tabs>
  );
}
```

### Table Component Pattern

```tsx
// components/<name>-table.tsx
"use client";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export function DataTable({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No data available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Column</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow key={row.id} className="border-[#1e1e2e] hover:bg-[#1a1a2e] transition-colors">
              <TableCell>...</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
```

### Visual Patterns
- **Score bars**: Progress bar with dynamic width (`style={{ width: \`${score}%\` }}`)
- **Color coding**: Red (`#ff4466`, `text-red-400`) for alerts/high risk, Amber for medium, Yellow/Green for low
- **Factor breakdown bars**: Thin (`h-1.5`) horizontal bars per factor with muted fill
- **Wallet display**: `truncateWallet()` function: `${wallet.slice(0, 6)}...${wallet.slice(-4)}`
- **Time display**: `formatRelativeTime()` function: `Xs ago`, `Xm ago`, `Xh ago`, `Xd ago`
- **Dark theme colors**: Background `bg-[#111118]`, borders `border-[#1e1e2e]`, hover `hover:bg-[#1a1a2e]`

### Query Pattern

```typescript
// lib/queries.ts
export async function getDataFunction(
  param = defaultValue,
  limit = 50
): Promise<TypeName[]> {
  return query<TypeName>(
    `SELECT ...
    FROM table FINAL
    WHERE ...
    ORDER BY ...
    LIMIT {limit:UInt32}`,
    { param, limit }
  );
}
```

Key conventions:
- Use parameterized queries with `{name:Type}` syntax
- Wrap ReplacingMergeTree tables in subquery for JOINs: `FROM (SELECT * FROM table FINAL) AS alias`
- All types defined in `types/market.ts`
- All queries in `lib/queries.ts`
- Overview/stats queries return a single row with scalar aggregates

### Navigation Pattern

Navigation items are defined in `layout.tsx`:
```tsx
const navItems = [
  { href: "/path", label: "Label", icon: LucideIcon },
];
```

Each nav item has an icon from `lucide-react` and links to a top-level route.

---

## 6. Recommended Design Decisions

### Schema: Extend vs New Table
**Recommend: New table(s)** — The existing `insider_scores` table has a flat structure with 5 factor columns. An enhanced insider detection system needs:
- Per-trade suspicious trade records (new append-only table)
- Enhanced wallet-level scores (extend or replace existing `insider_scores`)
- Market-level insider risk (could be a new table or enhance `composite_signals`)

### Table Design Recommendations

**Table 1: `insider_trades`** (new, MergeTree, append-only)
- Individual suspicious trades with evidence
- ORDER BY: (proxy_wallet, condition_id, timestamp)
- TTL: 90 days
- Fields: proxy_wallet, condition_id, side, outcome, size, usdc_size, price, timestamp, resolution_time, hours_before_resolution, bet_correct, suspicion_score, flags (JSON)

**Table 2: Enhanced `insider_scores`** (modify existing)
- Add new factor columns: `timing_score` (fill the placeholder), `early_mover_score`, `resolution_accuracy_score`, `pattern_score`
- Add `suspicious_trade_count`, `avg_trade_suspicion`
- Keep backward compatibility with existing column names

### Pipeline Job Recommendations
- New job: `insider_detector.py` — run every 5-10 minutes
- Reads: wallet_activity, market_events, markets, market_trades, market_microstructure
- Writes: insider_trades (new), insider_scores (enhanced)
- Should replace the scoring logic in wallet_analyzer.py or coexist alongside it

### Dashboard Recommendations
- Can be a new page (`/insider`) with dedicated URL, or enhance existing Analytics `/analytics` Insider tab
- Recommend new page for depth, with link from analytics
- Tab structure: Suspicious Trades | Wallet Scores | Market Risk | Timeline
