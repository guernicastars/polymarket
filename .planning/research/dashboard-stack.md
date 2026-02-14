# Polymarket Analytics Dashboard - Technology Stack Research

> Research compiled 2026-02-14. All package versions verified against npm/PyPI.

---

## Table of Contents

1. [Next.js 15 + ClickHouse Cloud Integration](#1-nextjs-15--clickhouse-cloud-integration)
2. [clickhouse-js Client Library](#2-clickhouse-js-client-library)
3. [Real-Time Dashboard Patterns on Vercel](#3-real-time-dashboard-patterns-on-vercel)
4. [Chart Libraries for Financial/Market Data](#4-chart-libraries-for-financialmarket-data)
5. [Vercel Deployment Considerations](#5-vercel-deployment-considerations)
6. [Docker Compose for Python Data Pipeline](#6-docker-compose-for-python-data-pipeline)
7. [Dashboard Structure for Betting Signals](#7-dashboard-structure-for-betting-signals)
8. [Tailwind CSS + shadcn/ui Dashboard Layout](#8-tailwind-css--shadcnui-dashboard-layout)
9. [Polymarket API Endpoints Reference](#9-polymarket-api-endpoints-reference)
10. [Recommended Architecture Summary](#10-recommended-architecture-summary)

---

## 1. Next.js 15 + ClickHouse Cloud Integration

### Architecture Pattern: Server Components + Route Handlers

Next.js 15 App Router server components are ideal for ClickHouse queries because they run server-side only, keeping database credentials off the client.

**Recommended pattern:**

```
app/
  lib/
    clickhouse.ts          # Singleton client (shared across server components)
  api/
    markets/route.ts       # Route handler for client-side polling
    prices/route.ts        # Route handler for price history
  dashboard/
    page.tsx               # Server component - initial data load
    components/
      PriceChart.tsx       # Client component - receives data via props + polls
      MarketTable.tsx      # Client component - interactive table
```

### Server Component Data Loading

```typescript
// app/lib/clickhouse.ts
import { createClient } from '@clickhouse/client';

// Singleton pattern - reuse across serverless invocations
const client = createClient({
  url: process.env.CLICKHOUSE_URL,       // e.g., https://<id>.<region>.clickhouse.cloud:8443
  username: process.env.CLICKHOUSE_USER,  // default
  password: process.env.CLICKHOUSE_PASSWORD,
  database: process.env.CLICKHOUSE_DB,
  request_timeout: 30_000,
  clickhouse_settings: {
    async_insert: 1,
    wait_for_async_insert: 1,
  },
});

export default client;
```

```typescript
// app/dashboard/page.tsx (Server Component)
import client from '@/lib/clickhouse';

export default async function DashboardPage() {
  const result = await client.query({
    query: `SELECT * FROM markets ORDER BY volume_24h DESC LIMIT 50`,
    format: 'JSONEachRow',
  });
  const markets = await result.json();

  return (
    <div>
      <MarketTable initialData={markets} />
      <PriceChart initialData={markets} />
    </div>
  );
}
```

### Route Handlers for Client-Side Polling

```typescript
// app/api/markets/route.ts
import { NextResponse } from 'next/server';
import client from '@/lib/clickhouse';

export const dynamic = 'force-dynamic'; // Required: prevents caching

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') ?? '50');

  const result = await client.query({
    query: 'SELECT * FROM markets ORDER BY volume_24h DESC LIMIT {limit:UInt32}',
    query_params: { limit },
    format: 'JSONEachRow',
  });

  const data = await result.json();
  return NextResponse.json(data);
}
```

### Key Principles

- **Server Components** for initial page load (SSR, no client JS, direct DB access)
- **Route Handlers** (`app/api/`) for client-side polling endpoints
- **Singleton client** exported from `lib/clickhouse.ts` to reuse connections
- **Parameterized queries** using `{name:Type}` syntax to prevent SQL injection
- Always use `format: 'JSONEachRow'` for typed JSON results
- Set `export const dynamic = 'force-dynamic'` on route handlers to prevent Vercel caching

---

## 2. clickhouse-js Client Library

### Package: `@clickhouse/client` v1.17.0

**Latest version:** 1.17.0 (released Feb 6, 2026)

```bash
npm install @clickhouse/client    # Node.js (server components, route handlers)
npm install @clickhouse/client-web # Browser/Edge (NOT recommended for our use case)
```

### Two Client Variants

| Feature | `@clickhouse/client` | `@clickhouse/client-web` |
|---------|---------------------|-------------------------|
| Runtime | Node.js only | Browser, CloudFlare Workers |
| Transport | HTTP + Node Streams | Fetch + Web Streams |
| Select streaming | Yes | Yes |
| Insert streaming | Yes | No |
| Vercel Serverless | Yes (Node.js runtime) | Possibly Edge, untested |

**Recommendation:** Use `@clickhouse/client` (Node.js variant) in server components and route handlers. Do NOT use edge runtime for ClickHouse queries -- stick with Node.js runtime.

### ClickHouse Cloud Connection

```typescript
import { createClient } from '@clickhouse/client';

const client = createClient({
  url: 'https://<random_id>.<region>.clickhouse.cloud:8443', // HTTPS, port 8443
  username: 'default',
  password: process.env.CLICKHOUSE_PASSWORD,
  database: 'polymarket',
  application: 'polymarket-dashboard',
  request_timeout: 30_000,
  compression: {
    request: true,  // Compress outgoing data (inserts)
    response: true, // Decompress incoming data (queries)
  },
  clickhouse_settings: {
    async_insert: 1,
    wait_for_async_insert: 1,
  },
});
```

### Serverless Considerations

- **No built-in connection pooling:** The client uses HTTP(S), so each request creates a new connection. This is actually fine for serverless since connections are short-lived.
- **Keep-alive:** ClickHouse Cloud keeps HTTPS connections alive; the client handles reconnection transparently.
- **Cold start:** Client instantiation is lightweight (~2ms). Create a singleton at module scope and it persists across warm invocations.
- **Timeouts:** Set `request_timeout` to stay within Vercel's function duration limits.

### Query Patterns

```typescript
// Parameterized query (SQL injection safe)
const result = await client.query({
  query: `
    SELECT condition_id, question, outcome_prices, volume_24h
    FROM markets
    WHERE active = 1
    ORDER BY {order_col:Identifier} DESC
    LIMIT {limit:UInt32}
  `,
  query_params: { order_col: 'volume_24h', limit: 50 },
  format: 'JSONEachRow',
});
const rows = await result.json<MarketRow[]>();

// Batch insert (for pipeline data)
await client.insert({
  table: 'price_snapshots',
  values: snapshots, // Array of objects
  format: 'JSONEachRow',
});
```

---

## 3. Real-Time Dashboard Patterns on Vercel

### Option Comparison

| Approach | Latency | Complexity | Vercel Support | Recommendation |
|----------|---------|------------|---------------|----------------|
| SWR polling | 1-30s | Low | Full | **Primary choice** |
| SSE (Server-Sent Events) | Real-time | Medium | Partial (25s timeout) | For specific widgets |
| WebSocket | Real-time | High | Not supported natively | Avoid on Vercel |
| React Server Components + revalidation | 1-60s | Low | Full | For initial loads |

### Recommended: SWR Polling (Primary)

SWR (stale-while-revalidate) from Vercel is the best fit for a dashboard on Vercel:

```bash
npm install swr
```

```typescript
'use client';
import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then(r => r.json());

function MarketTable({ initialData }: { initialData: Market[] }) {
  const { data: markets } = useSWR('/api/markets', fetcher, {
    fallbackData: initialData,     // SSR data from server component
    refreshInterval: 10_000,       // Poll every 10 seconds
    revalidateOnFocus: true,       // Refresh when user returns to tab
    dedupingInterval: 5_000,       // Dedup requests within 5s window
  });

  return <Table data={markets} />;
}
```

**Why SWR over React Query:**
- SWR is from Vercel, tighter Next.js integration
- Lighter bundle (4.2kB vs ~13kB for React Query)
- Built-in focus revalidation and interval polling
- SWRConfig allows global provider in layout.tsx

### SSE for Specific Widgets

SSE works on Vercel with caveats (25s initial response timeout, 300s streaming limit on Edge):

```typescript
// app/api/stream/route.ts
export const runtime = 'edge'; // Required for SSE on Vercel

export async function GET() {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // Must send first byte within 25 seconds
      const send = (data: object) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };

      // Poll ClickHouse and stream updates
      const interval = setInterval(async () => {
        const data = await fetchLatestPrices(); // Use fetch to CH Cloud HTTP API
        send(data);
      }, 5000);

      // Clean up after 4 minutes (below 300s edge limit)
      setTimeout(() => {
        clearInterval(interval);
        controller.close();
      }, 240_000);
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
```

### Hybrid Pattern (Recommended)

```
Initial page load → Server Component (SSR from ClickHouse)
         ↓
Client hydration → SWR with fallbackData (SSR data, no flash)
         ↓
Background polling → SWR refreshInterval (10s for tables, 30s for charts)
         ↓
Optional SSE → For a single "live ticker" widget
```

---

## 4. Chart Libraries for Financial/Market Data

### Comparison Matrix

| Library | Bundle Size | Max Points | Type | Best For |
|---------|------------|------------|------|----------|
| **TradingView Lightweight Charts** | 45 kB | 100K+ | Canvas | Price charts, candlesticks |
| **Recharts** | ~50 kB | <1000 | SVG/DOM | Bar charts, area charts, pies |
| **D3.js** | ~90 kB | Unlimited | SVG/Canvas | Custom visualizations, heatmaps |
| **visx** | Modular | ~5000 | SVG | React-native D3 components |

### Primary: TradingView Lightweight Charts (for price data)

```bash
npm install lightweight-charts@^4.2.0
```

**Why:** 45kB, GPU-accelerated canvas rendering, handles 100K+ data points smoothly. Built specifically for financial data with candlestick, area, line, bar, and histogram series.

```typescript
'use client'; // Must be client component (uses DOM)

import { createChart, ColorType, IChartApi } from 'lightweight-charts';
import { useEffect, useRef } from 'react';

interface PricePoint { time: number; value: number; }

export function PriceChart({ data }: { data: PricePoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: containerRef.current.clientWidth,
      height: 400,
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const series = chart.addAreaSeries({
      lineColor: '#22c55e',
      topColor: 'rgba(34, 197, 94, 0.4)',
      bottomColor: 'rgba(34, 197, 94, 0.0)',
      lineWidth: 2,
    });

    series.setData(data.map(d => ({
      time: d.time as any, // Unix timestamp
      value: d.value,
    })));

    chart.timeScale().fitContent();
    chartRef.current = chart;

    const handleResize = () => {
      chart.applyOptions({ width: containerRef.current!.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data]);

  return <div ref={containerRef} className="w-full" />;
}
```

### Secondary: Recharts (for summary/aggregate charts)

```bash
npm install recharts@^2.15.0
```

**Why:** Integrates natively with shadcn/ui chart components. Good for bar charts (volume by category), pie charts (market distribution), and area charts with <1000 data points. SVG-based so it renders poorly with large datasets.

```typescript
'use client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export function VolumeChart({ data }: { data: { name: string; volume: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <XAxis dataKey="name" stroke="#6b7280" />
        <YAxis stroke="#6b7280" />
        <Tooltip />
        <Bar dataKey="volume" fill="#3b82f6" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
```

### Heatmap: Custom D3-based or recharts Treemap

For volume heatmaps, use either:
- **Recharts Treemap** for simple category heatmaps
- **D3 + Canvas** for time-based heatmaps (volume over time-of-day grid)

### Decision

| Component | Library |
|-----------|---------|
| Price line/area charts | TradingView Lightweight Charts |
| Candlestick charts | TradingView Lightweight Charts |
| Volume bar charts | Recharts (via shadcn/ui Charts) |
| Category heatmaps | Recharts Treemap |
| Market distribution pies | Recharts |
| Sparklines in tables | TradingView Lightweight Charts (mini) |

---

## 5. Vercel Deployment Considerations

### Runtime Limits

| Limit | Hobby (Free) | Pro | Enterprise |
|-------|-------------|-----|------------|
| Serverless fn duration | 10s | 60s | 900s |
| Serverless fn memory | 1024 MB | 1024 MB | 3008 MB |
| Edge fn duration | 25s initial, 300s streaming | Same | Same |
| Serverless fn size | 50 MB (compressed) | 50 MB | 250 MB |
| Edge fn size | 1-4 MB (with deps) | Same | Same |
| Fluid Compute duration | 60s | 800s | 800s |

### Runtime Selection

```typescript
// app/api/markets/route.ts
// Use Node.js runtime (DEFAULT) - for ClickHouse queries
export const runtime = 'nodejs'; // This is the default, explicit for clarity

// app/api/stream/route.ts
// Use Edge runtime ONLY for SSE streaming
export const runtime = 'edge';
```

**Rule:** Use Node.js runtime for all ClickHouse queries. The `@clickhouse/client` package requires Node.js APIs (streams, HTTP agent). Edge runtime is only for SSE/lightweight proxying.

### Environment Variables

Set in Vercel Dashboard > Project > Settings > Environment Variables:

```
CLICKHOUSE_URL=https://<id>.<region>.clickhouse.cloud:8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=<password>
CLICKHOUSE_DB=polymarket

# Optional: for pipeline API
POLYMARKET_CLOB_URL=https://clob.polymarket.com
POLYMARKET_GAMMA_URL=https://gamma-api.polymarket.com
```

### Deployment Configuration

```json
// next.config.ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Required: @clickhouse/client uses Node.js APIs not available in edge
  serverExternalPackages: ['@clickhouse/client'],
};

export default nextConfig;
```

### Vercel-Specific Optimizations

1. **ISR for static pages:** Use `revalidate` for pages that don't need real-time data
2. **Route segment config:** Set `dynamic = 'force-dynamic'` on API routes
3. **Streaming:** Use React Suspense boundaries for progressive loading
4. **Cold starts:** Keep serverless functions warm with cron jobs (Vercel Cron)

```typescript
// vercel.json
{
  "crons": [
    {
      "path": "/api/cron/warm",
      "schedule": "*/5 * * * *"  // Every 5 minutes
    }
  ]
}
```

---

## 6. Docker Compose for Python Data Pipeline

### Architecture

```
docker-compose.yml
  pipeline/          # Python data pipeline
    Dockerfile
    requirements.txt
    main.py          # Entry point: fetch -> transform -> load
    fetcher.py       # Polymarket API client
    transformer.py   # Data transformation
    loader.py        # ClickHouse loader
    cron/            # Cron schedule configs
```

### Dockerfile (Python Pipeline)

```dockerfile
# pipeline/Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
```

### requirements.txt

```
httpx==0.28.1
clickhouse-connect==0.8.14
schedule==1.2.2
pydantic==2.10.5
python-json-logger==3.2.1
```

### compose.yaml (Modern naming)

```yaml
# compose.yaml
services:
  pipeline:
    build:
      context: ./pipeline
      dockerfile: Dockerfile
    environment:
      - CLICKHOUSE_URL=${CLICKHOUSE_URL}
      - CLICKHOUSE_USER=${CLICKHOUSE_USER}
      - CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
      - CLICKHOUSE_DB=${CLICKHOUSE_DB}
      - FETCH_INTERVAL_SECONDS=300
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
    healthcheck:
      test: ["CMD", "python", "-c", "print('ok')"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - pipeline-data:/app/data  # Persist state between restarts
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  pipeline-data:
```

### Pipeline Design Pattern

```python
# pipeline/main.py
import schedule
import time
import logging
from fetcher import PolymarketFetcher
from transformer import DataTransformer
from loader import ClickHouseLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Single pipeline run: fetch -> transform -> load."""
    try:
        fetcher = PolymarketFetcher()
        transformer = DataTransformer()
        loader = ClickHouseLoader()

        # Fetch from Polymarket APIs
        events = fetcher.fetch_active_events()
        markets = fetcher.fetch_markets_for_events(events)
        prices = fetcher.fetch_price_history(markets)

        # Transform
        records = transformer.transform(events, markets, prices)

        # Load to ClickHouse
        loader.insert_batch(records)
        logger.info(f"Pipeline completed: {len(records)} records loaded")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

# Run every 5 minutes
schedule.every(5).minutes.do(run_pipeline)

# Initial run
run_pipeline()

while True:
    schedule.run_pending()
    time.sleep(1)
```

### Single-Command Deployment

```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f pipeline

# Rebuild after code changes
docker compose up -d --build pipeline

# Stop
docker compose down
```

---

## 7. Dashboard Structure for Betting Signals

### Page Layout (4 main sections)

```
+--------------------------------------------------+
|  HEADER: Search bar | Filters | Time range toggle |
+--------------------------------------------------+
|                                                    |
|  SECTION 1: Market Overview Cards                  |
|  [Total Volume] [Active Markets] [24h Change] ... |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 2: Featured Market Price Chart            |
|  [TradingView Lightweight Chart - full width]      |
|  [Time range: 1h | 6h | 1d | 1w | max]          |
|                                                    |
+--------------------------------------------------+
|                       |                            |
|  SECTION 3a:          |  SECTION 3b:              |
|  Trending Markets     |  Volume Heatmap           |
|  Table (sortable)     |  (by category/time)       |
|                       |                            |
+--------------------------------------------------+
|                                                    |
|  SECTION 4: Momentum Signals / Market Scanner      |
|  [Biggest movers] [Volume spikes] [New markets]   |
|                                                    |
+--------------------------------------------------+
```

### Key Dashboard Components

#### 1. Overview KPI Cards
- Total 24h volume across all markets
- Number of active markets
- Total open interest
- Number of markets with >10% price move in 24h

#### 2. Price Chart (Featured Market)
- TradingView Lightweight Charts area/line chart
- Clickable market selector
- Time range toggles: 1m, 1h, 6h, 1d, 1w, max
- Volume histogram overlay
- Current price + 24h change percentage

#### 3a. Trending Markets Table
- Sortable columns: Market name, Price (Yes), 24h Volume, 24h Change%, Liquidity
- Sparkline mini-charts in each row
- Color-coded price changes (green up, red down)
- Click to expand and see full price chart
- Filterable by category (Politics, Sports, Crypto, etc.)

#### 3b. Volume Heatmap
- Grid: categories on Y-axis, time buckets on X-axis
- Color intensity = volume magnitude
- Helps identify which categories are hot at what times

#### 4. Momentum/Signal Scanner
- **Biggest Movers:** Markets with largest absolute price change in selected timeframe
- **Volume Spikes:** Markets where current volume > 2x average volume
- **New Markets:** Recently created markets with growing volume
- **Convergence Signals:** Markets where price is approaching 0 or 100 (near resolution)

### Computed Indicators (ClickHouse Queries)

```sql
-- Momentum: Rate of price change
SELECT
    condition_id,
    question,
    last_price,
    price_1h_ago,
    (last_price - price_1h_ago) / price_1h_ago * 100 AS momentum_1h,
    volume_24h,
    volume_24h / nullIf(volume_24h_prev, 0) AS volume_ratio
FROM market_signals
ORDER BY abs(momentum_1h) DESC
LIMIT 20;

-- Volume spike detection
SELECT
    condition_id,
    question,
    volume_1h,
    avg_volume_1h,
    volume_1h / avg_volume_1h AS spike_ratio
FROM market_signals
WHERE volume_1h / avg_volume_1h > 2.0
ORDER BY spike_ratio DESC;

-- Near-resolution markets
SELECT
    condition_id,
    question,
    last_price,
    volume_24h
FROM markets
WHERE last_price > 0.90 OR last_price < 0.10
ORDER BY volume_24h DESC;
```

---

## 8. Tailwind CSS + shadcn/ui Dashboard Layout

### Setup

```bash
npx create-next-app@latest polymarket-dashboard --typescript --tailwind --eslint --app --src-dir
cd polymarket-dashboard
npx shadcn@latest init
```

When prompted, select:
- Style: **New York**
- Base color: **Zinc** (dark-mode friendly neutral)
- CSS variables: **Yes**

### Required shadcn/ui Components

```bash
npx shadcn@latest add card table badge tabs select input button
npx shadcn@latest add chart        # Recharts-based charts
npx shadcn@latest add data-table   # TanStack Table wrapper (if available)
npx shadcn@latest add separator skeleton tooltip popover command
```

### Layout Structure

```typescript
// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Polymarket Analytics',
  description: 'Real-time Polymarket prediction market analytics',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-foreground`}>
        <div className="min-h-screen">
          <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center justify-between">
              <span className="font-bold text-lg">Polymarket Analytics</span>
              {/* Search + filters */}
            </div>
          </header>
          <main className="container py-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
```

### Dashboard Page with Grid Layout

```typescript
// app/dashboard/page.tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Suspense } from 'react';

export default async function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* KPI Cards Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Suspense fallback={<CardSkeleton />}>
          <KPICard title="24h Volume" value="$12.4M" change="+14.2%" />
        </Suspense>
        <Suspense fallback={<CardSkeleton />}>
          <KPICard title="Active Markets" value="1,247" change="+23" />
        </Suspense>
        <Suspense fallback={<CardSkeleton />}>
          <KPICard title="Open Interest" value="$89.2M" change="+2.1%" />
        </Suspense>
        <Suspense fallback={<CardSkeleton />}>
          <KPICard title="Big Movers" value="34" change="markets >10%" />
        </Suspense>
      </div>

      {/* Featured Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Market Price</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<div className="h-[400px] animate-pulse bg-muted rounded" />}>
            <FeaturedPriceChart />
          </Suspense>
        </CardContent>
      </Card>

      {/* Two-column: Table + Heatmap */}
      <div className="grid gap-4 lg:grid-cols-7">
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle>Trending Markets</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense fallback={<TableSkeleton />}>
              <TrendingMarketsTable />
            </Suspense>
          </CardContent>
        </Card>
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>Volume Heatmap</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense fallback={<div className="h-[300px] animate-pulse bg-muted rounded" />}>
              <VolumeHeatmap />
            </Suspense>
          </CardContent>
        </Card>
      </div>

      {/* Signals Scanner */}
      <Card>
        <CardHeader>
          <CardTitle>Market Signals</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="movers">
            <TabsList>
              <TabsTrigger value="movers">Biggest Movers</TabsTrigger>
              <TabsTrigger value="volume">Volume Spikes</TabsTrigger>
              <TabsTrigger value="new">New Markets</TabsTrigger>
              <TabsTrigger value="resolution">Near Resolution</TabsTrigger>
            </TabsList>
            <TabsContent value="movers"><MoversList /></TabsContent>
            <TabsContent value="volume"><VolumeSpikes /></TabsContent>
            <TabsContent value="new"><NewMarkets /></TabsContent>
            <TabsContent value="resolution"><NearResolution /></TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
```

### TanStack Table for Market Data

```bash
npm install @tanstack/react-table@^8.20.0
```

Use shadcn's `<DataTable>` pattern with TanStack Table for sorting, filtering, and pagination on the trending markets table.

---

## 9. Polymarket API Endpoints Reference

### API Architecture

Polymarket exposes three main API services:

| Service | Base URL | Auth Required | Purpose |
|---------|----------|---------------|---------|
| **CLOB API** | `https://clob.polymarket.com` | No (read) / Yes (trade) | Order books, prices, trades |
| **Gamma API** | `https://gamma-api.polymarket.com` | No | Market discovery, metadata, categories |
| **Data API** | `https://data-api.polymarket.com` | Yes | User positions, activity |
| **WebSocket** | `wss://ws-subscriptions-clob.polymarket.com/ws/` | No (market) | Real-time price/orderbook updates |

### Key Endpoints for Dashboard

#### Gamma API (Market Discovery)

```
GET /events
  ?active=true
  &limit=50
  &offset=0
  &order=volume24hr     # Sort by 24h volume
  &ascending=false

GET /events/{event_id}

GET /markets
  ?active=true
  &limit=100
  &offset=0
  &order=volume24hr
  &tag_id={tag_id}      # Filter by category

GET /markets/{condition_id}
```

**Response shape (event):**
```json
{
  "id": "...",
  "ticker": "...",
  "slug": "...",
  "title": "Will X happen?",
  "description": "...",
  "startDate": "2026-01-01T00:00:00Z",
  "endDate": "2026-03-01T00:00:00Z",
  "markets": [
    {
      "id": "...",
      "question": "Will X happen?",
      "conditionId": "0x...",
      "outcomes": ["Yes", "No"],
      "outcomePrices": ["0.65", "0.35"],
      "volume": 1234567.89,
      "liquidity": 98765.43,
      "active": true,
      "closed": false,
      "clobTokenIds": ["token_yes_id", "token_no_id"]
    }
  ]
}
```

#### CLOB API (Prices & Trading Data)

```
GET /prices-history
  ?market={clob_token_id}
  &interval=1d           # 1m, 1h, 6h, 1d, 1w, max
  &fidelity=60           # Resolution in minutes

GET /price?token_id={token_id}&side=BUY
GET /midpoint?token_id={token_id}
GET /spread?token_id={token_id}
GET /book?token_id={token_id}
GET /last-trade-price?token_id={token_id}
GET /tick-size?token_id={token_id}
```

**Price history response:**
```json
{
  "history": [
    { "t": 1697875200, "p": 0.65 },
    { "t": 1697878800, "p": 0.67 }
  ]
}
```

#### WebSocket (Real-Time Updates)

```typescript
const ws = new WebSocket('wss://ws-subscriptions-clob.polymarket.com/ws/market');

// Subscribe to a market's price updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market',
  assets_id: 'clob_token_id',
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // { asset_id, price, timestamp, ... }
};
```

### Rate Limits

- ~100 requests/minute for unauthenticated access (soft limit)
- Batch API calls where possible (get multiple orderbooks in one call)
- Use WebSocket for real-time instead of polling individual prices

---

## 10. Recommended Architecture Summary

### Package Versions (Pinned)

```json
{
  "dependencies": {
    "next": "^15.2.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "@clickhouse/client": "^1.17.0",
    "lightweight-charts": "^4.2.0",
    "recharts": "^2.15.0",
    "swr": "^2.3.0",
    "@tanstack/react-table": "^8.20.0",
    "tailwindcss": "^4.0.0",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.6.0",
    "lucide-react": "^0.469.0"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "@types/node": "^22.0.0",
    "@types/react": "^19.0.0",
    "eslint": "^9.0.0",
    "eslint-config-next": "^15.2.0"
  }
}
```

### Data Flow Architecture

```
[Polymarket APIs] ──(every 5 min)──> [Python Pipeline (Docker)]
                                              │
                                              ▼
                                     [ClickHouse Cloud]
                                              │
                                              ▼
                                     [Next.js Dashboard (Vercel)]
                                       ├── Server Components (SSR initial load)
                                       ├── Route Handlers (API for SWR polling)
                                       └── Client Components (SWR + Charts)
```

### File Structure

```
polymarket/
  dashboard/                        # Next.js app (deployed to Vercel)
    src/
      app/
        layout.tsx                  # Root layout with dark theme
        page.tsx                    # Redirect to /dashboard
        dashboard/
          page.tsx                  # Main dashboard (server component)
          loading.tsx               # Suspense fallback
          components/
            kpi-cards.tsx           # Overview stats
            price-chart.tsx         # TradingView Lightweight Charts
            trending-table.tsx      # TanStack Table + shadcn
            volume-heatmap.tsx      # Recharts treemap
            signal-scanner.tsx      # Momentum/volume signals
        api/
          markets/route.ts          # GET markets from ClickHouse
          prices/route.ts           # GET price history
          signals/route.ts          # GET computed signals
      lib/
        clickhouse.ts               # ClickHouse client singleton
        polymarket.ts               # Polymarket API helpers (for client)
        utils.ts                    # cn() helper, formatters
      components/
        ui/                         # shadcn/ui components
      types/
        market.ts                   # TypeScript interfaces
    next.config.ts
    tailwind.config.ts
    components.json                 # shadcn config

  pipeline/                         # Python data pipeline (Docker)
    Dockerfile
    requirements.txt
    main.py                         # Scheduler entry point
    fetcher.py                      # Polymarket API client
    transformer.py                  # Data transformation + signals
    loader.py                       # ClickHouse bulk loader
    schema/
      migrations.sql                # ClickHouse table DDL

  compose.yaml                      # Docker Compose for pipeline
  .env.example                      # Environment variable template
```

### ClickHouse Schema (Key Tables)

```sql
-- Events metadata
CREATE TABLE events (
    event_id String,
    slug String,
    title String,
    description String,
    category String,
    start_date DateTime,
    end_date Nullable(DateTime),
    active UInt8,
    updated_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY event_id;

-- Markets (outcomes within events)
CREATE TABLE markets (
    condition_id String,
    event_id String,
    question String,
    outcomes Array(String),
    outcome_prices Array(Float64),
    clob_token_ids Array(String),
    volume Float64,
    volume_24h Float64,
    liquidity Float64,
    active UInt8,
    closed UInt8,
    updated_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY condition_id;

-- Price snapshots (time series)
CREATE TABLE price_snapshots (
    token_id String,
    condition_id String,
    price Float64,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (token_id, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Materialized view: hourly aggregates
CREATE MATERIALIZED VIEW price_hourly_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (token_id, hour)
AS SELECT
    token_id,
    toStartOfHour(timestamp) AS hour,
    argMinState(price, timestamp) AS open,
    argMaxState(price, timestamp) AS close,
    minState(price) AS low,
    maxState(price) AS high,
    countState() AS num_snapshots
FROM price_snapshots
GROUP BY token_id, hour;

-- Pre-computed signals (updated by pipeline)
CREATE TABLE market_signals (
    condition_id String,
    question String,
    last_price Float64,
    price_1h_ago Float64,
    price_24h_ago Float64,
    momentum_1h Float64,
    momentum_24h Float64,
    volume_1h Float64,
    volume_24h Float64,
    avg_volume_1h Float64,
    volume_ratio Float64,
    updated_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY condition_id;
```

### Key Design Decisions

1. **Node.js runtime only** (not Edge) for all ClickHouse queries
2. **SWR polling** as primary real-time strategy (10-30s intervals)
3. **Server Components** for initial data load (SEO irrelevant, but fast TTFB)
4. **TradingView Lightweight Charts** for all price/financial charts
5. **Recharts** only for aggregate visualizations (volume bars, category breakdown)
6. **ReplacingMergeTree** for mutable data (markets, events) to handle upserts
7. **MergeTree with TTL** for time-series price data (auto-expire after 1 year)
8. **Materialized views** for hourly aggregates to speed up chart queries
9. **Docker Compose** for pipeline with resource limits and health checks
10. **shadcn/ui + Tailwind CSS 4** for consistent, dark-mode dashboard UI
