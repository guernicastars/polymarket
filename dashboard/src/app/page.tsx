import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getTopMarkets,
  getTopMovers,
  getTrendingMarkets,
  getCategoryBreakdown,
  getOverviewStats,
} from "@/lib/queries";
import { StatsCards, StatsCardsSkeleton } from "@/components/stats-cards";
import { MarketTable, MarketTableSkeleton } from "@/components/market-table";
import { TopMovers } from "@/components/top-movers";
import { TrendingMarkets } from "@/components/trending-markets";
import { CategoryBreakdownList } from "@/components/category-breakdown";

export const dynamic = "force-dynamic";

async function StatsSection() {
  const stats = await getOverviewStats();
  return <StatsCards stats={stats} />;
}

async function MarketsSection() {
  const markets = await getTopMarkets(50);
  return <MarketTable initialData={markets} />;
}

async function MoversSection() {
  const movers = await getTopMovers(20);
  return <TopMovers movers={movers} />;
}

async function TrendingSection() {
  const trending = await getTrendingMarkets(10);
  return <TrendingMarkets markets={trending} />;
}

async function CategorySection() {
  const categories = await getCategoryBreakdown();
  return <CategoryBreakdownList categories={categories} />;
}

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <Suspense fallback={<StatsCardsSkeleton />}>
        <StatsSection />
      </Suspense>

      {/* Main content grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Markets Table - takes 2 cols */}
        <Card
          id="markets"
          className="lg:col-span-2 bg-[#111118] border-[#1e1e2e]"
        >
          <CardHeader>
            <CardTitle className="text-base">Top Markets</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense fallback={<MarketTableSkeleton />}>
              <MarketsSection />
            </Suspense>
          </CardContent>
        </Card>

        {/* Top Movers sidebar */}
        <Card id="movers" className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">Top Movers (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense
              fallback={
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-12 bg-[#1e1e2e] rounded animate-pulse"
                    />
                  ))}
                </div>
              }
            >
              <MoversSection />
            </Suspense>
          </CardContent>
        </Card>
      </div>

      {/* Trending + Categories row */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card id="trending" className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">
              Trending (Volume Spikes)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense
              fallback={
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-14 bg-[#1e1e2e] rounded animate-pulse"
                    />
                  ))}
                </div>
              }
            >
              <TrendingSection />
            </Suspense>
          </CardContent>
        </Card>

        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">Categories</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense
              fallback={
                <div className="space-y-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <div className="h-4 bg-[#1e1e2e] rounded animate-pulse" />
                      <div className="h-1.5 bg-[#1e1e2e] rounded animate-pulse" />
                    </div>
                  ))}
                </div>
              }
            >
              <CategorySection />
            </Suspense>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
