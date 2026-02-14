import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getAnalyticsOverview,
  getCompositeSignals,
  getArbitrageOpportunities,
  getWalletClusters,
  getInsiderAlerts,
} from "@/lib/queries";
import {
  AnalyticsStatsCards,
  AnalyticsStatsCardsSkeleton,
} from "@/components/analytics-stats-cards";
import { AnalyticsTabs } from "@/components/analytics-tabs";

export const dynamic = "force-dynamic";

async function AnalyticsStatsSection() {
  const stats = await getAnalyticsOverview();
  return <AnalyticsStatsCards stats={stats} />;
}

async function AnalyticsContent() {
  const [composite, arbitrage, clusters, insider] = await Promise.all([
    getCompositeSignals(50),
    getArbitrageOpportunities(50),
    getWalletClusters(30),
    getInsiderAlerts(30, 50),
  ]);
  return (
    <AnalyticsTabs
      composite={composite}
      arbitrage={arbitrage}
      clusters={clusters}
      insider={insider}
    />
  );
}

export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Advanced analytics: arbitrage detection, wallet clustering, insider
          scoring, and composite signals
        </p>
      </div>

      <Suspense fallback={<AnalyticsStatsCardsSkeleton />}>
        <AnalyticsStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Advanced Signals</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense
            fallback={
              <div className="space-y-3">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-12 bg-[#1e1e2e] rounded animate-pulse"
                  />
                ))}
              </div>
            }
          >
            <AnalyticsContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
