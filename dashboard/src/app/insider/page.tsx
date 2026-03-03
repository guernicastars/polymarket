import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getInsiderOverview,
  getInsiderSuspects,
  getPreNewsTradeEvents,
  getCoordinatedGroups,
  getInsiderTradeSignals,
} from "@/lib/queries";
import {
  InsiderStatsCards,
  InsiderStatsCardsSkeleton,
} from "@/components/insider-stats-cards";
import { InsiderTabs } from "@/components/insider-tabs";

export const dynamic = "force-dynamic";

async function InsiderStatsSection() {
  const stats = await getInsiderOverview();
  return <InsiderStatsCards stats={stats} />;
}

async function InsiderContent() {
  const [suspects, preNews, groups, signals] = await Promise.all([
    getInsiderSuspects(50),
    getPreNewsTradeEvents(50),
    getCoordinatedGroups(30),
    getInsiderTradeSignals(50),
  ]);
  return (
    <InsiderTabs
      suspects={suspects}
      preNews={preNews}
      groups={groups}
      signals={signals}
    />
  );
}

export default function InsiderPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          Insider Trading Detection
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Suspicious trading pattern analysis with focus on pre-news timing,
          coordinated wallets, and anomalous profitability across Middle East
          and geopolitical markets
        </p>
      </div>

      <Suspense fallback={<InsiderStatsCardsSkeleton />}>
        <InsiderStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Insider Intelligence</CardTitle>
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
            <InsiderContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
