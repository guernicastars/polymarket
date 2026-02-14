import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getSignalsOverview,
  getOrderBookImbalance,
  getVolumeAnomalies,
  getLargeTrades,
  getCompositeSignals,
} from "@/lib/queries";
import {
  SignalsStatsCards,
  SignalsStatsCardsSkeleton,
} from "@/components/signals-stats-cards";
import { SignalsTabs } from "@/components/signals-tabs";

export const dynamic = "force-dynamic";

async function SignalsStatsSection() {
  const stats = await getSignalsOverview();
  return <SignalsStatsCards stats={stats} />;
}

async function SignalsContent() {
  const [obi, volume, largeTrades, compositeSignals] = await Promise.all([
    getOrderBookImbalance(50),
    getVolumeAnomalies(30),
    getLargeTrades(1000, 50),
    getCompositeSignals(50),
  ]);
  return (
    <SignalsTabs
      obi={obi}
      volumeAnomalies={volume}
      largeTrades={largeTrades}
      compositeSignals={compositeSignals}
    />
  );
}

export default function SignalsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Signals</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Real-time trading signals from order book, volume, and trade data
        </p>
      </div>

      <Suspense fallback={<SignalsStatsCardsSkeleton />}>
        <SignalsStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Active Signals</CardTitle>
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
            <SignalsContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
