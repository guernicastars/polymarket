import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getWhalesOverview,
  getLeaderboard,
  getWhaleActivity,
  getSmartMoneyPositions,
  getPositionConcentration,
} from "@/lib/queries";
import {
  WhalesStatsCards,
  WhalesStatsCardsSkeleton,
} from "@/components/whales-stats-cards";
import { WhalesTabs } from "@/components/whales-tabs";

export const dynamic = "force-dynamic";

async function WhalesStatsSection() {
  const stats = await getWhalesOverview();
  return <WhalesStatsCards stats={stats} />;
}

async function WhalesContent() {
  const [leaderboard, activity, smartMoney, concentration] = await Promise.all([
    getLeaderboard("OVERALL", "ALL", "PNL", 50),
    getWhaleActivity(50),
    getSmartMoneyPositions(50),
    getPositionConcentration(30),
  ]);
  return (
    <WhalesTabs
      leaderboard={leaderboard}
      activity={activity}
      smartMoney={smartMoney}
      concentration={concentration}
    />
  );
}

export default function WhalesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Whales</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Top trader rankings, whale activity, and smart money positions
        </p>
      </div>

      <Suspense fallback={<WhalesStatsCardsSkeleton />}>
        <WhalesStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Whale Intelligence</CardTitle>
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
            <WhalesContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
