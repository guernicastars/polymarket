"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { LeaderboardTable } from "@/components/leaderboard-table";
import { WhaleActivityTable } from "@/components/whale-activity-table";
import { SmartMoneyTable } from "@/components/smart-money-table";
import { ConcentrationTable } from "@/components/concentration-table";
import type {
  TraderRanking,
  WhaleActivityFeed,
  SmartMoneyPosition,
  PositionConcentration,
} from "@/types/market";

interface WhalesTabsProps {
  leaderboard: TraderRanking[];
  activity: WhaleActivityFeed[];
  smartMoney: SmartMoneyPosition[];
  concentration: PositionConcentration[];
}

export function WhalesTabs({
  leaderboard,
  activity,
  smartMoney,
  concentration,
}: WhalesTabsProps) {
  return (
    <Tabs defaultValue="leaderboard">
      <TabsList>
        <TabsTrigger value="leaderboard">
          Leaderboard ({leaderboard.length})
        </TabsTrigger>
        <TabsTrigger value="activity">
          Activity ({activity.length})
        </TabsTrigger>
        <TabsTrigger value="smart-money">
          Smart Money ({smartMoney.length})
        </TabsTrigger>
        <TabsTrigger value="concentration">
          Concentration ({concentration.length})
        </TabsTrigger>
      </TabsList>
      <TabsContent value="leaderboard">
        <LeaderboardTable data={leaderboard} />
      </TabsContent>
      <TabsContent value="activity">
        <WhaleActivityTable data={activity} />
      </TabsContent>
      <TabsContent value="smart-money">
        <SmartMoneyTable data={smartMoney} />
      </TabsContent>
      <TabsContent value="concentration">
        <ConcentrationTable data={concentration} />
      </TabsContent>
    </Tabs>
  );
}
