"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CompositeSignalsTable } from "./composite-signals-table";
import { ArbitrageTable } from "./arbitrage-table";
import { WalletClustersTable } from "./wallet-clusters-table";
import { InsiderAlertsTable } from "./insider-alerts-table";
import type {
  CompositeSignal,
  ArbitrageOpportunity,
  WalletCluster,
  InsiderAlert,
} from "@/types/market";

interface AnalyticsTabsProps {
  composite: CompositeSignal[];
  arbitrage: ArbitrageOpportunity[];
  clusters: WalletCluster[];
  insider: InsiderAlert[];
}

export function AnalyticsTabs({
  composite,
  arbitrage,
  clusters,
  insider,
}: AnalyticsTabsProps) {
  return (
    <Tabs defaultValue="composite">
      <TabsList>
        <TabsTrigger value="composite">
          Composite Scores ({composite.length})
        </TabsTrigger>
        <TabsTrigger value="arbitrage">
          Arbitrage ({arbitrage.length})
        </TabsTrigger>
        <TabsTrigger value="clusters">
          Clusters ({clusters.length})
        </TabsTrigger>
        <TabsTrigger value="insider">
          Insider ({insider.length})
        </TabsTrigger>
      </TabsList>
      <TabsContent value="composite">
        <CompositeSignalsTable data={composite} />
      </TabsContent>
      <TabsContent value="arbitrage">
        <ArbitrageTable data={arbitrage} />
      </TabsContent>
      <TabsContent value="clusters">
        <WalletClustersTable data={clusters} />
      </TabsContent>
      <TabsContent value="insider">
        <InsiderAlertsTable data={insider} />
      </TabsContent>
    </Tabs>
  );
}
