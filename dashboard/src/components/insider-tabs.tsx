"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SuspectTable } from "./suspect-table";
import { PreNewsTable } from "./pre-news-table";
import { CoordinatedGroupsTable } from "./coordinated-groups-table";
import { InsiderTradeSignalsTable } from "./insider-trade-signals-table";
import type {
  InsiderSuspect,
  PreNewsEvent,
  CoordinatedGroup,
  InsiderTradeSignal,
} from "@/types/market";

interface InsiderTabsProps {
  suspects: InsiderSuspect[];
  preNews: PreNewsEvent[];
  groups: CoordinatedGroup[];
  signals: InsiderTradeSignal[];
}

export function InsiderTabs({
  suspects,
  preNews,
  groups,
  signals,
}: InsiderTabsProps) {
  return (
    <Tabs defaultValue="suspects">
      <TabsList>
        <TabsTrigger value="suspects">
          Suspects ({suspects.length})
        </TabsTrigger>
        <TabsTrigger value="prenews">
          Pre-News ({preNews.length})
        </TabsTrigger>
        <TabsTrigger value="coordinated">
          Coordinated ({groups.length})
        </TabsTrigger>
        <TabsTrigger value="signals">
          Trade Signals ({signals.length})
        </TabsTrigger>
      </TabsList>
      <TabsContent value="suspects">
        <SuspectTable data={suspects} />
      </TabsContent>
      <TabsContent value="prenews">
        <PreNewsTable data={preNews} />
      </TabsContent>
      <TabsContent value="coordinated">
        <CoordinatedGroupsTable data={groups} />
      </TabsContent>
      <TabsContent value="signals">
        <InsiderTradeSignalsTable data={signals} />
      </TabsContent>
    </Tabs>
  );
}
