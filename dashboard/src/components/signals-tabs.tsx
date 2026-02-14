"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { OBISignalsTable } from "@/components/obi-signals-table";
import { VolumeAnomaliesTable } from "@/components/volume-anomalies-table";
import { LargeTradesTable } from "@/components/large-trades-table";
import { CompositeSignalsTable } from "@/components/composite-signals-table";
import type { OBISignal, VolumeAnomaly, LargeTrade, CompositeSignal } from "@/types/market";

interface SignalsTabsProps {
  obi: OBISignal[];
  volumeAnomalies: VolumeAnomaly[];
  largeTrades: LargeTrade[];
  compositeSignals?: CompositeSignal[];
}

export function SignalsTabs({ obi, volumeAnomalies, largeTrades, compositeSignals }: SignalsTabsProps) {
  return (
    <Tabs defaultValue="composite">
      <TabsList>
        {compositeSignals && (
          <TabsTrigger value="composite">
            Composite ({compositeSignals.length})
          </TabsTrigger>
        )}
        <TabsTrigger value="obi">Order Book ({obi.length})</TabsTrigger>
        <TabsTrigger value="volume">Volume Spikes ({volumeAnomalies.length})</TabsTrigger>
        <TabsTrigger value="large">Large Trades ({largeTrades.length})</TabsTrigger>
      </TabsList>
      {compositeSignals && (
        <TabsContent value="composite">
          <CompositeSignalsTable data={compositeSignals} />
        </TabsContent>
      )}
      <TabsContent value="obi">
        <OBISignalsTable data={obi} />
      </TabsContent>
      <TabsContent value="volume">
        <VolumeAnomaliesTable data={volumeAnomalies} />
      </TabsContent>
      <TabsContent value="large">
        <LargeTradesTable data={largeTrades} />
      </TabsContent>
    </Tabs>
  );
}
