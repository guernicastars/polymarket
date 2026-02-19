"use client";

import { useState, useCallback, useEffect } from "react";
import useSWR from "swr";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Network,
  Activity,
  ArrowLeftRight,
  ShieldAlert,
  GitBranch,
  Gauge,
} from "lucide-react";

import { CausalGraph } from "@/components/CausalGraph/CausalGraph";
import { ImpactAnalysis } from "@/components/ImpactAnalysis/ImpactAnalysis";
import { InformationFlow } from "@/components/InformationFlow/InformationFlow";
import { ManipulationAlert } from "@/components/ManipulationAlert/ManipulationAlert";
import { CounterfactualView } from "@/components/CounterfactualView/CounterfactualView";

import type {
  CausalNode,
  CausalEdge,
  ImpactResult,
  PricePoint,
  ManipulationAlertData,
  SyntheticControlResult,
  CounterfactualEvent,
  CausalOverview,
  CausalMarketOption,
} from "@/types/causal";

// ── Mock API Layer ────────────────────────────────────────
// These interfaces and fetchers will be replaced with real API
// calls once the backend endpoints are wired up.

interface CausalGraphResponse {
  nodes: CausalNode[];
  edges: CausalEdge[];
}

interface ImpactAnalysisResponse {
  priceData: PricePoint[];
  impactResult: ImpactResult;
  eventTimestamp: string;
  eventLabel: string;
}

interface InformationFlowResponse {
  flowMatrix: number[][];
  marketNames: string[];
  marketIds: string[];
  sourceMarkets: number[];
  derivativeMarkets: number[];
  maxTe: number;
}

interface ManipulationResponse {
  alerts: ManipulationAlertData[];
}

interface CounterfactualResponse {
  events: CounterfactualEvent[];
  result: SyntheticControlResult | null;
}

// The API base path for causal endpoints.
// All endpoints are typed but currently return empty/mock data from
// the client side until backend routes are implemented.
const CAUSAL_API_BASE = "/api/causal";

async function fetchCausalGraph(
  _marketIds: string[],
): Promise<CausalGraphResponse> {
  // TODO: Replace with real API call
  // return fetch(`${CAUSAL_API_BASE}/graph`, {
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify({ marketIds }),
  // }).then((r) => r.json());
  return { nodes: [], edges: [] };
}

async function fetchImpactAnalysis(
  _marketId: string,
  _eventId?: string,
): Promise<ImpactAnalysisResponse> {
  // TODO: Replace with real API call
  return {
    priceData: [],
    impactResult: {
      pointEffect: 0,
      cumulativeEffect: 0,
      relativeEffect: 0,
      ciLower: 0,
      ciUpper: 0,
      pValue: 1,
      significant: false,
      prePeriodR2: 0,
      actualPost: [],
      counterfactualPost: [],
      impactSeries: [],
    },
    eventTimestamp: new Date().toISOString(),
    eventLabel: "",
  };
}

async function fetchInformationFlow(
  _marketIds: string[],
): Promise<InformationFlowResponse> {
  // TODO: Replace with real API call
  return {
    flowMatrix: [],
    marketNames: [],
    marketIds: [],
    sourceMarkets: [],
    derivativeMarkets: [],
    maxTe: 0,
  };
}

async function fetchManipulationAlerts(
  _marketIds: string[],
): Promise<ManipulationResponse> {
  // TODO: Replace with real API call
  return { alerts: [] };
}

async function fetchCounterfactual(
  _marketId: string,
  _eventId?: string,
): Promise<CounterfactualResponse> {
  // TODO: Replace with real API call
  return { events: [], result: null };
}

async function fetchCausalOverview(): Promise<CausalOverview> {
  // TODO: Replace with real API call
  return {
    totalEdges: 0,
    significantPairs: 0,
    sourceMarkets: 0,
    derivativeMarkets: 0,
    alertCount: 0,
    avgRiskScore: 0,
  };
}

async function fetchMarketOptions(): Promise<CausalMarketOption[]> {
  // Fetch top markets for the market selector
  try {
    const res = await fetch("/api/markets?limit=50");
    if (!res.ok) return [];
    const data = await res.json();
    return (data ?? []).map((m: Record<string, unknown>) => ({
      conditionId: m.condition_id as string,
      question: m.question as string,
      tokenId: Array.isArray(m.token_ids) ? (m.token_ids[0] as string) : undefined,
    }));
  } catch {
    return [];
  }
}

// ── Stats Cards ───────────────────────────────────────────

function CausalStatsCards({ stats }: { stats: CausalOverview }) {
  const cards = [
    {
      label: "Causal Edges",
      value: String(stats.totalEdges),
      icon: GitBranch,
      color: "text-[#00d4aa]",
    },
    {
      label: "Significant Pairs",
      value: String(stats.significantPairs),
      icon: Network,
      color: "text-[#6366f1]",
    },
    {
      label: "Source Markets",
      value: String(stats.sourceMarkets),
      icon: ArrowLeftRight,
      color: "text-amber-400",
    },
    {
      label: "Derivative Markets",
      value: String(stats.derivativeMarkets),
      icon: Activity,
      color: "text-blue-400",
    },
    {
      label: "Manipulation Alerts",
      value: String(stats.alertCount),
      icon: ShieldAlert,
      color: stats.alertCount > 0 ? "text-[#ff4466]" : "text-muted-foreground",
    },
    {
      label: "Avg Risk Score",
      value: `${(stats.avgRiskScore * 100).toFixed(0)}%`,
      icon: Gauge,
      color:
        stats.avgRiskScore > 0.5
          ? "text-[#ff4466]"
          : stats.avgRiskScore > 0.25
            ? "text-amber-400"
            : "text-[#00d4aa]",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {cards.map((card) => (
        <Card key={card.label} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-2 mb-1">
              <card.icon className={`h-4 w-4 ${card.color}`} />
              <span className="text-xs text-muted-foreground">{card.label}</span>
            </div>
            <p className="text-xl font-bold">{card.value}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function CausalStatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse mb-2" />
            <div className="h-6 w-16 bg-[#1e1e2e] rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ── Main Dashboard Component ──────────────────────────────

export function CausalDashboard() {
  const [selectedMarkets, setSelectedMarkets] = useState<string[]>([]);
  const [primaryMarket, setPrimaryMarket] = useState<string>("");

  // Fetch market options
  const { data: marketOptions } = useSWR<CausalMarketOption[]>(
    "causal-market-options",
    fetchMarketOptions,
    { revalidateOnFocus: false },
  );

  // Fetch overview stats
  const { data: overview } = useSWR<CausalOverview>(
    "causal-overview",
    fetchCausalOverview,
    { revalidateOnFocus: false },
  );

  // Fetch causal graph
  const { data: graphData } = useSWR<CausalGraphResponse>(
    selectedMarkets.length > 0 ? ["causal-graph", selectedMarkets] : null,
    () => fetchCausalGraph(selectedMarkets),
    { revalidateOnFocus: false },
  );

  // Fetch impact analysis
  const { data: impactData } = useSWR<ImpactAnalysisResponse>(
    primaryMarket ? ["causal-impact", primaryMarket] : null,
    () => fetchImpactAnalysis(primaryMarket),
    { revalidateOnFocus: false },
  );

  // Fetch information flow
  const { data: flowData } = useSWR<InformationFlowResponse>(
    selectedMarkets.length > 0 ? ["causal-flow", selectedMarkets] : null,
    () => fetchInformationFlow(selectedMarkets),
    { revalidateOnFocus: false },
  );

  // Fetch manipulation alerts
  const { data: manipulationData } = useSWR<ManipulationResponse>(
    selectedMarkets.length > 0 ? ["causal-manipulation", selectedMarkets] : null,
    () => fetchManipulationAlerts(selectedMarkets),
    { revalidateOnFocus: false },
  );

  // Fetch counterfactual
  const [cfEventId, setCfEventId] = useState<string>("");
  const { data: cfData, isLoading: cfLoading } = useSWR<CounterfactualResponse>(
    primaryMarket ? ["causal-cf", primaryMarket, cfEventId] : null,
    () => fetchCounterfactual(primaryMarket, cfEventId),
    { revalidateOnFocus: false },
  );

  // Set default primary market when options load
  useEffect(() => {
    if (marketOptions && marketOptions.length > 0 && !primaryMarket) {
      setPrimaryMarket(marketOptions[0].conditionId);
      setSelectedMarkets(marketOptions.slice(0, 10).map((m) => m.conditionId));
    }
  }, [marketOptions, primaryMarket]);

  const handlePrimaryMarketChange = useCallback(
    (conditionId: string) => {
      setPrimaryMarket(conditionId);
      // Ensure the primary market is in selected markets
      setSelectedMarkets((prev) =>
        prev.includes(conditionId) ? prev : [conditionId, ...prev].slice(0, 20),
      );
    },
    [],
  );

  const handleNodeClick = useCallback(
    (node: CausalNode) => {
      setPrimaryMarket(node.id);
    },
    [],
  );

  const primaryMarketLabel =
    marketOptions?.find((m) => m.conditionId === primaryMarket)?.question ??
    primaryMarket;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Causal Analysis
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Cross-market causal discovery, event impact analysis, information flow,
            manipulation detection, and counterfactual reasoning
          </p>
        </div>
        {/* Market selector */}
        {marketOptions && marketOptions.length > 0 && (
          <Select value={primaryMarket} onValueChange={handlePrimaryMarketChange}>
            <SelectTrigger className="w-[320px] bg-[#111118] border-[#1e1e2e]">
              <SelectValue placeholder="Select primary market" />
            </SelectTrigger>
            <SelectContent className="bg-[#111118] border-[#1e1e2e] max-h-[300px]">
              {marketOptions.map((m) => (
                <SelectItem key={m.conditionId} value={m.conditionId}>
                  <span className="truncate">{m.question}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {/* Stats cards */}
      {overview ? (
        <CausalStatsCards stats={overview} />
      ) : (
        <CausalStatsCardsSkeleton />
      )}

      {/* Tabbed content */}
      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardContent className="pt-4">
          <Tabs defaultValue="graph">
            <TabsList>
              <TabsTrigger value="graph">
                <Network className="h-3.5 w-3.5 mr-1.5" />
                Causal Graph
              </TabsTrigger>
              <TabsTrigger value="impact">
                <Activity className="h-3.5 w-3.5 mr-1.5" />
                Event Impact
              </TabsTrigger>
              <TabsTrigger value="flow">
                <ArrowLeftRight className="h-3.5 w-3.5 mr-1.5" />
                Information Flow
              </TabsTrigger>
              <TabsTrigger value="manipulation">
                <ShieldAlert className="h-3.5 w-3.5 mr-1.5" />
                Manipulation
                {manipulationData && manipulationData.alerts.length > 0 && (
                  <span className="ml-1 text-xs bg-[#ff4466]/20 text-[#ff4466] px-1.5 py-0.5 rounded-full">
                    {manipulationData.alerts.length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="counterfactual">
                <GitBranch className="h-3.5 w-3.5 mr-1.5" />
                What If
              </TabsTrigger>
            </TabsList>

            {/* Causal Graph Tab */}
            <TabsContent value="graph" className="mt-4">
              <CausalGraph
                nodes={graphData?.nodes ?? []}
                edges={graphData?.edges ?? []}
                onNodeClick={handleNodeClick}
              />
            </TabsContent>

            {/* Impact Analysis Tab */}
            <TabsContent value="impact" className="mt-4">
              {impactData ? (
                <ImpactAnalysis
                  marketId={primaryMarket}
                  marketLabel={primaryMarketLabel}
                  eventTimestamp={impactData.eventTimestamp}
                  eventLabel={impactData.eventLabel}
                  priceData={impactData.priceData}
                  impactResult={impactData.impactResult}
                />
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  {primaryMarket
                    ? "Loading impact analysis..."
                    : "Select a market to view event impact analysis"}
                </div>
              )}
            </TabsContent>

            {/* Information Flow Tab */}
            <TabsContent value="flow" className="mt-4">
              {flowData && flowData.flowMatrix.length > 0 ? (
                <InformationFlow
                  flowMatrix={flowData.flowMatrix}
                  marketNames={flowData.marketNames}
                  marketIds={flowData.marketIds}
                  sourceMarkets={flowData.sourceMarkets}
                  derivativeMarkets={flowData.derivativeMarkets}
                  maxTe={flowData.maxTe}
                />
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  {selectedMarkets.length > 0
                    ? "No information flow data available. Run transfer entropy analysis."
                    : "Select markets to analyze information flow"}
                </div>
              )}
            </TabsContent>

            {/* Manipulation Detection Tab */}
            <TabsContent value="manipulation" className="mt-4">
              <ManipulationAlert
                alerts={manipulationData?.alerts ?? []}
              />
            </TabsContent>

            {/* Counterfactual Tab */}
            <TabsContent value="counterfactual" className="mt-4">
              {primaryMarket ? (
                <CounterfactualView
                  marketId={primaryMarket}
                  marketLabel={primaryMarketLabel}
                  events={cfData?.events ?? []}
                  syntheticControlResult={cfData?.result ?? null}
                  onEventSelect={setCfEventId}
                  loading={cfLoading}
                />
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  Select a market to run counterfactual analysis
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
