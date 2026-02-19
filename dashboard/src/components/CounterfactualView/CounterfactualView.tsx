"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  AreaSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import type { SyntheticControlResult, CounterfactualEvent } from "@/types/causal";

// ── Helpers ───────────────────────────────────────────────

function toUnix(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}

// ── Props ─────────────────────────────────────────────────

interface CounterfactualViewProps {
  /** Market ID */
  marketId: string;
  /** Market question / label */
  marketLabel?: string;
  /** Available events to select from */
  events: CounterfactualEvent[];
  /** Current synthetic control result (for selected event) */
  syntheticControlResult: SyntheticControlResult | null;
  /** Callback when user selects a different event */
  onEventSelect?: (eventId: string) => void;
  /** Loading state */
  loading?: boolean;
}

// ── Component ─────────────────────────────────────────────

export function CounterfactualView({
  marketId,
  marketLabel,
  events,
  syntheticControlResult,
  onEventSelect,
  loading = false,
}: CounterfactualViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string>(
    events[0]?.id ?? "",
  );

  const handleEventChange = useCallback(
    (eventId: string) => {
      setSelectedEventId(eventId);
      onEventSelect?.(eventId);
    },
    [onEventSelect],
  );

  const selectedEvent = events.find((e) => e.id === selectedEventId);

  const initChart = useCallback(() => {
    if (!containerRef.current) return;
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0d0d14" },
        textColor: "#6b7280",
        fontFamily: "monospace",
      },
      grid: {
        vertLines: { color: "#1e1e2e" },
        horzLines: { color: "#1e1e2e" },
      },
      width: containerRef.current.clientWidth,
      height: 350,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: "#1e1e2e",
      },
      rightPriceScale: {
        borderColor: "#1e1e2e",
      },
      crosshair: {
        vertLine: { color: "#374151", labelBackgroundColor: "#1e1e2e" },
        horzLine: { color: "#374151", labelBackgroundColor: "#1e1e2e" },
      },
    });

    chartRef.current = chart;
    return chart;
  }, []);

  // Render chart when result changes
  useEffect(() => {
    const chart = initChart();
    if (!chart || !syntheticControlResult) return;

    const { actualPost, counterfactualPost, timestampsPost, impactSeries } =
      syntheticControlResult;

    if (actualPost.length === 0 || timestampsPost.length === 0) {
      return;
    }

    // Actual post-event trajectory
    const actualSeries = chart.addSeries(LineSeries, {
      color: "#00d4aa",
      lineWidth: 2,
      title: "Actual",
    });
    actualSeries.setData(
      actualPost.map((price, i) => ({
        time: (i < timestampsPost.length
          ? toUnix(timestampsPost[i])
          : toUnix(timestampsPost[0]) + (i + 1) * 3600) as Time,
        value: price,
      })),
    );

    // Synthetic control counterfactual
    const counterfactualSeries = chart.addSeries(LineSeries, {
      color: "#6366f1",
      lineWidth: 2,
      lineStyle: 2,
      title: "Synthetic Control",
    });
    counterfactualSeries.setData(
      counterfactualPost.map((price, i) => ({
        time: (i < timestampsPost.length
          ? toUnix(timestampsPost[i])
          : toUnix(timestampsPost[0]) + (i + 1) * 3600) as Time,
        value: price,
      })),
    );

    // Impact area between actual and counterfactual
    if (impactSeries.length > 0) {
      const impactAreaSeries = chart.addSeries(AreaSeries, {
        lineColor: "transparent",
        topColor: "rgba(0, 212, 170, 0.15)",
        bottomColor: "rgba(255, 68, 102, 0.15)",
        lineWidth: 1,
        title: "Impact",
        priceScaleId: "impact",
      });
      chart.priceScale("impact").applyOptions({
        scaleMargins: { top: 0.05, bottom: 0.05 },
      });
      impactAreaSeries.setData(
        impactSeries.map((impact, i) => ({
          time: (i < timestampsPost.length
            ? toUnix(timestampsPost[i])
            : toUnix(timestampsPost[0]) + (i + 1) * 3600) as Time,
          value: impact,
        })),
      );
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current && chart) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [initChart, syntheticControlResult]);

  return (
    <div className="space-y-4">
      {/* Header with event selector */}
      <div className="flex items-start justify-between gap-4">
        <div>
          {marketLabel && <h4 className="font-medium text-sm">{marketLabel}</h4>}
          <p className="text-xs text-muted-foreground mt-0.5">
            Synthetic control counterfactual: what would have happened without the event?
          </p>
        </div>
        {events.length > 0 && (
          <Select value={selectedEventId} onValueChange={handleEventChange}>
            <SelectTrigger className="w-[260px] bg-[#111118] border-[#1e1e2e]">
              <SelectValue placeholder="Select an event" />
            </SelectTrigger>
            <SelectContent className="bg-[#111118] border-[#1e1e2e]">
              {events.map((event) => (
                <SelectItem key={event.id} value={event.id}>
                  <span className="truncate">{event.label}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {/* Selected event info */}
      {selectedEvent && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Event: {selectedEvent.label}</span>
          <span className="font-mono">
            ({new Date(selectedEvent.timestamp).toLocaleDateString()})
          </span>
          {selectedEvent.description && (
            <span className="text-muted-foreground/60">
              \u2014 {selectedEvent.description}
            </span>
          )}
        </div>
      )}

      {/* Chart */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#0d0d14]/80 rounded-lg">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <div className="h-4 w-4 border-2 border-muted-foreground/30 border-t-muted-foreground rounded-full animate-spin" />
              Computing synthetic control...
            </div>
          </div>
        )}
        <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
        {(!syntheticControlResult || syntheticControlResult.actualPost.length === 0) &&
          !loading && (
            <div className="flex items-center justify-center h-[350px] text-muted-foreground rounded-lg bg-[#0d0d14] border border-[#1e1e2e]">
              {events.length === 0
                ? "No events available for counterfactual analysis"
                : "Select an event to view counterfactual trajectory"}
            </div>
          )}
        {/* Legend overlay */}
        {syntheticControlResult && syntheticControlResult.actualPost.length > 0 && (
          <div className="absolute top-2 left-2 flex items-center gap-4 text-xs bg-[#0d0d14]/80 px-3 py-1.5 rounded">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[#00d4aa] inline-block rounded-full" />
              Actual
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[#6366f1] inline-block rounded-full" />
              Counterfactual
            </span>
          </div>
        )}
      </div>

      {/* Stats and donor weights */}
      {syntheticControlResult && syntheticControlResult.actualPost.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Metrics */}
          <Card className="bg-[#111118] border-[#1e1e2e]">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Impact Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                <MetricRow
                  label="Average Effect"
                  value={`${syntheticControlResult.averageEffect >= 0 ? "+" : ""}${(syntheticControlResult.averageEffect * 100).toFixed(2)}\u00A2`}
                  color={
                    syntheticControlResult.averageEffect > 0
                      ? "#00d4aa"
                      : syntheticControlResult.averageEffect < 0
                        ? "#ff4466"
                        : "#6b7280"
                  }
                />
                <MetricRow
                  label="Cumulative Effect"
                  value={`${syntheticControlResult.cumulativeEffect >= 0 ? "+" : ""}${(syntheticControlResult.cumulativeEffect * 100).toFixed(2)}\u00A2`}
                  color={
                    syntheticControlResult.cumulativeEffect > 0
                      ? "#00d4aa"
                      : syntheticControlResult.cumulativeEffect < 0
                        ? "#ff4466"
                        : "#6b7280"
                  }
                />
                <MetricRow
                  label="Pre-fit RMSE"
                  value={syntheticControlResult.preFitRmse.toFixed(6)}
                  color={
                    syntheticControlResult.preFitRmse < 0.01
                      ? "#00d4aa"
                      : syntheticControlResult.preFitRmse < 0.05
                        ? "#f59e0b"
                        : "#ff4466"
                  }
                />
                <MetricRow
                  label="Post Steps"
                  value={String(syntheticControlResult.actualPost.length)}
                  color="#6b7280"
                />
              </div>
            </CardContent>
          </Card>

          {/* Donor weights */}
          <Card className="bg-[#111118] border-[#1e1e2e]">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Donor Market Weights
              </CardTitle>
            </CardHeader>
            <CardContent>
              {Object.keys(syntheticControlResult.weights).length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No donor markets with significant weight
                </p>
              ) : (
                <div className="space-y-2">
                  {Object.entries(syntheticControlResult.weights)
                    .sort((a, b) => b[1] - a[1])
                    .map(([tokenId, weight]) => (
                      <div
                        key={tokenId}
                        className="flex items-center gap-2"
                      >
                        <span className="text-xs truncate max-w-[180px] text-muted-foreground font-mono">
                          {tokenId.length > 12
                            ? `${tokenId.slice(0, 6)}...${tokenId.slice(-4)}`
                            : tokenId}
                        </span>
                        <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#6366f1]"
                            style={{ width: `${weight * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono w-12 text-right text-muted-foreground">
                          {(weight * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

// ── Metric row sub-component ──────────────────────────────

function MetricRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-sm font-medium" style={{ color }}>
        {value}
      </p>
    </div>
  );
}

// ── Skeleton ──────────────────────────────────────────────

export function CounterfactualViewSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="h-4 w-48 bg-[#1e1e2e] rounded animate-pulse" />
          <div className="h-3 w-64 bg-[#1e1e2e] rounded animate-pulse" />
        </div>
        <div className="h-9 w-[260px] bg-[#1e1e2e] rounded animate-pulse" />
      </div>
      <div className="h-[350px] rounded-lg bg-[#1e1e2e] animate-pulse" />
      <div className="grid grid-cols-2 gap-4">
        <div className="h-32 rounded-lg bg-[#1e1e2e] animate-pulse" />
        <div className="h-32 rounded-lg bg-[#1e1e2e] animate-pulse" />
      </div>
    </div>
  );
}
