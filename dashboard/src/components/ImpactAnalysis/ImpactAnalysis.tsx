"use client";

import { useEffect, useRef, useCallback } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  AreaSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import { Badge } from "@/components/ui/badge";
import type { ImpactResult, PricePoint } from "@/types/causal";

// ── Props ─────────────────────────────────────────────────

interface ImpactAnalysisProps {
  /** Market identifier */
  marketId: string;
  /** Market question / label */
  marketLabel?: string;
  /** ISO timestamp of the event */
  eventTimestamp: string;
  /** Event description for display */
  eventLabel?: string;
  /** Full price time series (pre + post period) */
  priceData: PricePoint[];
  /** Causal impact analysis result */
  impactResult: ImpactResult;
}

// ── Helpers ───────────────────────────────────────────────

function toUnix(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}

function formatSignedPct(value: number): string {
  const pct = value * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function formatSignedPrice(value: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}\u00A2`;
}

// ── Component ─────────────────────────────────────────────

export function ImpactAnalysis({
  marketId,
  marketLabel,
  eventTimestamp,
  eventLabel,
  priceData,
  impactResult,
}: ImpactAnalysisProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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

  useEffect(() => {
    const chart = initChart();
    if (!chart) return;

    const eventUnix = toUnix(eventTimestamp);

    // Find the post-period start index in priceData
    const postStartIdx = priceData.findIndex((p) => toUnix(p.timestamp) >= eventUnix);
    const hasPostData = postStartIdx >= 0 && impactResult.actualPost.length > 0;

    // 1. Actual price line (full series)
    const actualSeries = chart.addSeries(LineSeries, {
      color: "#00d4aa",
      lineWidth: 2,
      title: "Actual",
    });
    actualSeries.setData(
      priceData.map((p) => ({
        time: toUnix(p.timestamp) as Time,
        value: p.price,
      })),
    );

    // 2. Counterfactual line (post-period only)
    if (hasPostData) {
      const counterfactualSeries = chart.addSeries(LineSeries, {
        color: "#6366f1",
        lineWidth: 2,
        lineStyle: 2, // dashed
        title: "Counterfactual",
      });

      const postPricePoints = priceData.slice(postStartIdx);
      const cfData = impactResult.counterfactualPost.map((price, i) => ({
        time: (i < postPricePoints.length
          ? toUnix(postPricePoints[i].timestamp)
          : eventUnix + (i + 1) * 3600) as Time,
        value: price,
      }));
      counterfactualSeries.setData(cfData);

      // 3. CI upper bound area
      if (impactResult.ciUpper !== 0 || impactResult.ciLower !== 0) {
        const ciUpperSeries = chart.addSeries(AreaSeries, {
          lineColor: "rgba(99, 102, 241, 0.3)",
          topColor: "rgba(99, 102, 241, 0.1)",
          bottomColor: "rgba(99, 102, 241, 0.0)",
          lineWidth: 1,
          title: "CI Upper",
        });

        const ciLowerSeries = chart.addSeries(AreaSeries, {
          lineColor: "rgba(99, 102, 241, 0.3)",
          topColor: "rgba(99, 102, 241, 0.0)",
          bottomColor: "rgba(99, 102, 241, 0.1)",
          lineWidth: 1,
          title: "CI Lower",
        });

        // Approximate CI as offset from counterfactual
        const ciOffset = (impactResult.ciUpper - impactResult.ciLower) / 2;
        const ciUpperData = cfData.map((p) => ({
          ...p,
          value: p.value + ciOffset,
        }));
        const ciLowerData = cfData.map((p) => ({
          ...p,
          value: p.value - ciOffset,
        }));

        ciUpperSeries.setData(ciUpperData);
        ciLowerSeries.setData(ciLowerData);
      }
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
  }, [initChart, priceData, impactResult, eventTimestamp]);

  const { significant, pValue, pointEffect, cumulativeEffect, relativeEffect, prePeriodR2 } =
    impactResult;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          {marketLabel && <h4 className="font-medium text-sm">{marketLabel}</h4>}
          {eventLabel && (
            <p className="text-xs text-muted-foreground mt-0.5">
              Event: {eventLabel} ({new Date(eventTimestamp).toLocaleDateString()})
            </p>
          )}
        </div>
        <Badge
          variant="secondary"
          className={`text-xs border-0 ${
            significant
              ? "bg-[#00d4aa]/10 text-[#00d4aa]"
              : "bg-[#6b7280]/10 text-[#6b7280]"
          }`}
        >
          {significant ? "Significant" : "Not Significant"}
        </Badge>
      </div>

      {/* Chart */}
      <div className="relative">
        <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
        {priceData.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            No price data available for impact analysis
          </div>
        )}
        {/* Legend overlay */}
        <div className="absolute top-2 left-2 flex items-center gap-4 text-xs bg-[#0d0d14]/80 px-3 py-1.5 rounded">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#00d4aa] inline-block rounded-full" />
            Actual
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#6366f1] inline-block rounded-full" style={{ borderBottom: "1px dashed #6366f1" }} />
            Counterfactual
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 bg-[#6366f1]/20 inline-block rounded" />
            95% CI
          </span>
        </div>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatBox
          label="Point Effect"
          value={formatSignedPrice(pointEffect)}
          color={pointEffect > 0 ? "#00d4aa" : pointEffect < 0 ? "#ff4466" : "#6b7280"}
        />
        <StatBox
          label="Cumulative"
          value={formatSignedPrice(cumulativeEffect)}
          color={cumulativeEffect > 0 ? "#00d4aa" : cumulativeEffect < 0 ? "#ff4466" : "#6b7280"}
        />
        <StatBox
          label="Relative Effect"
          value={formatSignedPct(relativeEffect)}
          color={relativeEffect > 0 ? "#00d4aa" : relativeEffect < 0 ? "#ff4466" : "#6b7280"}
        />
        <StatBox
          label="p-value"
          value={pValue < 0.001 ? "< 0.001" : pValue.toFixed(4)}
          color={pValue < 0.05 ? "#00d4aa" : "#6b7280"}
        />
        <StatBox
          label="Pre-period R\u00B2"
          value={prePeriodR2.toFixed(4)}
          color={prePeriodR2 > 0.7 ? "#00d4aa" : prePeriodR2 > 0.4 ? "#f59e0b" : "#ff4466"}
        />
        <StatBox
          label="Post Steps"
          value={String(impactResult.actualPost.length)}
          color="#6b7280"
        />
      </div>
    </div>
  );
}

// ── Stat box sub-component ────────────────────────────────

function StatBox({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-[#111118] border border-[#1e1e2e] rounded-lg px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-sm font-medium mt-0.5" style={{ color }}>
        {value}
      </p>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────

export function ImpactAnalysisSkeleton() {
  return (
    <div className="space-y-4">
      <div className="h-[350px] rounded-lg bg-[#1e1e2e] animate-pulse" />
      <div className="grid grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-14 rounded-lg bg-[#1e1e2e] animate-pulse" />
        ))}
      </div>
    </div>
  );
}
