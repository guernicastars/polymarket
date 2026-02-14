"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  createChart,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type Time,
} from "lightweight-charts";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import type { OHLCVBar } from "@/types/market";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

interface PriceChartProps {
  conditionId: string;
  initialData?: OHLCVBar[];
}

type Period = "1h" | "6h" | "1d" | "1w" | "1m";

const PERIOD_CONFIG: Record<Period, { interval: string; label: string }> = {
  "1h": { interval: "1m", label: "1H" },
  "6h": { interval: "1m", label: "6H" },
  "1d": { interval: "1m", label: "1D" },
  "1w": { interval: "1h", label: "1W" },
  "1m": { interval: "1h", label: "1M" },
};

function barsToCandles(bars: OHLCVBar[]): CandlestickData<Time>[] {
  return bars.map((bar) => ({
    time: (new Date(bar.bar_time).getTime() / 1000) as Time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  }));
}

function barsToVolume(bars: OHLCVBar[]): HistogramData<Time>[] {
  return bars.map((bar) => ({
    time: (new Date(bar.bar_time).getTime() / 1000) as Time,
    value: bar.volume,
    color: bar.close >= bar.open ? "rgba(0, 212, 170, 0.3)" : "rgba(255, 68, 102, 0.3)",
  }));
}

export function PriceChart({ conditionId, initialData }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleSeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volumeSeriesRef = useRef<any>(null);
  const [period, setPeriod] = useState<Period>("1d");

  const interval = PERIOD_CONFIG[period].interval;
  const apiUrl = `/api/prices/${conditionId}?interval=${interval}&outcome=Yes`;

  const { data: bars } = useSWR<OHLCVBar[]>(apiUrl, fetcher, {
    fallbackData: initialData,
    revalidateOnFocus: false,
  });

  const initChart = useCallback(() => {
    if (!containerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111118" },
        textColor: "#6b7280",
        fontFamily: "monospace",
      },
      grid: {
        vertLines: { color: "#1e1e2e" },
        horzLines: { color: "#1e1e2e" },
      },
      width: containerRef.current.clientWidth,
      height: 400,
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

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00d4aa",
      downColor: "#ff4466",
      borderUpColor: "#00d4aa",
      borderDownColor: "#ff4466",
      wickUpColor: "#00d4aa",
      wickDownColor: "#ff4466",
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    return chart;
  }, []);

  useEffect(() => {
    const chart = initChart();

    const handleResize = () => {
      if (containerRef.current && chart) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart?.remove();
      chartRef.current = null;
    };
  }, [initChart]);

  useEffect(() => {
    if (!bars || bars.length === 0) return;
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;

    const candles = barsToCandles(bars);
    const volume = barsToVolume(bars);

    candleSeriesRef.current.setData(candles);
    volumeSeriesRef.current.setData(volume);
    chartRef.current?.timeScale().fitContent();
  }, [bars]);

  return (
    <div>
      <div className="flex gap-1 mb-4">
        {(Object.keys(PERIOD_CONFIG) as Period[]).map((p) => (
          <Button
            key={p}
            variant={period === p ? "default" : "ghost"}
            size="sm"
            className={
              period === p
                ? "bg-[#1e1e2e] text-white font-mono text-xs"
                : "text-muted-foreground font-mono text-xs hover:bg-[#1e1e2e]"
            }
            onClick={() => setPeriod(p)}
          >
            {PERIOD_CONFIG[p].label}
          </Button>
        ))}
      </div>
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
      {(!bars || bars.length === 0) && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
          No price data available
        </div>
      )}
    </div>
  );
}

export function PriceChartSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex gap-1">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-8 w-10 rounded bg-[#1e1e2e] animate-pulse" />
        ))}
      </div>
      <div className="h-[400px] rounded-lg bg-[#1e1e2e] animate-pulse" />
    </div>
  );
}
