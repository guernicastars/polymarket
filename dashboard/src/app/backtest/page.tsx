"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// ── Types ──────────────────────────────────────────────────
interface CalibrationBin {
  bin_start: number;
  bin_end: number;
  predicted_mean: number;
  observed_freq: number;
  count: number;
}

interface Prediction {
  settlement: string;
  name: string;
  deadline: string;
  model_prob: number;
  market_prob: number;
  current_market_prob: number;
  resolved: boolean;
  outcome: number | null;
  edge: number;
  direction: string;
  kelly_fraction: number;
  pnl: number;
}

interface BacktestReport {
  generated_at: string;
  total_predictions: number;
  resolved_predictions: number;
  pending_predictions: number;
  brier_score: number;
  log_loss: number;
  market_brier_score: number;
  market_log_loss: number;
  calibration_bins: CalibrationBin[];
  total_signals: number;
  correct_direction: number;
  direction_accuracy: number;
  total_invested: number;
  total_return: number;
  roi_pct: number;
  best_trade: { name: string; pnl: number; direction: string };
  worst_trade: { name: string; pnl: number; direction: string };
  predictions: Prediction[];
}

// ── Stat card ──────────────────────────────────────────────
function StatCard({
  title,
  value,
  subtitle,
  trend,
}: {
  title: string;
  value: string;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
}) {
  const trendColor =
    trend === "up"
      ? "text-emerald-400"
      : trend === "down"
        ? "text-red-400"
        : "text-zinc-400";
  return (
    <Card className="bg-[#12121a] border-[#1e1e2e]">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`text-2xl font-bold ${trendColor}`}>{value}</div>
        {subtitle && (
          <p className="text-xs text-zinc-500 mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Calibration chart (SVG) ────────────────────────────────
function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  const w = 400,
    h = 300,
    pad = 50;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-lg">
      {/* Background */}
      <rect width={w} height={h} fill="#12121a" rx={8} />

      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
        <g key={v}>
          <line
            x1={pad}
            y1={pad + plotH * (1 - v)}
            x2={pad + plotW}
            y2={pad + plotH * (1 - v)}
            stroke="#1e1e2e"
            strokeWidth={1}
          />
          <text
            x={pad - 8}
            y={pad + plotH * (1 - v) + 4}
            fill="#666"
            fontSize={10}
            textAnchor="end"
          >
            {(v * 100).toFixed(0)}%
          </text>
          <text
            x={pad + plotW * v}
            y={h - pad + 16}
            fill="#666"
            fontSize={10}
            textAnchor="middle"
          >
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* Perfect calibration line */}
      <line
        x1={pad}
        y1={pad + plotH}
        x2={pad + plotW}
        y2={pad}
        stroke="#333"
        strokeWidth={1.5}
        strokeDasharray="6,4"
      />
      <text x={pad + plotW - 60} y={pad + 16} fill="#555" fontSize={9}>
        Perfect
      </text>

      {/* Model predictions (bars) */}
      {bins.map((bin, i) => {
        const cx = pad + plotW * bin.predicted_mean;
        const cy = pad + plotH * (1 - bin.observed_freq);
        const barWidth = plotW / bins.length * 0.6;
        const barX = pad + (plotW / bins.length) * i + (plotW / bins.length * 0.2);
        const predH = plotH * bin.predicted_mean;
        const obsH = plotH * bin.observed_freq;

        return (
          <g key={i}>
            {/* Predicted bar */}
            <rect
              x={barX}
              y={pad + plotH - predH}
              width={barWidth / 2}
              height={predH}
              fill="#00d4aa"
              opacity={0.3}
              rx={2}
            />
            {/* Observed bar */}
            <rect
              x={barX + barWidth / 2}
              y={pad + plotH - obsH}
              width={barWidth / 2}
              height={Math.max(obsH, 1)}
              fill="#f59e0b"
              opacity={0.6}
              rx={2}
            />
            {/* Count label */}
            {bin.count > 0 && (
              <text
                x={barX + barWidth / 2}
                y={pad + plotH - Math.max(predH, obsH) - 6}
                fill="#888"
                fontSize={9}
                textAnchor="middle"
              >
                n={bin.count}
              </text>
            )}
            {/* Dot on calibration scatter */}
            {bin.count > 0 && (
              <circle
                cx={cx}
                cy={cy}
                r={4 + bin.count}
                fill="#ff6b6b"
                opacity={0.8}
              />
            )}
          </g>
        );
      })}

      {/* Axes labels */}
      <text
        x={w / 2}
        y={h - 8}
        fill="#888"
        fontSize={11}
        textAnchor="middle"
      >
        Predicted Probability
      </text>
      <text
        x={14}
        y={h / 2}
        fill="#888"
        fontSize={11}
        textAnchor="middle"
        transform={`rotate(-90 14 ${h / 2})`}
      >
        Observed Frequency
      </text>

      {/* Legend */}
      <rect x={pad + 8} y={pad + 8} width={10} height={10} fill="#00d4aa" opacity={0.4} rx={2} />
      <text x={pad + 22} y={pad + 17} fill="#888" fontSize={9}>Predicted</text>
      <rect x={pad + 80} y={pad + 8} width={10} height={10} fill="#f59e0b" opacity={0.6} rx={2} />
      <text x={pad + 94} y={pad + 17} fill="#888" fontSize={9}>Observed</text>
      <circle cx={pad + 158} cy={pad + 13} r={4} fill="#ff6b6b" opacity={0.8} />
      <text x={pad + 166} y={pad + 17} fill="#888" fontSize={9}>Calibration dot</text>
    </svg>
  );
}

// ── Model vs Market comparison chart ───────────────────────
function ComparisonChart({ predictions }: { predictions: Prediction[] }) {
  const sorted = [...predictions].sort(
    (a, b) => Math.abs(b.edge) - Math.abs(a.edge)
  );
  const top = sorted.slice(0, 12);
  const w = 600,
    h = 320,
    pad = { top: 30, right: 20, bottom: 80, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const barW = plotW / top.length;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect width={w} height={h} fill="#12121a" rx={8} />

      {/* Y-axis grid */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
        <g key={v}>
          <line
            x1={pad.left}
            y1={pad.top + plotH * (1 - v)}
            x2={pad.left + plotW}
            y2={pad.top + plotH * (1 - v)}
            stroke="#1e1e2e"
          />
          <text
            x={pad.left - 6}
            y={pad.top + plotH * (1 - v) + 4}
            fill="#666"
            fontSize={10}
            textAnchor="end"
          >
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* Bars */}
      {top.map((p, i) => {
        const x = pad.left + i * barW;
        const modelH = plotH * p.model_prob;
        const marketH = plotH * p.market_prob;
        const bw = barW * 0.35;

        return (
          <g key={`${p.name}-${p.deadline}`}>
            {/* Model bar */}
            <rect
              x={x + barW * 0.1}
              y={pad.top + plotH - modelH}
              width={bw}
              height={modelH}
              fill="#00d4aa"
              opacity={0.7}
              rx={2}
            />
            {/* Market bar */}
            <rect
              x={x + barW * 0.1 + bw + 2}
              y={pad.top + plotH - marketH}
              width={bw}
              height={marketH}
              fill="#6366f1"
              opacity={0.7}
              rx={2}
            />
            {/* Label */}
            <text
              x={x + barW / 2}
              y={h - pad.bottom + 14}
              fill="#888"
              fontSize={8}
              textAnchor="middle"
              transform={`rotate(-45 ${x + barW / 2} ${h - pad.bottom + 14})`}
            >
              {p.name.length > 12 ? p.name.slice(0, 11) + "…" : p.name}
            </text>
            {/* Status indicator */}
            <circle
              cx={x + barW / 2}
              cy={pad.top + plotH + 4}
              r={3}
              fill={
                p.outcome === 0
                  ? "#22c55e"
                  : p.outcome === 1
                    ? "#ef4444"
                    : "#f59e0b"
              }
            />
          </g>
        );
      })}

      {/* Legend */}
      <rect x={pad.left + 8} y={8} width={10} height={10} fill="#00d4aa" opacity={0.7} rx={2} />
      <text x={pad.left + 22} y={17} fill="#888" fontSize={10}>Model</text>
      <rect x={pad.left + 72} y={8} width={10} height={10} fill="#6366f1" opacity={0.7} rx={2} />
      <text x={pad.left + 86} y={17} fill="#888" fontSize={10}>Market</text>
      <circle cx={pad.left + 146} cy={13} r={3} fill="#22c55e" />
      <text x={pad.left + 154} y={17} fill="#888" fontSize={10}>Resolved NO</text>
      <circle cx={pad.left + 236} cy={13} r={3} fill="#f59e0b" />
      <text x={pad.left + 244} y={17} fill="#888" fontSize={10}>Pending</text>
    </svg>
  );
}

// ── Predictions table ──────────────────────────────────────
function PredictionsTable({ predictions }: { predictions: Prediction[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#1e1e2e] text-zinc-400 text-xs uppercase tracking-wider">
            <th className="text-left py-3 px-3">Settlement</th>
            <th className="text-left py-3 px-2">Deadline</th>
            <th className="text-right py-3 px-2">Model</th>
            <th className="text-right py-3 px-2">Market</th>
            <th className="text-right py-3 px-2">Edge</th>
            <th className="text-center py-3 px-2">Signal</th>
            <th className="text-right py-3 px-2">Kelly</th>
            <th className="text-right py-3 px-2">P&L</th>
            <th className="text-center py-3 px-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, i) => {
            const edgeColor =
              p.edge > 0.05
                ? "text-emerald-400"
                : p.edge < -0.05
                  ? "text-red-400"
                  : "text-zinc-500";
            const dirBadge =
              p.direction === "BUY"
                ? "bg-emerald-500/20 text-emerald-400"
                : p.direction === "SELL"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-zinc-500/20 text-zinc-400";
            const status =
              p.outcome === 0
                ? "✅ NO"
                : p.outcome === 1
                  ? "❌ YES"
                  : "⏳";

            return (
              <tr
                key={`${p.name}-${p.deadline}-${i}`}
                className="border-b border-[#1e1e2e]/50 hover:bg-[#1e1e2e]/30"
              >
                <td className="py-2.5 px-3 font-medium">{p.name}</td>
                <td className="py-2.5 px-2 text-zinc-400 text-xs font-mono">
                  {p.deadline}
                </td>
                <td className="py-2.5 px-2 text-right font-mono">
                  {(p.model_prob * 100).toFixed(0)}%
                </td>
                <td className="py-2.5 px-2 text-right font-mono text-zinc-400">
                  {(p.market_prob * 100).toFixed(0)}%
                </td>
                <td className={`py-2.5 px-2 text-right font-mono ${edgeColor}`}>
                  {p.edge > 0 ? "+" : ""}
                  {(p.edge * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-2 text-center">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${dirBadge}`}
                  >
                    {p.direction}
                  </span>
                </td>
                <td className="py-2.5 px-2 text-right font-mono text-zinc-400">
                  {p.kelly_fraction > 0
                    ? `${(p.kelly_fraction * 100).toFixed(1)}%`
                    : "—"}
                </td>
                <td className="py-2.5 px-2 text-right font-mono">
                  {p.outcome !== null && p.direction !== "HOLD" ? (
                    <span
                      className={
                        p.pnl > 0 ? "text-emerald-400" : "text-red-400"
                      }
                    >
                      {p.pnl > 0 ? "+" : ""}
                      {p.pnl.toFixed(4)}
                    </span>
                  ) : (
                    <span className="text-zinc-600">—</span>
                  )}
                </td>
                <td className="py-2.5 px-2 text-center">{status}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Edge distribution mini-chart ───────────────────────────
function EdgeDistribution({ predictions }: { predictions: Prediction[] }) {
  const edges = predictions
    .filter((p) => p.direction !== "HOLD")
    .sort((a, b) => b.edge - a.edge);
  const w = 400,
    h = 160,
    pad = 30;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;
  const maxEdge = Math.max(...edges.map((e) => Math.abs(e.edge)), 0.5);
  const barW = plotW / edges.length;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-md">
      <rect width={w} height={h} fill="#12121a" rx={8} />
      {/* Zero line */}
      <line
        x1={pad}
        y1={pad + plotH / 2}
        x2={pad + plotW}
        y2={pad + plotH / 2}
        stroke="#333"
        strokeWidth={1}
      />
      <text x={pad - 4} y={pad + plotH / 2 + 4} fill="#666" fontSize={9} textAnchor="end">
        0%
      </text>

      {edges.map((p, i) => {
        const barH = (p.edge / maxEdge) * (plotH / 2);
        const x = pad + i * barW;
        const color = p.edge > 0 ? "#00d4aa" : "#ef4444";

        return (
          <g key={`${p.name}-${p.deadline}`}>
            <rect
              x={x + 1}
              y={barH > 0 ? pad + plotH / 2 - barH : pad + plotH / 2}
              width={Math.max(barW - 2, 2)}
              height={Math.abs(barH)}
              fill={color}
              opacity={0.7}
              rx={1}
            />
            <title>
              {p.name}: {(p.edge * 100).toFixed(1)}% edge ({p.direction})
            </title>
          </g>
        );
      })}

      <text x={w / 2} y={h - 4} fill="#888" fontSize={9} textAnchor="middle">
        Sorted by edge (green=BUY, red=SELL)
      </text>
    </svg>
  );
}

// ── Main page ──────────────────────────────────────────────
export default function BacktestPage() {
  const [report, setReport] = useState<BacktestReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/backtest_report.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Model Backtest</h1>
        <Card className="bg-[#12121a] border-[#1e1e2e]">
          <CardContent className="pt-6">
            <p className="text-red-400">
              Failed to load backtest report: {error}
            </p>
            <p className="text-zinc-500 text-sm mt-2">
              Run <code className="bg-[#1e1e2e] px-1 rounded">python -m network.backtest</code> to generate the report.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Model Backtest</h1>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="bg-[#12121a] border-[#1e1e2e] animate-pulse">
              <CardContent className="pt-6 h-20" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Backtest & Calibration</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Conflict vulnerability model vs Polymarket prices — {report.total_predictions} predictions across {report.resolved_predictions} resolved + {report.pending_predictions} active markets
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <StatCard
          title="Brier Score"
          value={report.brier_score.toFixed(4)}
          subtitle={`Market: ${report.market_brier_score.toFixed(4)}`}
          trend={report.brier_score < report.market_brier_score ? "up" : "down"}
        />
        <StatCard
          title="Log Loss"
          value={report.log_loss.toFixed(4)}
          subtitle={`Market: ${report.market_log_loss.toFixed(4)}`}
          trend={report.log_loss < report.market_log_loss ? "up" : "down"}
        />
        <StatCard
          title="Direction"
          value={`${report.correct_direction}/${report.total_signals}`}
          subtitle={`${(report.direction_accuracy * 100).toFixed(0)}% accuracy`}
          trend={report.direction_accuracy > 0.5 ? "up" : "down"}
        />
        <StatCard
          title="ROI"
          value={`${report.roi_pct > 0 ? "+" : ""}${report.roi_pct.toFixed(1)}%`}
          subtitle={`$${report.total_return.toFixed(4)} return`}
          trend={report.roi_pct > 0 ? "up" : "down"}
        />
        <StatCard
          title="Resolved"
          value={`${report.resolved_predictions}`}
          subtitle={`of ${report.total_predictions} total`}
          trend="neutral"
        />
        <StatCard
          title="Active Signals"
          value={`${report.total_signals}`}
          subtitle="BUY + SELL (excl HOLD)"
          trend="neutral"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Calibration chart */}
        <Card className="bg-[#12121a] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Calibration Curve
            </CardTitle>
            <p className="text-xs text-zinc-500">
              Predicted vs observed probability — dots on the diagonal = perfectly calibrated
            </p>
          </CardHeader>
          <CardContent>
            <CalibrationChart bins={report.calibration_bins} />
          </CardContent>
        </Card>

        {/* Model vs Market comparison */}
        <Card className="bg-[#12121a] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Model vs Market (Top 12 by edge)
            </CardTitle>
            <p className="text-xs text-zinc-500">
              Green = model prediction, purple = market price. Dots: 🟢 resolved NO, 🟡 pending
            </p>
          </CardHeader>
          <CardContent>
            <ComparisonChart predictions={report.predictions} />
          </CardContent>
        </Card>
      </div>

      {/* Edge distribution + P&L */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-[#12121a] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Edge Distribution
            </CardTitle>
            <p className="text-xs text-zinc-500">
              Model-market disagreement for active signals (green = BUY edge, red = SELL edge)
            </p>
          </CardHeader>
          <CardContent>
            <EdgeDistribution predictions={report.predictions} />
          </CardContent>
        </Card>

        <Card className="bg-[#12121a] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Hypothetical P&L Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-wider">Total Invested</p>
                <p className="text-lg font-mono font-medium">
                  ${report.total_invested.toFixed(4)}
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-wider">Total Return</p>
                <p className={`text-lg font-mono font-medium ${report.total_return > 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {report.total_return > 0 ? "+" : ""}${report.total_return.toFixed(4)}
                </p>
              </div>
            </div>
            {report.best_trade.name && (
              <div className="border-t border-[#1e1e2e] pt-3">
                <p className="text-xs text-zinc-500">Best Trade</p>
                <p className="text-sm">
                  <span className="text-emerald-400 font-medium">{report.best_trade.name}</span>{" "}
                  <span className="text-zinc-500">({report.best_trade.direction})</span>{" "}
                  <span className="text-emerald-400 font-mono">+${report.best_trade.pnl.toFixed(4)}</span>
                </p>
              </div>
            )}
            {report.worst_trade.name && (
              <div>
                <p className="text-xs text-zinc-500">Worst Trade</p>
                <p className="text-sm">
                  <span className="text-red-400 font-medium">{report.worst_trade.name}</span>{" "}
                  <span className="text-zinc-500">({report.worst_trade.direction})</span>{" "}
                  <span className="text-red-400 font-mono">${report.worst_trade.pnl.toFixed(4)}</span>
                </p>
              </div>
            )}
            <div className="border-t border-[#1e1e2e] pt-3">
              <p className="text-xs text-zinc-500 mb-2">Key Insight</p>
              <p className="text-xs text-zinc-400 leading-relaxed">
                {report.brier_score < report.market_brier_score
                  ? `Model Brier score (${report.brier_score.toFixed(4)}) beats market (${report.market_brier_score.toFixed(4)}) — the model's probability estimates are better calibrated than the crowd on resolved outcomes.`
                  : `Market Brier score (${report.market_brier_score.toFixed(4)}) beats model (${report.brier_score.toFixed(4)}) — the crowd's probability estimates are better calibrated than our model.`}
                {" "}
                {report.roi_pct < 0
                  ? `However, negative ROI (${report.roi_pct.toFixed(1)}%) on resolved trades suggests the model's directional signals need refinement. All resolved outcomes were NO (no city captured), which penalized BUY signals.`
                  : `Positive ROI (${report.roi_pct.toFixed(1)}%) on resolved trades validates the signal generation approach.`}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Full predictions table */}
      <Card className="bg-[#12121a] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-sm font-medium">
            All Predictions
          </CardTitle>
          <p className="text-xs text-zinc-500">
            {report.predictions.length} predictions — {report.resolved_predictions} resolved, {report.pending_predictions} pending
          </p>
        </CardHeader>
        <CardContent>
          <PredictionsTable predictions={report.predictions} />
        </CardContent>
      </Card>

      {/* Footer */}
      <p className="text-xs text-zinc-600 text-center">
        Report generated {new Date(report.generated_at).toLocaleString()} — Model: Vulnerability + Supply Chain + Cascade (40/30/20/10 blend)
      </p>
    </div>
  );
}
