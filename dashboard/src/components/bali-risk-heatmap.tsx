"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { AnyRisk, RiskCategory } from "@/types/bali-risk";
import { CATEGORY_CONFIG } from "@/types/bali-risk";

interface BaliRiskHeatmapProps {
  risksByCategory: Record<RiskCategory, AnyRisk[]>;
}

function getAvgScore(risks: AnyRisk[]): number {
  if (risks.length === 0) return 0;
  return Math.round(risks.reduce((a, r) => a + r.score, 0) / risks.length);
}

function getScoreColor(score: number): string {
  if (score >= 80) return "bg-red-500/30 border-red-500/50";
  if (score >= 65) return "bg-orange-500/25 border-orange-500/40";
  if (score >= 50) return "bg-amber-500/20 border-amber-500/35";
  if (score >= 35) return "bg-yellow-500/15 border-yellow-500/30";
  return "bg-emerald-500/15 border-emerald-500/30";
}

function getScoreTextColor(score: number): string {
  if (score >= 80) return "text-red-400";
  if (score >= 65) return "text-orange-400";
  if (score >= 50) return "text-amber-400";
  if (score >= 35) return "text-yellow-400";
  return "text-emerald-400";
}

const categories: RiskCategory[] = [
  "legal",
  "climate",
  "political",
  "financial",
  "regulatory",
  "market",
];

export function BaliRiskHeatmap({ risksByCategory }: BaliRiskHeatmapProps) {
  return (
    <Card className="bg-[#111118] border-[#1e1e2e]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Risk Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {categories.map((cat) => {
            const risks = risksByCategory[cat] || [];
            const avg = getAvgScore(risks);
            const cfg = CATEGORY_CONFIG[cat];
            const critCount = risks.filter(
              (r) => r.severity === "critical"
            ).length;
            const highCount = risks.filter(
              (r) => r.severity === "high"
            ).length;
            const worseningCount = risks.filter(
              (r) => r.trend === "worsening"
            ).length;

            return (
              <div
                key={cat}
                className={`rounded-lg border p-4 ${getScoreColor(avg)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${cfg.color}`}>
                    {cfg.label}
                  </span>
                  <span
                    className={`text-2xl font-bold ${getScoreTextColor(avg)}`}
                  >
                    {avg}
                  </span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span>{risks.length} indicators</span>
                  {critCount > 0 && (
                    <span className="text-red-400">{critCount} critical</span>
                  )}
                  {highCount > 0 && (
                    <span className="text-orange-400">{highCount} high</span>
                  )}
                  {worseningCount > 0 && (
                    <span className="text-red-400/70">
                      {worseningCount} worsening
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4 pt-3 border-t border-[#1e1e2e]">
          <span className="text-xs text-muted-foreground">Score:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-emerald-500/30" />
            <span className="text-xs text-muted-foreground">0-34</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-yellow-500/30" />
            <span className="text-xs text-muted-foreground">35-49</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-amber-500/30" />
            <span className="text-xs text-muted-foreground">50-64</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-orange-500/30" />
            <span className="text-xs text-muted-foreground">65-79</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500/30" />
            <span className="text-xs text-muted-foreground">80-100</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
