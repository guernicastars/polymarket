"use client";

import { Card, CardContent } from "@/components/ui/card";
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  BarChart3,
  Target,
  ArrowUp,
  ArrowDown,
  Minus,
} from "lucide-react";
import type { RiskOverview } from "@/types/bali-risk";
import { CATEGORY_CONFIG, TREND_CONFIG } from "@/types/bali-risk";

interface BaliRiskStatsCardsProps {
  overview: RiskOverview;
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "worsening") return <ArrowUp className="h-3 w-3" />;
  if (trend === "improving") return <ArrowDown className="h-3 w-3" />;
  return <Minus className="h-3 w-3" />;
}

function ScoreGauge({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return "bg-red-500";
    if (s >= 60) return "bg-orange-500";
    if (s >= 40) return "bg-amber-500";
    return "bg-emerald-500";
  };

  return (
    <div className="w-full h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${getColor(score)}`}
        style={{ width: `${score}%` }}
      />
    </div>
  );
}

export function BaliRiskStatsCards({ overview }: BaliRiskStatsCardsProps) {
  const trendCfg = TREND_CONFIG[overview.trend];
  const topCatCfg = CATEGORY_CONFIG[overview.top_risk_category];

  const cards = [
    {
      label: "Overall Risk Score",
      value: `${overview.overall_score}/100`,
      icon: Shield,
      color:
        overview.overall_score >= 70
          ? "text-red-400"
          : overview.overall_score >= 50
            ? "text-amber-400"
            : "text-emerald-400",
      extra: <ScoreGauge score={overview.overall_score} />,
    },
    {
      label: "Critical Risks",
      value: overview.critical_count.toString(),
      icon: AlertTriangle,
      color: "text-red-400",
      extra: (
        <span className="text-xs text-muted-foreground">
          + {overview.high_count} high severity
        </span>
      ),
    },
    {
      label: "Risk Trend",
      value: trendCfg.label,
      icon: TrendingUp,
      color: trendCfg.color,
      extra: (
        <div className={`flex items-center gap-1 text-xs ${trendCfg.color}`}>
          <TrendIcon trend={overview.trend} />
          <span>
            {overview.trend === "worsening"
              ? "Conditions deteriorating"
              : overview.trend === "improving"
                ? "Conditions improving"
                : "No significant change"}
          </span>
        </div>
      ),
    },
    {
      label: "Top Risk Category",
      value: topCatCfg.label,
      icon: Target,
      color: topCatCfg.color,
      extra: (
        <span className="text-xs text-muted-foreground truncate">
          {topCatCfg.description}
        </span>
      ),
    },
    {
      label: "Total Indicators",
      value: overview.total_indicators.toString(),
      icon: BarChart3,
      color: "text-blue-400",
      extra: (
        <span className="text-xs text-muted-foreground">
          {overview.critical_count}C / {overview.high_count}H /{" "}
          {overview.medium_count}M / {overview.low_count}L
        </span>
      ),
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {cards.map((card) => (
        <Card key={card.label} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-2 mb-1">
              <card.icon className={`h-4 w-4 ${card.color}`} />
              <span className="text-xs text-muted-foreground">
                {card.label}
              </span>
            </div>
            <p className="text-xl font-bold mb-1">{card.value}</p>
            {card.extra}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function BaliRiskStatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse mb-2" />
            <div className="h-6 w-16 bg-[#1e1e2e] rounded animate-pulse mb-1" />
            <div className="h-3 w-32 bg-[#1e1e2e] rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
