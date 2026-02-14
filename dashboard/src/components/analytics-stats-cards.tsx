"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeftRight, Network, ShieldAlert, Target, Gauge } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { AnalyticsOverview } from "@/types/market";

interface AnalyticsStatsCardsProps {
  stats: AnalyticsOverview;
}

export function AnalyticsStatsCards({ stats }: AnalyticsStatsCardsProps) {
  const cards = [
    { label: "Open Arbitrages", value: formatNumber(stats.open_arbitrages), icon: ArrowLeftRight, color: "text-amber-400" },
    { label: "Wallet Clusters", value: formatNumber(stats.wallet_clusters), icon: Network, color: "text-violet-400" },
    { label: "Insider Alerts", value: formatNumber(stats.insider_alerts), icon: ShieldAlert, color: "text-red-400" },
    { label: "Markets Scored", value: formatNumber(stats.markets_scored), icon: Target, color: "text-emerald-400" },
    { label: "Avg Confidence", value: `${(stats.avg_confidence * 100).toFixed(0)}%`, icon: Gauge, color: "text-blue-400" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
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

export function AnalyticsStatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {Array.from({ length: 5 }).map((_, i) => (
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
