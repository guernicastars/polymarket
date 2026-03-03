"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ShieldAlert, AlertTriangle, Newspaper, Network, Gauge } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { InsiderOverview } from "@/types/market";

interface InsiderStatsCardsProps {
  stats: InsiderOverview;
}

export function InsiderStatsCards({ stats }: InsiderStatsCardsProps) {
  const cards = [
    { label: "Total Suspects", value: formatNumber(stats.total_suspects), icon: ShieldAlert, color: "text-amber-400" },
    { label: "Critical Alerts", value: formatNumber(stats.critical_alerts), icon: AlertTriangle, color: "text-red-400" },
    { label: "Pre-News Events", value: formatNumber(stats.pre_news_events), icon: Newspaper, color: "text-violet-400" },
    { label: "Coordinated Groups", value: formatNumber(stats.coordinated_groups), icon: Network, color: "text-blue-400" },
    { label: "Avg Suspicion", value: stats.avg_suspicion_score.toFixed(1), icon: Gauge, color: "text-emerald-400" },
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

export function InsiderStatsCardsSkeleton() {
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
