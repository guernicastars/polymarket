"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown, BarChart3, Activity, Zap } from "lucide-react";
import { formatUSD, formatNumber } from "@/lib/format";
import type { OverviewStats } from "@/types/market";

interface StatsCardsProps {
  stats: OverviewStats;
}

export function StatsCards({ stats }: StatsCardsProps) {
  const cards = [
    {
      title: "24h Volume",
      value: formatUSD(stats.total_volume_24h),
      icon: BarChart3,
      color: "text-blue-400",
    },
    {
      title: "Active Markets",
      value: formatNumber(stats.active_markets),
      icon: Activity,
      color: "text-emerald-400",
    },
    {
      title: "Total Markets",
      value: formatNumber(stats.total_markets),
      icon: TrendingUp,
      color: "text-violet-400",
    },
    {
      title: "Trending",
      value: formatNumber(stats.trending_count),
      icon: Zap,
      color: "text-amber-400",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <Card key={card.title} className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {card.title}
            </CardTitle>
            <card.icon className={`h-4 w-4 ${card.color}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono">{card.value}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function StatsCardsSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="h-4 w-24 bg-muted rounded animate-pulse" />
          </CardHeader>
          <CardContent>
            <div className="h-8 w-32 bg-muted rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
