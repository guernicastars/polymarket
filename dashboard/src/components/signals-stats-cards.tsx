"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BookOpen, TrendingUp, Zap, Activity } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { SignalsOverview } from "@/types/market";

interface SignalsStatsCardsProps {
  stats: SignalsOverview;
}

export function SignalsStatsCards({ stats }: SignalsStatsCardsProps) {
  const cards = [
    {
      title: "OBI Signals",
      value: formatNumber(stats.obi_signals),
      icon: BookOpen,
      color: "text-violet-400",
    },
    {
      title: "Volume Anomalies",
      value: formatNumber(stats.volume_active_markets),
      icon: TrendingUp,
      color: "text-amber-400",
    },
    {
      title: "Large Trades (24h)",
      value: formatNumber(stats.large_trades_24h),
      icon: Zap,
      color: "text-red-400",
    },
    {
      title: "Active Markets",
      value: formatNumber(stats.active_markets),
      icon: Activity,
      color: "text-emerald-400",
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

export function SignalsStatsCardsSkeleton() {
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
