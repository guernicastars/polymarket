"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, Zap, Wallet, BarChart3 } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { WhalesOverview } from "@/types/market";

interface WhalesStatsCardsProps {
  stats: WhalesOverview;
}

export function WhalesStatsCards({ stats }: WhalesStatsCardsProps) {
  const cards = [
    {
      title: "Tracked Wallets",
      value: formatNumber(stats.tracked_wallets),
      icon: Users,
      color: "text-violet-400",
    },
    {
      title: "Whale Trades (24h)",
      value: formatNumber(stats.whale_trades_24h),
      icon: Zap,
      color: "text-amber-400",
    },
    {
      title: "Total Positions",
      value: formatNumber(stats.total_whale_positions),
      icon: Wallet,
      color: "text-emerald-400",
    },
    {
      title: "Markets Held",
      value: formatNumber(stats.unique_markets_held),
      icon: BarChart3,
      color: "text-blue-400",
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

export function WhalesStatsCardsSkeleton() {
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
