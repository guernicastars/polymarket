"use client";

import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { TrendingUp } from "lucide-react";
import { formatUSD, formatPrice } from "@/lib/format";
import type { TrendingMarket } from "@/types/market";

interface TrendingMarketsProps {
  markets: TrendingMarket[];
}

export function TrendingMarkets({ markets }: TrendingMarketsProps) {
  const router = useRouter();

  if (!markets || markets.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        No trending markets detected
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {markets.map((market) => (
        <div
          key={market.condition_id}
          className="flex items-center gap-3 p-3 rounded-lg hover:bg-[#1a1a2e] cursor-pointer transition-colors"
          onClick={() => router.push(`/market/${market.condition_id}`)}
        >
          <div className="flex-shrink-0">
            <div className="flex items-center justify-center h-8 w-8 rounded-full bg-amber-500/10">
              <TrendingUp className="h-4 w-4 text-amber-400" />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{market.question}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs font-mono text-muted-foreground">
                {formatPrice(market.current_price)}
              </span>
              <span className="text-xs text-muted-foreground">
                Vol: {formatUSD(market.volume_1h)}
              </span>
            </div>
          </div>
          <Badge
            variant="secondary"
            className="bg-amber-500/10 text-amber-400 border-0 font-mono text-xs whitespace-nowrap"
          >
            {market.volume_ratio.toFixed(1)}x
          </Badge>
        </div>
      ))}
    </div>
  );
}
