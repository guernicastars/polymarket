"use client";

import { ArrowUp, ArrowDown } from "lucide-react";
import { formatPrice, formatPct } from "@/lib/format";
import type { TopMover } from "@/types/market";
import { useRouter } from "next/navigation";

interface TopMoversProps {
  movers: TopMover[];
}

export function TopMovers({ movers }: TopMoversProps) {
  const router = useRouter();

  if (!movers || movers.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        No price movement data available
      </div>
    );
  }

  const gainers = movers
    .filter((m) => m.pct_change > 0)
    .slice(0, 5);
  const losers = movers
    .filter((m) => m.pct_change < 0)
    .sort((a, b) => a.pct_change - b.pct_change)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Top Gainers
        </h4>
        <div className="space-y-2">
          {gainers.map((m) => (
            <div
              key={`${m.condition_id}-${m.outcome}`}
              className="flex items-center justify-between p-2 rounded-lg hover:bg-[#1a1a2e] cursor-pointer transition-colors"
              onClick={() => router.push(`/market/${m.condition_id}`)}
            >
              <div className="flex-1 min-w-0 mr-3">
                <p className="text-sm truncate">{m.question}</p>
                <p className="text-xs text-muted-foreground font-mono">
                  {formatPrice(m.current_price)}
                </p>
              </div>
              <div className="flex items-center gap-1 text-[#00d4aa] font-mono text-sm whitespace-nowrap">
                <ArrowUp className="h-3 w-3" />
                {formatPct(m.pct_change)}
              </div>
            </div>
          ))}
          {gainers.length === 0 && (
            <p className="text-xs text-muted-foreground">No gainers</p>
          )}
        </div>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Top Losers
        </h4>
        <div className="space-y-2">
          {losers.map((m) => (
            <div
              key={`${m.condition_id}-${m.outcome}`}
              className="flex items-center justify-between p-2 rounded-lg hover:bg-[#1a1a2e] cursor-pointer transition-colors"
              onClick={() => router.push(`/market/${m.condition_id}`)}
            >
              <div className="flex-1 min-w-0 mr-3">
                <p className="text-sm truncate">{m.question}</p>
                <p className="text-xs text-muted-foreground font-mono">
                  {formatPrice(m.current_price)}
                </p>
              </div>
              <div className="flex items-center gap-1 text-[#ff4466] font-mono text-sm whitespace-nowrap">
                <ArrowDown className="h-3 w-3" />
                {formatPct(m.pct_change)}
              </div>
            </div>
          ))}
          {losers.length === 0 && (
            <p className="text-xs text-muted-foreground">No losers</p>
          )}
        </div>
      </div>
    </div>
  );
}
