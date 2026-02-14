"use client";

import { formatUSD, formatNumber } from "@/lib/format";
import type { CategoryBreakdown } from "@/types/market";

interface CategoryBreakdownProps {
  categories: CategoryBreakdown[];
}

export function CategoryBreakdownList({ categories }: CategoryBreakdownProps) {
  if (!categories || categories.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        No category data available
      </div>
    );
  }

  const maxVolume = Math.max(...categories.map((c) => c.total_volume));

  return (
    <div className="space-y-3">
      {categories.map((cat) => {
        const pct = maxVolume > 0 ? (cat.total_volume / maxVolume) * 100 : 0;
        return (
          <div key={cat.category} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="capitalize">{cat.category || "Other"}</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground font-mono">
                  {formatNumber(cat.market_count)} markets
                </span>
                <span className="font-mono text-xs">
                  {formatUSD(cat.total_volume)}
                </span>
              </div>
            </div>
            <div className="h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
