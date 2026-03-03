"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import type { InsiderSuspect } from "@/types/market";

function formatRelativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffSec = Math.floor((now - then) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

function truncateWallet(wallet: string): string {
  if (wallet.length <= 10) return wallet;
  return `${wallet.slice(0, 6)}...${wallet.slice(-4)}`;
}

function tierBadgeColor(tier: string): string {
  switch (tier) {
    case "critical":
      return "bg-red-500/15 text-red-400";
    case "high":
      return "bg-orange-500/15 text-orange-400";
    case "medium":
      return "bg-yellow-500/15 text-yellow-400";
    default:
      return "bg-emerald-500/15 text-emerald-400";
  }
}

function scoreBarColor(score: number): string {
  if (score > 70) return "bg-red-400";
  if (score > 50) return "bg-orange-400";
  if (score > 30) return "bg-yellow-400";
  return "bg-emerald-400";
}

function factorBarColor(value: number): string {
  if (value > 60) return "bg-red-400/60";
  if (value > 30) return "bg-amber-400/60";
  return "bg-[#1e1e2e]";
}

interface SuspectTableProps {
  data: InsiderSuspect[];
}

export function SuspectTable({ data }: SuspectTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No insider suspects detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Trader</TableHead>
            <TableHead className="text-muted-foreground">Tier</TableHead>
            <TableHead className="text-muted-foreground">Score</TableHead>
            <TableHead className="text-muted-foreground">Pre-News</TableHead>
            <TableHead className="text-muted-foreground">Statistical</TableHead>
            <TableHead className="text-muted-foreground">Profitability</TableHead>
            <TableHead className="text-muted-foreground">Coordination</TableHead>
            <TableHead className="text-right text-muted-foreground">Win Rate</TableHead>
            <TableHead className="text-right text-muted-foreground">ME %</TableHead>
            <TableHead className="text-right text-muted-foreground">Flagged</TableHead>
            <TableHead className="text-muted-foreground">Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.proxy_wallet}
              className="border-[#1e1e2e] hover:bg-[#1a1a2e] transition-colors"
            >
              <TableCell>
                <div className="flex items-center gap-2">
                  {row.profile_image ? (
                    <img
                      src={row.profile_image}
                      alt=""
                      className="h-6 w-6 rounded-full bg-[#1e1e2e]"
                    />
                  ) : (
                    <div className="h-6 w-6 rounded-full bg-[#1e1e2e]" />
                  )}
                  <span className="font-medium truncate max-w-[180px]">
                    {row.pseudonym || truncateWallet(row.proxy_wallet)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <Badge
                  variant="secondary"
                  className={`text-xs border-0 capitalize ${tierBadgeColor(row.suspicion_tier)}`}
                >
                  {row.suspicion_tier}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[100px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${scoreBarColor(row.suspicion_score)}`}
                      style={{ width: `${row.suspicion_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-8 text-right text-muted-foreground">
                    {row.suspicion_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.pre_news_score)}`}
                      style={{ width: `${row.pre_news_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.pre_news_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.statistical_score)}`}
                      style={{ width: `${row.statistical_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.statistical_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.profitability_score)}`}
                      style={{ width: `${row.profitability_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.profitability_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.coordination_score)}`}
                      style={{ width: `${row.coordination_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.coordination_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.win_rate * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                <span className={row.mideast_trade_pct > 0.5 ? "text-amber-400" : "text-muted-foreground"}>
                  {(row.mideast_trade_pct * 100).toFixed(0)}%
                </span>
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                <span className={row.flagged_trade_count > 5 ? "text-red-400" : "text-muted-foreground"}>
                  {row.flagged_trade_count}
                </span>
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                {formatRelativeTime(row.computed_at)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
