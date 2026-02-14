"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { InsiderAlert } from "@/types/market";

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

function scoreBarColor(score: number): string {
  if (score > 70) return "bg-[#ff4466]";
  if (score > 50) return "bg-amber-400";
  return "bg-yellow-400";
}

function factorBarColor(value: number): string {
  if (value > 60) return "bg-[#ff4466]/60";
  if (value > 30) return "bg-amber-400/60";
  return "bg-[#1e1e2e]";
}

interface InsiderAlertsTableProps {
  data: InsiderAlert[];
}

export function InsiderAlertsTable({ data }: InsiderAlertsTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No insider alerts detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Wallet</TableHead>
            <TableHead className="text-muted-foreground">Score</TableHead>
            <TableHead className="text-muted-foreground">Freshness</TableHead>
            <TableHead className="text-muted-foreground">Win Rate</TableHead>
            <TableHead className="text-muted-foreground">Niche</TableHead>
            <TableHead className="text-muted-foreground">Size</TableHead>
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
                <div className="flex items-center gap-2 min-w-[100px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${scoreBarColor(row.score)}`}
                      style={{ width: `${row.score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-8 text-right text-muted-foreground">
                    {row.score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.freshness_score)}`}
                      style={{ width: `${row.freshness_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.freshness_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.win_rate_score)}`}
                      style={{ width: `${row.win_rate_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.win_rate_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.niche_score)}`}
                      style={{ width: `${row.niche_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.niche_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-1 min-w-[60px]">
                  <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${factorBarColor(row.size_score)}`}
                      style={{ width: `${row.size_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-6 text-right text-muted-foreground">
                    {row.size_score.toFixed(0)}
                  </span>
                </div>
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
