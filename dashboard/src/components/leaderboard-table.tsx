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
import { formatUSD } from "@/lib/format";
import type { TraderRanking } from "@/types/market";

function truncateWallet(wallet: string): string {
  if (wallet.length <= 10) return wallet;
  return `${wallet.slice(0, 6)}...${wallet.slice(-4)}`;
}

interface LeaderboardTableProps {
  data: TraderRanking[];
}

export function LeaderboardTable({ data }: LeaderboardTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No leaderboard data available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground w-16">Rank</TableHead>
            <TableHead className="text-muted-foreground">Trader</TableHead>
            <TableHead className="text-right text-muted-foreground">PnL</TableHead>
            <TableHead className="text-right text-muted-foreground">Volume</TableHead>
            <TableHead className="text-muted-foreground">Verified</TableHead>
            <TableHead className="text-muted-foreground">X</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => {
            const pnlPositive = row.pnl >= 0;
            return (
              <TableRow
                key={`${row.proxy_wallet}-${row.category}-${row.time_period}`}
                className="border-[#1e1e2e] hover:bg-[#1a1a2e] transition-colors"
              >
                <TableCell className="font-mono text-muted-foreground">
                  #{row.rank}
                </TableCell>
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
                    <span className="font-medium truncate max-w-[200px]">
                      {row.user_name || truncateWallet(row.proxy_wallet)}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span className={pnlPositive ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                    {pnlPositive ? "+" : ""}{formatUSD(row.pnl)}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatUSD(row.volume)}
                </TableCell>
                <TableCell>
                  {row.verified_badge ? (
                    <Badge variant="secondary" className="text-xs bg-[#00d4aa]/10 text-[#00d4aa] border-0">
                      Verified
                    </Badge>
                  ) : null}
                </TableCell>
                <TableCell>
                  {row.x_username ? (
                    <a
                      href={`https://x.com/${row.x_username}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-400 hover:underline"
                      onClick={(e) => e.stopPropagation()}
                    >
                      @{row.x_username}
                    </a>
                  ) : (
                    <span className="text-xs text-muted-foreground">-</span>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
