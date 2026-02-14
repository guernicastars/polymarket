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
import { formatNumber } from "@/lib/format";
import type { MarketHolder } from "@/types/market";

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

interface TopHoldersTableProps {
  data: MarketHolder[];
}

export function TopHoldersTable({ data }: TopHoldersTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No holder data available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Holder</TableHead>
            <TableHead className="text-muted-foreground">Outcome</TableHead>
            <TableHead className="text-right text-muted-foreground">Amount</TableHead>
            <TableHead className="text-muted-foreground">Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, i) => (
            <TableRow
              key={`${row.proxy_wallet}-${row.outcome_index}-${i}`}
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
                  <span className="font-medium truncate max-w-[200px]">
                    {row.pseudonym || truncateWallet(row.proxy_wallet)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <Badge
                  variant="secondary"
                  className={`text-xs border-0 ${
                    row.outcome_index === 0
                      ? "bg-[#00d4aa]/10 text-[#00d4aa]"
                      : "bg-[#ff4466]/10 text-[#ff4466]"
                  }`}
                >
                  {row.outcome_index === 0 ? "Yes" : "No"}
                </Badge>
              </TableCell>
              <TableCell className="text-right font-mono">
                {formatNumber(row.amount)}
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                {formatRelativeTime(row.snapshot_time)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
