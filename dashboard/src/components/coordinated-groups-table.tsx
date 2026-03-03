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
import type { CoordinatedGroup } from "@/types/market";

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

interface CoordinatedGroupsTableProps {
  data: CoordinatedGroup[];
}

export function CoordinatedGroupsTable({ data }: CoordinatedGroupsTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No coordinated trading groups detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Group</TableHead>
            <TableHead className="text-muted-foreground">Members</TableHead>
            <TableHead className="text-muted-foreground">Correlation</TableHead>
            <TableHead className="text-right text-muted-foreground">Timing</TableHead>
            <TableHead className="text-right text-muted-foreground">Overlap</TableHead>
            <TableHead className="text-right text-muted-foreground">Direction</TableHead>
            <TableHead className="text-right text-muted-foreground">Volume</TableHead>
            <TableHead className="text-right text-muted-foreground">Avg Suspicion</TableHead>
            <TableHead className="text-muted-foreground">Categories</TableHead>
            <TableHead className="text-muted-foreground">Detected</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.group_id}
              className="border-[#1e1e2e] hover:bg-[#1a1a2e] transition-colors"
            >
              <TableCell className="font-mono text-xs text-muted-foreground">
                {row.group_id.slice(0, 8)}...
              </TableCell>
              <TableCell>
                <div className="flex flex-col gap-0.5">
                  {row.wallets.slice(0, 3).map((w) => (
                    <span key={w} className="font-mono text-xs text-muted-foreground">
                      {truncateWallet(w)}
                    </span>
                  ))}
                  {row.wallets.length > 3 && (
                    <Badge variant="secondary" className="text-xs bg-[#1e1e2e] w-fit">
                      +{row.wallets.length - 3} more
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[100px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-violet-400"
                      style={{ width: `${(row.correlation_score * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-10 text-right text-muted-foreground">
                    {(row.correlation_score * 100).toFixed(0)}%
                  </span>
                </div>
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.timing_correlation * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.market_overlap * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.direction_agreement * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {formatUSD(row.total_volume)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                <span className={row.avg_suspicion > 50 ? "text-red-400" : "text-muted-foreground"}>
                  {row.avg_suspicion.toFixed(0)}
                </span>
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1">
                  {(row.common_categories || []).slice(0, 2).map((cat) => (
                    <Badge key={cat} variant="secondary" className="text-xs bg-[#1e1e2e] border-0">
                      {cat}
                    </Badge>
                  ))}
                </div>
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                {formatRelativeTime(row.detected_at)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
