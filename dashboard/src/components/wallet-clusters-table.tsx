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
import type { WalletCluster } from "@/types/market";

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

interface WalletClustersTableProps {
  data: WalletCluster[];
}

export function WalletClustersTable({ data }: WalletClustersTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No wallet clusters detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Cluster</TableHead>
            <TableHead className="text-muted-foreground">Wallets</TableHead>
            <TableHead className="text-muted-foreground">Similarity</TableHead>
            <TableHead className="text-right text-muted-foreground">Timing</TableHead>
            <TableHead className="text-right text-muted-foreground">Market Overlap</TableHead>
            <TableHead className="text-right text-muted-foreground">Direction</TableHead>
            <TableHead className="text-muted-foreground">Common Markets</TableHead>
            <TableHead className="text-muted-foreground">Detected</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.cluster_id}
              className="border-[#1e1e2e] hover:bg-[#1a1a2e] transition-colors"
            >
              <TableCell className="font-mono text-xs text-muted-foreground">
                {row.cluster_id.slice(0, 8)}...
              </TableCell>
              <TableCell>
                <Badge variant="secondary" className="text-xs bg-violet-400/10 text-violet-400 border-0">
                  {row.size}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[100px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-violet-400"
                      style={{ width: `${(row.similarity_score * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-10 text-right text-muted-foreground">
                    {(row.similarity_score * 100).toFixed(0)}%
                  </span>
                </div>
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.timing_corr * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.market_overlap * 100).toFixed(0)}%
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {(row.direction_agreement * 100).toFixed(0)}%
              </TableCell>
              <TableCell>
                <Badge variant="secondary" className="text-xs bg-[#1e1e2e]">
                  {row.common_markets ? row.common_markets.length : 0}
                </Badge>
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                {formatRelativeTime(row.created_at)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
