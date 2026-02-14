"use client";

import { useRouter } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { formatUSD, formatPrice, formatNumber } from "@/lib/format";
import type { LargeTrade } from "@/types/market";

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

interface LargeTradesTableProps {
  data: LargeTrade[];
}

export function LargeTradesTable({ data }: LargeTradesTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No large trades detected in the last 24 hours
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Outcome</TableHead>
            <TableHead className="text-muted-foreground">Side</TableHead>
            <TableHead className="text-right text-muted-foreground">Price</TableHead>
            <TableHead className="text-right text-muted-foreground">Size</TableHead>
            <TableHead className="text-right text-muted-foreground">Tokens</TableHead>
            <TableHead className="text-muted-foreground">Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => {
            const isBuy = row.side === "buy" || row.side === "BUY";
            return (
              <TableRow
                key={row.trade_id}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() => router.push(`/market/${row.condition_id}`)}
              >
                <TableCell className="max-w-[300px]">
                  <div className="font-medium truncate">{row.question}</div>
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className="text-xs bg-[#1e1e2e]">
                    {row.outcome}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge
                    variant="secondary"
                    className={`text-xs border-0 ${
                      isBuy
                        ? "bg-[#00d4aa]/10 text-[#00d4aa]"
                        : "bg-[#ff4466]/10 text-[#ff4466]"
                    }`}
                  >
                    {row.side.toUpperCase()}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span className={isBuy ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                    {formatPrice(row.price)}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatUSD(row.usd_size)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatNumber(row.size)}
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                  {formatRelativeTime(row.timestamp)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
