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
import { formatUSD } from "@/lib/format";
import type { OBISignal } from "@/types/market";

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

function getSignalLabel(obi: number): { label: string; className: string } {
  if (obi > 0.6) {
    return {
      label: "Bullish",
      className: "bg-[#00d4aa]/10 text-[#00d4aa] border-0",
    };
  }
  if (obi < 0.4) {
    return {
      label: "Bearish",
      className: "bg-[#ff4466]/10 text-[#ff4466] border-0",
    };
  }
  return {
    label: "Neutral",
    className: "bg-[#1e1e2e] text-muted-foreground border-0",
  };
}

function obiBarColor(obi: number): string {
  if (obi > 0.6) return "bg-[#00d4aa]";
  if (obi < 0.4) return "bg-[#ff4466]";
  return "bg-muted-foreground";
}

interface OBISignalsTableProps {
  data: OBISignal[];
}

export function OBISignalsTable({ data }: OBISignalsTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No order book imbalance signals detected
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
            <TableHead className="text-muted-foreground">OBI</TableHead>
            <TableHead className="text-right text-muted-foreground">Bid Depth</TableHead>
            <TableHead className="text-right text-muted-foreground">Ask Depth</TableHead>
            <TableHead className="text-right text-muted-foreground">Spread</TableHead>
            <TableHead className="text-muted-foreground">Signal</TableHead>
            <TableHead className="text-muted-foreground">Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => {
            const signal = getSignalLabel(row.obi);
            const spread = row.best_ask - row.best_bid;
            return (
              <TableRow
                key={`${row.condition_id}-${row.outcome}-${row.snapshot_time}`}
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
                  <div className="flex items-center gap-2 min-w-[120px]">
                    <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${obiBarColor(row.obi)}`}
                        style={{ width: `${(row.obi * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs w-12 text-right">
                      {(row.obi * 100).toFixed(1)}%
                    </span>
                  </div>
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatUSD(row.total_bid)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatUSD(row.total_ask)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {(spread * 100).toFixed(1)}&cent;
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className={`text-xs ${signal.className}`}>
                    {signal.label}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                  {formatRelativeTime(row.snapshot_time)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
