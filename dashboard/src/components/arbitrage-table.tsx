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
import type { ArbitrageOpportunity } from "@/types/market";

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

interface ArbitrageTableProps {
  data: ArbitrageOpportunity[];
}

export function ArbitrageTable({ data }: ArbitrageTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No open arbitrage opportunities detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Type</TableHead>
            <TableHead className="text-right text-muted-foreground">Expected</TableHead>
            <TableHead className="text-right text-muted-foreground">Actual</TableHead>
            <TableHead className="text-right text-muted-foreground">Spread</TableHead>
            <TableHead className="text-muted-foreground">Related</TableHead>
            <TableHead className="text-muted-foreground">Description</TableHead>
            <TableHead className="text-muted-foreground">Detected</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, idx) => (
            <TableRow
              key={`${row.condition_id}-${row.arb_type}-${idx}`}
              className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
              onClick={() => router.push(`/market/${row.condition_id}`)}
            >
              <TableCell className="max-w-[250px]">
                <div className="font-medium truncate">
                  {row.question || row.condition_id.slice(0, 12) + "..."}
                </div>
              </TableCell>
              <TableCell>
                <Badge
                  variant="secondary"
                  className={`text-xs border-0 ${
                    row.arb_type === "sum_to_one"
                      ? "bg-blue-400/10 text-blue-400"
                      : "bg-violet-400/10 text-violet-400"
                  }`}
                >
                  {row.arb_type === "sum_to_one" ? "Sum" : "Related"}
                </Badge>
              </TableCell>
              <TableCell className="text-right font-mono text-muted-foreground">
                {row.expected_sum.toFixed(4)}
              </TableCell>
              <TableCell className="text-right font-mono text-muted-foreground">
                {row.actual_sum.toFixed(4)}
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className={row.spread > 0.05 ? "text-[#ff4466]" : "text-amber-400"}>
                  {(row.spread * 100).toFixed(2)}%
                </span>
              </TableCell>
              <TableCell>
                {row.related_condition_ids && row.related_condition_ids.length > 0 ? (
                  <Badge variant="secondary" className="text-xs bg-[#1e1e2e]">
                    {row.related_condition_ids.length}
                  </Badge>
                ) : (
                  <span className="text-xs text-muted-foreground">-</span>
                )}
              </TableCell>
              <TableCell className="max-w-[200px]">
                <div className="text-xs text-muted-foreground truncate">
                  {row.description || "-"}
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
