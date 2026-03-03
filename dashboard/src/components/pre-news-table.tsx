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
import type { PreNewsEvent } from "@/types/market";

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

function eventTypeBadge(type: string): { label: string; className: string } {
  switch (type) {
    case "resolution":
      return { label: "Resolution", className: "bg-red-500/15 text-red-400" };
    case "price_move":
      return { label: "Price Move", className: "bg-amber-500/15 text-amber-400" };
    case "volume_spike":
      return { label: "Vol Spike", className: "bg-blue-500/15 text-blue-400" };
    default:
      return { label: type, className: "bg-[#1e1e2e] text-muted-foreground" };
  }
}

interface PreNewsTableProps {
  data: PreNewsEvent[];
}

export function PreNewsTable({ data }: PreNewsTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No pre-news events detected
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
            <TableHead className="text-muted-foreground">Direction</TableHead>
            <TableHead className="text-right text-muted-foreground">Magnitude</TableHead>
            <TableHead className="text-right text-muted-foreground">Price Before</TableHead>
            <TableHead className="text-right text-muted-foreground">Price After</TableHead>
            <TableHead className="text-muted-foreground">Category</TableHead>
            <TableHead className="text-right text-muted-foreground">Preceding Trades</TableHead>
            <TableHead className="text-muted-foreground">Detected</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => {
            const badge = eventTypeBadge(row.event_type);
            return (
              <TableRow
                key={row.event_id}
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
                    className={`text-xs border-0 ${badge.className}`}
                  >
                    {badge.label}
                  </Badge>
                </TableCell>
                <TableCell>
                  {row.direction === "up" ? (
                    <span className="text-[#00d4aa] font-mono text-xs">UP</span>
                  ) : row.direction === "down" ? (
                    <span className="text-[#ff4466] font-mono text-xs">DOWN</span>
                  ) : (
                    <span className="text-muted-foreground text-xs">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  <span className={row.magnitude > 10 ? "text-red-400" : "text-amber-400"}>
                    {row.magnitude.toFixed(1)}%
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono text-xs text-muted-foreground">
                  {row.price_before.toFixed(3)}
                </TableCell>
                <TableCell className="text-right font-mono text-xs text-muted-foreground">
                  {row.price_after.toFixed(3)}
                </TableCell>
                <TableCell>
                  {row.category ? (
                    <Badge variant="secondary" className="text-xs bg-[#1e1e2e] border-0">
                      {row.category}
                    </Badge>
                  ) : (
                    <span className="text-xs text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  <span className={row.preceding_trades > 3 ? "text-red-400" : "text-muted-foreground"}>
                    {row.preceding_trades}
                  </span>
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                  {formatRelativeTime(row.detected_at)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
