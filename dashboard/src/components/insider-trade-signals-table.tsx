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
import type { InsiderTradeSignal } from "@/types/market";

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
  if (score > 70) return "bg-red-400";
  if (score > 50) return "bg-orange-400";
  if (score > 30) return "bg-yellow-400";
  return "bg-emerald-400";
}

function componentColor(value: number): string {
  if (value > 50) return "text-red-400";
  if (value > 30) return "text-amber-400";
  return "text-muted-foreground";
}

interface InsiderTradeSignalsTableProps {
  data: InsiderTradeSignal[];
}

export function InsiderTradeSignalsTable({ data }: InsiderTradeSignalsTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No insider trade signals detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Trader</TableHead>
            <TableHead className="text-muted-foreground">Side</TableHead>
            <TableHead className="text-right text-muted-foreground">Size</TableHead>
            <TableHead className="text-muted-foreground">Score</TableHead>
            <TableHead className="text-right text-muted-foreground">Pre-News</TableHead>
            <TableHead className="text-right text-muted-foreground">Statistical</TableHead>
            <TableHead className="text-right text-muted-foreground">Profit</TableHead>
            <TableHead className="text-right text-muted-foreground">Coord</TableHead>
            <TableHead className="text-muted-foreground">Correct</TableHead>
            <TableHead className="text-right text-muted-foreground">Hrs Before</TableHead>
            <TableHead className="text-muted-foreground">Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.trade_id}
              className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
              onClick={() => router.push(`/market/${row.condition_id}`)}
            >
              <TableCell className="max-w-[200px]">
                <div className="font-medium truncate text-sm">
                  {row.question || row.condition_id.slice(0, 12) + "..."}
                </div>
              </TableCell>
              <TableCell>
                <span className="font-mono text-xs text-muted-foreground">
                  {row.pseudonym || truncateWallet(row.proxy_wallet)}
                </span>
              </TableCell>
              <TableCell>
                <Badge
                  variant="secondary"
                  className={`text-xs border-0 ${
                    row.side === "BUY"
                      ? "bg-[#00d4aa]/15 text-[#00d4aa]"
                      : "bg-[#ff4466]/15 text-[#ff4466]"
                  }`}
                >
                  {row.side}
                </Badge>
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground">
                {formatUSD(row.usdc_size)}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[80px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${scoreBarColor(row.composite_score)}`}
                      style={{ width: `${row.composite_score}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-8 text-right text-muted-foreground">
                    {row.composite_score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.pre_news_score)}`}>
                {row.pre_news_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.statistical_score)}`}>
                {row.statistical_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.profitability_score)}`}>
                {row.profitability_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.coordination_score)}`}>
                {row.coordination_score.toFixed(0)}
              </TableCell>
              <TableCell>
                {row.direction_correct === 1 ? (
                  <Badge variant="secondary" className="text-xs bg-[#00d4aa]/15 text-[#00d4aa] border-0">
                    Yes
                  </Badge>
                ) : (
                  <span className="text-xs text-muted-foreground">No</span>
                )}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                <span className={row.hours_before_move < 6 ? "text-red-400" : "text-muted-foreground"}>
                  {row.hours_before_move.toFixed(1)}
                </span>
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                {formatRelativeTime(row.trade_timestamp)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
