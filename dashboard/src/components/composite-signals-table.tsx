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
import type { CompositeSignal } from "@/types/market";

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

function scoreColor(score: number): string {
  if (score > 30) return "text-[#00d4aa]";
  if (score < -30) return "text-[#ff4466]";
  return "text-muted-foreground";
}

function scoreBarColor(score: number): string {
  if (score > 0) return "bg-[#00d4aa]";
  if (score < 0) return "bg-[#ff4466]";
  return "bg-muted-foreground";
}

function componentColor(value: number): string {
  if (value > 20) return "text-[#00d4aa]";
  if (value < -20) return "text-[#ff4466]";
  return "text-muted-foreground";
}

interface CompositeSignalsTableProps {
  data: CompositeSignal[];
}

export function CompositeSignalsTable({ data }: CompositeSignalsTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No composite signals computed yet
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Score</TableHead>
            <TableHead className="text-muted-foreground">Confidence</TableHead>
            <TableHead className="text-right text-muted-foreground">OBI</TableHead>
            <TableHead className="text-right text-muted-foreground">Volume</TableHead>
            <TableHead className="text-right text-muted-foreground">Trades</TableHead>
            <TableHead className="text-right text-muted-foreground">Momentum</TableHead>
            <TableHead className="text-right text-muted-foreground">Smart Money</TableHead>
            <TableHead className="text-muted-foreground">Arb</TableHead>
            <TableHead className="text-muted-foreground">Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.condition_id}
              className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
              onClick={() => router.push(`/market/${row.condition_id}`)}
            >
              <TableCell className="max-w-[250px]">
                <div className="font-medium truncate">{row.question}</div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[120px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden relative">
                    <div
                      className={`absolute h-full rounded-full ${scoreBarColor(row.score)}`}
                      style={{
                        width: `${Math.abs(row.score)}%`,
                        left: row.score >= 0 ? "50%" : `${50 - Math.abs(row.score)}%`,
                      }}
                    />
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[#2a2a3e]" />
                  </div>
                  <span className={`font-mono text-xs w-12 text-right ${scoreColor(row.score)}`}>
                    {row.score > 0 ? "+" : ""}{row.score.toFixed(0)}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 min-w-[80px]">
                  <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-400"
                      style={{ width: `${(row.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs w-10 text-right text-muted-foreground">
                    {(row.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.obi_score)}`}>
                {row.obi_score > 0 ? "+" : ""}{row.obi_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.volume_score)}`}>
                {row.volume_score > 0 ? "+" : ""}{row.volume_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.trade_bias_score)}`}>
                {row.trade_bias_score > 0 ? "+" : ""}{row.trade_bias_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.momentum_score)}`}>
                {row.momentum_score > 0 ? "+" : ""}{row.momentum_score.toFixed(0)}
              </TableCell>
              <TableCell className={`text-right font-mono text-xs ${componentColor(row.smart_money_score)}`}>
                {row.smart_money_score > 0 ? "+" : ""}{row.smart_money_score.toFixed(0)}
              </TableCell>
              <TableCell>
                {row.arbitrage_flag === 1 ? (
                  <Badge variant="secondary" className="text-xs bg-amber-400/10 text-amber-400 border-0">
                    Yes
                  </Badge>
                ) : null}
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
