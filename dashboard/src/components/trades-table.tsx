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
import type { Trade } from "@/types/market";

interface TradesTableProps {
  trades: Trade[];
}

export function TradesTable({ trades }: TradesTableProps) {
  if (!trades || trades.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        No recent trades
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Time</TableHead>
            <TableHead className="text-muted-foreground">Side</TableHead>
            <TableHead className="text-muted-foreground">Outcome</TableHead>
            <TableHead className="text-right text-muted-foreground">Price</TableHead>
            <TableHead className="text-right text-muted-foreground">Size</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {trades.map((trade) => {
            const time = new Date(trade.timestamp);
            const isBuy = trade.side === "buy";
            return (
              <TableRow
                key={trade.trade_id}
                className="border-[#1e1e2e] hover:bg-[#1a1a2e]"
              >
                <TableCell className="font-mono text-xs text-muted-foreground">
                  {time.toLocaleTimeString()}
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
                    {trade.side.toUpperCase()}
                  </Badge>
                </TableCell>
                <TableCell className="text-sm">{trade.outcome}</TableCell>
                <TableCell className="text-right font-mono">
                  <span className={isBuy ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                    {(trade.price * 100).toFixed(1)}\u00A2
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  ${trade.size.toFixed(2)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
