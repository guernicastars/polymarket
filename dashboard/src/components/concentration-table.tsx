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
import { formatNumber } from "@/lib/format";
import type { PositionConcentration } from "@/types/market";

function truncateWallet(wallet: string): string {
  if (!wallet || wallet.length <= 10) return wallet || "-";
  return `${wallet.slice(0, 6)}...${wallet.slice(-4)}`;
}

interface ConcentrationTableProps {
  data: PositionConcentration[];
}

export function ConcentrationTable({ data }: ConcentrationTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No concentration data available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-right text-muted-foreground">Holders</TableHead>
            <TableHead className="text-right text-muted-foreground">Total Held</TableHead>
            <TableHead className="text-muted-foreground">Top-5 Share</TableHead>
            <TableHead className="text-muted-foreground">Top Holder</TableHead>
            <TableHead className="text-right text-muted-foreground">Top Amount</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => {
            const sharePercent = Math.min(row.top5_share * 100, 100);
            const barColor =
              sharePercent > 80
                ? "bg-[#ff4466]"
                : sharePercent > 50
                  ? "bg-amber-400"
                  : "bg-[#00d4aa]";
            return (
              <TableRow
                key={row.condition_id}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() => router.push(`/market/${row.condition_id}`)}
              >
                <TableCell className="max-w-[300px]">
                  <div className="font-medium truncate">{row.question}</div>
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatNumber(row.total_holders)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatNumber(row.total_amount)}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2 min-w-[120px]">
                    <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${barColor}`}
                        style={{ width: `${sharePercent.toFixed(0)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs w-12 text-right">
                      {sharePercent.toFixed(1)}%
                    </span>
                  </div>
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground">
                  {truncateWallet(row.top_holder_wallet)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatNumber(row.top_holder_amount)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
