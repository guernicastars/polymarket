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
import { formatUSD, formatPrice, formatNumber } from "@/lib/format";
import type { VolumeAnomaly } from "@/types/market";

function spikeColor(ratio: number): string {
  if (ratio > 5) return "text-[#ff4466]";
  if (ratio > 3) return "text-amber-400";
  return "text-yellow-400";
}

interface VolumeAnomaliesTableProps {
  data: VolumeAnomaly[];
}

export function VolumeAnomaliesTable({ data }: VolumeAnomaliesTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No volume anomalies detected
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-right text-muted-foreground">Price</TableHead>
            <TableHead className="text-right text-muted-foreground">4h Volume</TableHead>
            <TableHead className="text-right text-muted-foreground">Avg Daily</TableHead>
            <TableHead className="text-right text-muted-foreground">Spike Ratio</TableHead>
            <TableHead className="text-right text-muted-foreground">Trades</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.condition_id}
              className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
              onClick={() => router.push(`/market/${row.condition_id}`)}
            >
              <TableCell className="max-w-[350px]">
                <div className="font-medium truncate">{row.question}</div>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className={row.current_price >= 0.5 ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                  {formatPrice(row.current_price)}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono text-muted-foreground">
                {formatUSD(row.volume_4h)}
              </TableCell>
              <TableCell className="text-right font-mono text-muted-foreground">
                {formatUSD(row.avg_daily_volume)}
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className={spikeColor(row.volume_ratio)}>
                  {row.volume_ratio.toFixed(1)}x
                </span>
              </TableCell>
              <TableCell className="text-right font-mono text-muted-foreground">
                {formatNumber(row.trade_count)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
