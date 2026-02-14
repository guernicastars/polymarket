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
import { formatUSD, formatPrice } from "@/lib/format";
import type { WhaleActivityFeed } from "@/types/market";

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

const activityTypeColors: Record<string, string> = {
  TRADE: "bg-blue-500/10 text-blue-400",
  SPLIT: "bg-purple-500/10 text-purple-400",
  MERGE: "bg-indigo-500/10 text-indigo-400",
  REDEEM: "bg-emerald-500/10 text-emerald-400",
  REWARD: "bg-amber-500/10 text-amber-400",
  CONVERSION: "bg-cyan-500/10 text-cyan-400",
  MAKER_REBATE: "bg-pink-500/10 text-pink-400",
};

interface WhaleActivityTableProps {
  data: WhaleActivityFeed[];
}

export function WhaleActivityTable({ data }: WhaleActivityTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No whale activity in the last 24 hours
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground">Trader</TableHead>
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Type</TableHead>
            <TableHead className="text-muted-foreground">Side</TableHead>
            <TableHead className="text-right text-muted-foreground">Size</TableHead>
            <TableHead className="text-right text-muted-foreground">Price</TableHead>
            <TableHead className="text-muted-foreground">Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, i) => {
            const isBuy = row.side === "BUY";
            const typeColor = activityTypeColors[row.activity_type] || "bg-[#1e1e2e] text-muted-foreground";
            return (
              <TableRow
                key={`${row.proxy_wallet}-${row.timestamp}-${i}`}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() => router.push(`/market/${row.condition_id}`)}
              >
                <TableCell>
                  <div className="flex items-center gap-2">
                    {row.profile_image ? (
                      <img
                        src={row.profile_image}
                        alt=""
                        className="h-6 w-6 rounded-full bg-[#1e1e2e]"
                      />
                    ) : (
                      <div className="h-6 w-6 rounded-full bg-[#1e1e2e]" />
                    )}
                    <span className="font-medium truncate max-w-[120px]">
                      {row.pseudonym || truncateWallet(row.proxy_wallet)}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="max-w-[250px]">
                  <div className="font-medium truncate">{row.title}</div>
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className={`text-xs border-0 ${typeColor}`}>
                    {row.activity_type}
                  </Badge>
                </TableCell>
                <TableCell>
                  {row.side ? (
                    <Badge
                      variant="secondary"
                      className={`text-xs border-0 ${
                        isBuy
                          ? "bg-[#00d4aa]/10 text-[#00d4aa]"
                          : "bg-[#ff4466]/10 text-[#ff4466]"
                      }`}
                    >
                      {row.side}
                    </Badge>
                  ) : (
                    <span className="text-xs text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatUSD(row.usdc_size)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {row.price > 0 ? formatPrice(row.price) : "-"}
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
