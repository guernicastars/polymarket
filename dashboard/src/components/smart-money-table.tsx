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
import { formatUSD, formatPct } from "@/lib/format";
import type { SmartMoneyPosition } from "@/types/market";

function truncateWallet(wallet: string): string {
  if (wallet.length <= 10) return wallet;
  return `${wallet.slice(0, 6)}...${wallet.slice(-4)}`;
}

interface SmartMoneyTableProps {
  data: SmartMoneyPosition[];
}

export function SmartMoneyTable({ data }: SmartMoneyTableProps) {
  const router = useRouter();

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No smart money positions available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground w-16">Rank</TableHead>
            <TableHead className="text-muted-foreground">Trader</TableHead>
            <TableHead className="text-muted-foreground">Market</TableHead>
            <TableHead className="text-muted-foreground">Outcome</TableHead>
            <TableHead className="text-right text-muted-foreground">Position</TableHead>
            <TableHead className="text-right text-muted-foreground">PnL</TableHead>
            <TableHead className="text-right text-muted-foreground">Return</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, i) => {
            const pnlPositive = row.cash_pnl >= 0;
            const returnPositive = row.percent_pnl >= 0;
            return (
              <TableRow
                key={`${row.proxy_wallet}-${row.condition_id}-${i}`}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() => router.push(`/market/${row.condition_id}`)}
              >
                <TableCell className="font-mono text-muted-foreground">
                  #{row.rank}
                </TableCell>
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
                  <Badge variant="secondary" className="text-xs bg-[#1e1e2e]">
                    {row.outcome}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatUSD(row.current_value)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span className={pnlPositive ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                    {pnlPositive ? "+" : ""}{formatUSD(row.cash_pnl)}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span className={returnPositive ? "text-[#00d4aa]" : "text-[#ff4466]"}>
                    {formatPct(row.percent_pnl)}
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
