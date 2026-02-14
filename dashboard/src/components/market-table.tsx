"use client";

import { useRouter } from "next/navigation";
import useSWR from "swr";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatUSD, formatPrice, formatPct } from "@/lib/format";
import type { MarketRow } from "@/types/market";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

interface MarketTableProps {
  initialData: MarketRow[];
}

export function MarketTable({ initialData }: MarketTableProps) {
  const router = useRouter();
  const { data: markets } = useSWR<MarketRow[]>("/api/markets?limit=50", fetcher, {
    fallbackData: initialData,
    refreshInterval: 10_000,
    revalidateOnFocus: true,
    dedupingInterval: 5_000,
  });

  if (!markets || markets.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No active markets found
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
            <TableHead className="text-right text-muted-foreground">24h Vol</TableHead>
            <TableHead className="text-right text-muted-foreground">Liquidity</TableHead>
            <TableHead className="text-muted-foreground">Category</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {markets.map((market) => {
            const yesPrice = market.outcome_prices?.[0] ?? 0;
            return (
              <TableRow
                key={market.condition_id}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() => router.push(`/market/${market.condition_id}`)}
              >
                <TableCell className="max-w-[400px]">
                  <div className="font-medium truncate">{market.question}</div>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span
                    className={
                      yesPrice >= 0.5 ? "text-[#00d4aa]" : "text-[#ff4466]"
                    }
                  >
                    {formatPrice(yesPrice)}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatUSD(market.volume_24h)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatUSD(market.liquidity)}
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className="text-xs bg-[#1e1e2e]">
                    {market.category || "Other"}
                  </Badge>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function MarketTableSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 px-4">
          <Skeleton className="h-4 flex-1 bg-[#1e1e2e]" />
          <Skeleton className="h-4 w-16 bg-[#1e1e2e]" />
          <Skeleton className="h-4 w-20 bg-[#1e1e2e]" />
          <Skeleton className="h-4 w-20 bg-[#1e1e2e]" />
          <Skeleton className="h-4 w-16 bg-[#1e1e2e]" />
        </div>
      ))}
    </div>
  );
}
