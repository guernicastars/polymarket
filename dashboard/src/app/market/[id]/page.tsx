import { Suspense } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  getMarketDetail,
  getMarketPriceHistory,
  getMarketTrades,
  getTopHolders,
} from "@/lib/queries";
import { formatUSD, formatPrice } from "@/lib/format";
import { PriceChart, PriceChartSkeleton } from "@/components/price-chart";
import { TradesTable } from "@/components/trades-table";
import { TopHoldersTable } from "@/components/top-holders-table";

export const dynamic = "force-dynamic";

interface MarketPageProps {
  params: Promise<{ id: string }>;
}

export default async function MarketPage({ params }: MarketPageProps) {
  const { id } = await params;

  const [market, priceHistory, trades, holders] = await Promise.all([
    getMarketDetail(id),
    getMarketPriceHistory(id, "Yes", "1m"),
    getMarketTrades(id, 50),
    getTopHolders(id, 20),
  ]);

  if (!market) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <p className="text-lg text-muted-foreground">Market not found</p>
        <Link
          href="/"
          className="text-sm text-[#00d4aa] hover:underline flex items-center gap-1"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to dashboard
        </Link>
      </div>
    );
  }

  const yesPrice = market.outcome_prices?.[0] ?? 0;
  const noPrice = market.outcome_prices?.[1] ?? 0;

  return (
    <div className="space-y-6">
      {/* Back link */}
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to overview
      </Link>

      {/* Market header */}
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-4">
          <h1 className="text-xl font-semibold leading-tight">
            {market.question}
          </h1>
          <Badge
            variant="secondary"
            className="text-xs bg-[#1e1e2e] whitespace-nowrap"
          >
            {market.category || "Other"}
          </Badge>
        </div>
        {market.description && (
          <p className="text-sm text-muted-foreground max-w-3xl">
            {market.description}
          </p>
        )}
      </div>

      {/* Price + Info cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground font-normal">
              Yes Price
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-[#00d4aa]">
              {formatPrice(yesPrice)}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground font-normal">
              No Price
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-[#ff4466]">
              {formatPrice(noPrice)}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground font-normal">
              24h Volume
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono">
              {formatUSD(market.volume_24h)}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground font-normal">
              Liquidity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono">
              {formatUSD(market.liquidity)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Price Chart */}
      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Price Chart</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<PriceChartSkeleton />}>
            <PriceChart conditionId={id} initialData={priceHistory} />
          </Suspense>
        </CardContent>
      </Card>

      {/* Market info + Trades */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">Market Info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Status</p>
                <p className="font-medium">
                  {market.resolved
                    ? "Resolved"
                    : market.closed
                      ? "Closed"
                      : "Active"}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Total Volume</p>
                <p className="font-medium font-mono">
                  {formatUSD(market.volume_total)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">End Date</p>
                <p className="font-medium font-mono">
                  {new Date(market.end_date).toLocaleDateString()}
                </p>
              </div>
              {market.winning_outcome && (
                <div>
                  <p className="text-muted-foreground">Winner</p>
                  <p className="font-medium text-[#00d4aa]">
                    {market.winning_outcome}
                  </p>
                </div>
              )}
            </div>
            <Separator className="bg-[#1e1e2e]" />
            <div>
              <p className="text-xs text-muted-foreground mb-2">Outcomes</p>
              <div className="flex flex-wrap gap-2">
                {market.outcomes?.map((outcome, i) => (
                  <Badge
                    key={outcome}
                    variant="secondary"
                    className="bg-[#1e1e2e] font-mono text-xs"
                  >
                    {outcome}: {formatPrice(market.outcome_prices?.[i] ?? 0)}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">Recent Trades</CardTitle>
          </CardHeader>
          <CardContent>
            <TradesTable trades={trades} />
          </CardContent>
        </Card>
      </div>

      {/* Top Holders */}
      {holders.length > 0 && (
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader>
            <CardTitle className="text-base">Top Holders</CardTitle>
          </CardHeader>
          <CardContent>
            <TopHoldersTable data={holders} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
