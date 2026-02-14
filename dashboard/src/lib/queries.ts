import { query } from "./clickhouse";
import type {
  MarketRow,
  TopMover,
  TrendingMarket,
  Market,
  OHLCVBar,
  Trade,
  VolumeLeader,
  CategoryBreakdown,
  ResolvedMarket,
  OverviewStats,
} from "@/types/market";

export async function getTopMarkets(limit = 50): Promise<MarketRow[]> {
  return query<MarketRow>(
    `SELECT
      condition_id,
      question,
      category,
      outcome_prices,
      volume_24h,
      volume_total,
      liquidity,
      end_date
    FROM markets FINAL
    WHERE active = 1 AND closed = 0
    ORDER BY volume_24h DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getTopMovers(limit = 20): Promise<TopMover[]> {
  return query<TopMover>(
    `SELECT
      condition_id,
      question,
      'Yes' AS outcome,
      outcome_prices[1] AS current_price,
      outcome_prices[1] - one_day_price_change AS prev_price,
      one_day_price_change AS price_change,
      if(abs(outcome_prices[1] - one_day_price_change) > 0.001,
         one_day_price_change / (outcome_prices[1] - one_day_price_change) * 100,
         0) AS pct_change
    FROM markets FINAL
    WHERE active = 1
      AND closed = 0
      AND abs(one_day_price_change) > 0.001
      AND outcome_prices[1] > 0.05
      AND (outcome_prices[1] - one_day_price_change) > 0.05
    ORDER BY abs(pct_change) DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getTrendingMarkets(
  limit = 20
): Promise<TrendingMarket[]> {
  return query<TrendingMarket>(
    `SELECT
      condition_id,
      question,
      volume_24h AS volume_1h,
      volume_1wk / 7 AS avg_hourly_volume,
      volume_24h / greatest(volume_1wk / 7, 0.01) AS volume_ratio,
      outcome_prices[1] AS current_price
    FROM markets FINAL
    WHERE active = 1
      AND closed = 0
      AND volume_1wk > 0
      AND volume_24h > 0
      AND volume_24h / (volume_1wk / 7) > 1.5
    ORDER BY volume_ratio DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getMarketDetail(
  conditionId: string
): Promise<Market | null> {
  const rows = await query<Market>(
    `SELECT
      condition_id,
      market_slug,
      question,
      description,
      category,
      outcomes,
      outcome_prices,
      token_ids,
      active,
      closed,
      resolved,
      winning_outcome,
      volume_24h,
      volume_total,
      liquidity,
      start_date,
      end_date
    FROM markets FINAL
    WHERE condition_id = {conditionId:String}
    LIMIT 1`,
    { conditionId }
  );
  return rows[0] ?? null;
}

export async function getMarketPriceHistory(
  conditionId: string,
  outcome = "Yes",
  interval: "1m" | "1h" = "1m"
): Promise<OHLCVBar[]> {
  const table = interval === "1h" ? "ohlcv_1h" : "ohlcv_1m";
  return query<OHLCVBar>(
    `SELECT
      bar_time,
      argMinMerge(open) AS open,
      maxMerge(high) AS high,
      minMerge(low) AS low,
      argMaxMerge(close) AS close,
      sumMerge(volume) AS volume
    FROM ${table}
    WHERE condition_id = {conditionId:String}
      AND outcome = {outcome:String}
    GROUP BY bar_time
    ORDER BY bar_time`,
    { conditionId, outcome }
  );
}

export async function getMarketTrades(
  conditionId: string,
  limit = 50
): Promise<Trade[]> {
  return query<Trade>(
    `SELECT
      condition_id,
      token_id,
      outcome,
      price,
      size,
      side,
      trade_id,
      timestamp
    FROM market_trades
    WHERE condition_id = {conditionId:String}
    ORDER BY timestamp DESC
    LIMIT {limit:UInt32}`,
    { conditionId, limit }
  );
}

export async function getVolumeLeaders(limit = 20): Promise<VolumeLeader[]> {
  return query<VolumeLeader>(
    `SELECT
      t.condition_id,
      m.question,
      sum(t.size) AS volume_24h,
      count() AS trade_count
    FROM market_trades t
    LEFT JOIN (SELECT condition_id, question FROM markets FINAL) AS m ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 24 HOUR
    GROUP BY t.condition_id, m.question
    ORDER BY volume_24h DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getCategoryBreakdown(): Promise<CategoryBreakdown[]> {
  return query<CategoryBreakdown>(
    `SELECT
      category,
      count() AS market_count,
      sum(volume_total) AS total_volume,
      avg(liquidity) AS avg_liquidity
    FROM markets FINAL
    WHERE active = 1
    GROUP BY category
    ORDER BY total_volume DESC`
  );
}

export async function getRecentlyResolved(
  limit = 20
): Promise<ResolvedMarket[]> {
  return query<ResolvedMarket>(
    `SELECT
      condition_id,
      question,
      winning_outcome,
      outcome_prices,
      volume_total,
      end_date
    FROM markets FINAL
    WHERE resolved = 1
    ORDER BY end_date DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getOverviewStats(): Promise<OverviewStats> {
  const rows = await query<OverviewStats>(
    `SELECT
      count() AS total_markets,
      countIf(active = 1 AND closed = 0) AS active_markets,
      sum(volume_24h) AS total_volume_24h,
      countIf(active = 1 AND closed = 0 AND volume_1wk > 0 AND volume_24h > 0
              AND volume_24h / (volume_1wk / 7) > 1.5) AS trending_count
    FROM markets FINAL`
  );
  return rows[0] ?? {
    total_markets: 0,
    active_markets: 0,
    total_volume_24h: 0,
    trending_count: 0,
  };
}
