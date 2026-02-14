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
  OBISignal,
  VolumeAnomaly,
  LargeTrade,
  TechnicalBar,
  SignalsOverview,
  TraderRanking,
  WhaleActivityFeed,
  SmartMoneyPosition,
  MarketHolder,
  TraderProfile,
  PositionConcentration,
  WhalesOverview,
  ArbitrageOpportunity,
  WalletCluster,
  InsiderAlert,
  CompositeSignal,
  AnalyticsOverview,
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

// --- Phase 1 Signal Queries ---

export async function getOrderBookImbalance(limit = 50): Promise<OBISignal[]> {
  return query<OBISignal>(
    `SELECT
      os.condition_id,
      m.question,
      os.outcome,
      arraySum(os.bid_sizes) AS total_bid,
      arraySum(os.ask_sizes) AS total_ask,
      arraySum(os.bid_sizes) / greatest(arraySum(os.bid_sizes) + arraySum(os.ask_sizes), 0.001) AS obi,
      if(length(os.bid_prices) > 0, os.bid_prices[1], 0) AS best_bid,
      if(length(os.ask_prices) > 0, os.ask_prices[1], 0) AS best_ask,
      os.snapshot_time
    FROM orderbook_snapshots os
    INNER JOIN (
      SELECT condition_id, outcome, max(snapshot_time) AS max_time
      FROM orderbook_snapshots
      WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
      GROUP BY condition_id, outcome
    ) latest ON os.condition_id = latest.condition_id
      AND os.outcome = latest.outcome
      AND os.snapshot_time = latest.max_time
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON os.condition_id = m.condition_id
    WHERE (arraySum(os.bid_sizes) + arraySum(os.ask_sizes)) > 0
    ORDER BY abs(obi - 0.5) DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getVolumeAnomalies(limit = 30): Promise<VolumeAnomaly[]> {
  return query<VolumeAnomaly>(
    `SELECT
      t.condition_id,
      m.question,
      m.outcome_prices[1] AS current_price,
      sum(t.size) AS volume_4h,
      m.volume_1wk / 7 AS avg_daily_volume,
      sum(t.size) / greatest(m.volume_1wk / 7 / 6, 0.01) AS volume_ratio,
      count() AS trade_count
    FROM market_trades t
    INNER JOIN (
      SELECT condition_id, question, outcome_prices, volume_1wk
      FROM markets FINAL
      WHERE active = 1 AND closed = 0
    ) AS m ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 4 HOUR
    GROUP BY t.condition_id, m.question, m.outcome_prices, m.volume_1wk
    HAVING volume_ratio > 2.0 AND volume_4h > 100
    ORDER BY volume_ratio DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getLargeTrades(
  minSize = 1000,
  limit = 50
): Promise<LargeTrade[]> {
  return query<LargeTrade>(
    `SELECT
      t.condition_id,
      m.question,
      t.outcome,
      t.price,
      t.size,
      t.price * t.size AS usd_size,
      t.side,
      t.trade_id,
      t.timestamp
    FROM market_trades t
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON t.condition_id = m.condition_id
    WHERE t.timestamp >= now() - INTERVAL 24 HOUR
      AND t.price * t.size >= {minSize:Float64}
    ORDER BY usd_size DESC
    LIMIT {limit:UInt32}`,
    { minSize, limit }
  );
}

export async function getMarketTechnicals(
  conditionId: string,
  outcome = "Yes"
): Promise<TechnicalBar[]> {
  return query<TechnicalBar>(
    `WITH hourly AS (
      SELECT
        bar_time,
        argMinMerge(open) AS o,
        maxMerge(high) AS h,
        minMerge(low) AS l,
        argMaxMerge(close) AS c,
        sumMerge(volume) AS v
      FROM ohlcv_1h
      WHERE condition_id = {conditionId:String}
        AND outcome = {outcome:String}
        AND bar_time >= now() - INTERVAL 7 DAY
      GROUP BY bar_time
      ORDER BY bar_time
    ),
    deltas AS (
      SELECT
        bar_time, o, h, l, c, v,
        c - lagInFrame(c, 1, c) OVER (ORDER BY bar_time) AS delta
      FROM hourly
    ),
    gains_losses AS (
      SELECT
        bar_time, o, h, l, c, v, delta,
        if(delta > 0, delta, 0) AS gain,
        if(delta < 0, abs(delta), 0) AS loss
      FROM deltas
    )
    SELECT
      bar_time,
      o AS open,
      h AS high,
      l AS low,
      c AS close,
      v AS volume,
      100 - (100 / (1 + (
        avg(gain) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
        / greatest(avg(loss) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0.0001)
      ))) AS rsi,
      sum(c * v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time) /
        greatest(sum(v) OVER (PARTITION BY toDate(bar_time) ORDER BY bar_time), 0.0001) AS vwap,
      if(lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) > 0,
        (c - lagInFrame(c, 24, 0) OVER (ORDER BY bar_time)) / lagInFrame(c, 24, 0) OVER (ORDER BY bar_time) * 100,
        0) AS momentum
    FROM gains_losses
    ORDER BY bar_time`,
    { conditionId, outcome }
  );
}

export async function getSignalsOverview(): Promise<SignalsOverview> {
  const rows = await query<SignalsOverview>(
    `SELECT
      (
        SELECT count(DISTINCT condition_id)
        FROM orderbook_snapshots
        WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
          AND arraySum(bid_sizes) + arraySum(ask_sizes) > 0
          AND abs(arraySum(bid_sizes) / (arraySum(bid_sizes) + arraySum(ask_sizes)) - 0.5) > 0.15
      ) AS obi_signals,
      (
        SELECT uniq(condition_id)
        FROM market_trades
        WHERE timestamp >= now() - INTERVAL 4 HOUR
      ) AS volume_active_markets,
      (
        SELECT count()
        FROM market_trades
        WHERE timestamp >= now() - INTERVAL 24 HOUR
          AND price * size >= 1000
      ) AS large_trades_24h,
      (
        SELECT countIf(active = 1 AND closed = 0)
        FROM markets FINAL
      ) AS active_markets`
  );
  return rows[0] ?? {
    obi_signals: 0,
    volume_active_markets: 0,
    large_trades_24h: 0,
    active_markets: 0,
  };
}

// --- Phase 2 Whale/User Queries ---

export async function getLeaderboard(
  category = "OVERALL",
  timePeriod = "ALL",
  orderBy = "PNL",
  limit = 50
): Promise<TraderRanking[]> {
  return query<TraderRanking>(
    `SELECT
      proxy_wallet,
      user_name,
      profile_image,
      rank,
      category,
      time_period,
      order_by,
      pnl,
      volume,
      verified_badge,
      x_username,
      snapshot_time
    FROM trader_rankings FINAL
    WHERE category = {category:String}
      AND time_period = {timePeriod:String}
      AND order_by = {orderBy:String}
    ORDER BY rank ASC
    LIMIT {limit:UInt32}`,
    { category, timePeriod, orderBy, limit }
  );
}

export async function getWhaleActivity(limit = 50): Promise<WhaleActivityFeed[]> {
  return query<WhaleActivityFeed>(
    `SELECT
      wa.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      wa.condition_id,
      wa.activity_type,
      wa.side,
      wa.outcome,
      wa.size,
      wa.usdc_size,
      wa.price,
      wa.title,
      wa.timestamp
    FROM wallet_activity wa
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON wa.proxy_wallet = tp.proxy_wallet
    WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
      AND wa.usdc_size >= 500
    ORDER BY wa.timestamp DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getSmartMoneyPositions(limit = 50): Promise<SmartMoneyPosition[]> {
  return query<SmartMoneyPosition>(
    `SELECT
      wp.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      tr.rank,
      wp.condition_id,
      wp.outcome,
      wp.size,
      wp.current_value,
      wp.cash_pnl,
      wp.percent_pnl,
      wp.title
    FROM (SELECT * FROM wallet_positions FINAL) AS wp
    INNER JOIN (
      SELECT proxy_wallet, min(rank) AS rank
      FROM trader_rankings FINAL
      WHERE category = 'OVERALL' AND time_period = 'ALL' AND order_by = 'PNL'
      GROUP BY proxy_wallet
    ) AS tr ON wp.proxy_wallet = tr.proxy_wallet
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON wp.proxy_wallet = tp.proxy_wallet
    WHERE wp.size > 0
      AND wp.current_value > 100
    ORDER BY tr.rank ASC, wp.current_value DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getTopHolders(
  conditionId: string,
  limit = 20
): Promise<MarketHolder[]> {
  return query<MarketHolder>(
    `SELECT
      condition_id,
      proxy_wallet,
      pseudonym,
      profile_image,
      outcome_index,
      amount,
      snapshot_time
    FROM market_holders FINAL
    WHERE condition_id = {conditionId:String}
    ORDER BY amount DESC
    LIMIT {limit:UInt32}`,
    { conditionId, limit }
  );
}

export async function getTraderProfile(
  wallet: string
): Promise<TraderProfile | null> {
  const rows = await query<TraderProfile>(
    `SELECT
      proxy_wallet,
      pseudonym,
      name,
      bio,
      profile_image,
      x_username,
      verified_badge,
      first_seen_at
    FROM trader_profiles FINAL
    WHERE proxy_wallet = {wallet:String}
    LIMIT 1`,
    { wallet }
  );
  return rows[0] ?? null;
}

export async function getPositionConcentration(
  limit = 30
): Promise<PositionConcentration[]> {
  return query<PositionConcentration>(
    `WITH holder_stats AS (
      SELECT
        condition_id,
        count() AS total_holders,
        sum(amount) AS total_amount,
        arraySlice(
          groupArray(amount) AS amounts_sorted,
          1, 5
        ) AS top5_amounts,
        arraySlice(
          groupArray(proxy_wallet) AS wallets_sorted,
          1, 1
        ) AS top_wallets
      FROM (
        SELECT condition_id, proxy_wallet, amount
        FROM market_holders FINAL
        ORDER BY amount DESC
      )
      GROUP BY condition_id
      HAVING total_holders >= 3
    )
    SELECT
      hs.condition_id,
      m.question,
      hs.total_holders,
      hs.total_amount,
      arraySum(hs.top5_amounts) AS top5_amount,
      arraySum(hs.top5_amounts) / greatest(hs.total_amount, 0.01) AS top5_share,
      hs.top_wallets[1] AS top_holder_wallet,
      hs.top5_amounts[1] AS top_holder_amount
    FROM holder_stats hs
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON hs.condition_id = m.condition_id
    ORDER BY top5_share DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getWhalesOverview(): Promise<WhalesOverview> {
  const rows = await query<WhalesOverview>(
    `SELECT
      (SELECT uniq(proxy_wallet) FROM trader_rankings FINAL) AS tracked_wallets,
      (
        SELECT count()
        FROM wallet_activity
        WHERE timestamp >= now() - INTERVAL 24 HOUR
          AND usdc_size >= 500
      ) AS whale_trades_24h,
      (
        SELECT count()
        FROM (SELECT * FROM wallet_positions FINAL)
        WHERE size > 0
      ) AS total_whale_positions,
      (
        SELECT uniq(condition_id)
        FROM (SELECT * FROM wallet_positions FINAL)
        WHERE size > 0
      ) AS unique_markets_held`
  );
  return rows[0] ?? {
    tracked_wallets: 0,
    whale_trades_24h: 0,
    total_whale_positions: 0,
    unique_markets_held: 0,
  };
}

// --- Phase 3 Analytics Queries ---

export async function getArbitrageOpportunities(
  limit = 50
): Promise<ArbitrageOpportunity[]> {
  return query<ArbitrageOpportunity>(
    `SELECT
      ao.condition_id,
      ao.event_slug,
      ao.arb_type,
      ao.expected_sum,
      ao.actual_sum,
      ao.spread,
      ao.related_condition_ids,
      ao.description,
      ao.status,
      ao.detected_at,
      ao.resolved_at,
      m.question
    FROM (SELECT * FROM arbitrage_opportunities FINAL) AS ao
    LEFT JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON ao.condition_id = m.condition_id
    WHERE ao.status = 'open'
    ORDER BY ao.spread DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getWalletClusters(
  limit = 30
): Promise<WalletCluster[]> {
  return query<WalletCluster>(
    `SELECT
      cluster_id,
      wallets,
      size,
      similarity_score,
      timing_corr,
      market_overlap,
      direction_agreement,
      common_markets,
      label,
      created_at
    FROM wallet_clusters FINAL
    ORDER BY similarity_score DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getInsiderAlerts(
  minScore = 30,
  limit = 50
): Promise<InsiderAlert[]> {
  return query<InsiderAlert>(
    `SELECT
      ins.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      ins.score,
      ins.freshness_score,
      ins.win_rate_score,
      ins.niche_score,
      ins.size_score,
      ins.timing_score,
      ins.computed_at
    FROM (SELECT * FROM insider_scores FINAL) AS ins
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON ins.proxy_wallet = tp.proxy_wallet
    WHERE ins.score >= {minScore:Float64}
    ORDER BY ins.score DESC
    LIMIT {limit:UInt32}`,
    { minScore, limit }
  );
}

export async function getCompositeSignals(
  limit = 50
): Promise<CompositeSignal[]> {
  return query<CompositeSignal>(
    `SELECT
      cs.condition_id,
      m.question,
      cs.score,
      cs.confidence,
      cs.obi_score,
      cs.volume_score,
      cs.trade_bias_score,
      cs.momentum_score,
      cs.smart_money_score,
      cs.concentration_score,
      cs.arbitrage_flag,
      cs.insider_activity,
      cs.computed_at
    FROM (SELECT * FROM composite_signals FINAL) AS cs
    INNER JOIN (
      SELECT condition_id, question
      FROM markets FINAL
      WHERE active = 1 AND closed = 0
    ) AS m ON cs.condition_id = m.condition_id
    ORDER BY abs(cs.score) DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}

export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  const rows = await query<AnalyticsOverview>(
    `SELECT
      (
        SELECT count()
        FROM arbitrage_opportunities FINAL
        WHERE status = 'open'
      ) AS open_arbitrages,
      (
        SELECT count()
        FROM wallet_clusters FINAL
      ) AS wallet_clusters,
      (
        SELECT count()
        FROM insider_scores FINAL
        WHERE score > 50
      ) AS insider_alerts,
      (
        SELECT count()
        FROM composite_signals FINAL
      ) AS markets_scored,
      (
        SELECT avg(confidence)
        FROM composite_signals FINAL
      ) AS avg_confidence`
  );
  return rows[0] ?? {
    open_arbitrages: 0,
    wallet_clusters: 0,
    insider_alerts: 0,
    markets_scored: 0,
    avg_confidence: 0,
  };
}
