export interface Market {
  condition_id: string;
  market_slug: string;
  question: string;
  description: string;
  category: string;
  outcomes: string[];
  outcome_prices: number[];
  token_ids: string[];
  active: number;
  closed: number;
  resolved: number;
  winning_outcome: string;
  volume_24h: number;
  volume_total: number;
  liquidity: number;
  start_date: string;
  end_date: string;
}

export interface MarketRow {
  condition_id: string;
  question: string;
  category: string;
  outcome_prices: number[];
  volume_24h: number;
  volume_total: number;
  liquidity: number;
  end_date: string;
}

export interface TopMover {
  condition_id: string;
  question: string;
  outcome: string;
  current_price: number;
  prev_price: number;
  price_change: number;
  pct_change: number;
}

export interface TrendingMarket {
  condition_id: string;
  question: string;
  volume_1h: number;
  avg_hourly_volume: number;
  volume_ratio: number;
  current_price: number;
}

export interface OHLCVBar {
  bar_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  condition_id: string;
  token_id: string;
  outcome: string;
  price: number;
  size: number;
  side: string;
  trade_id: string;
  timestamp: string;
}

export interface CategoryBreakdown {
  category: string;
  market_count: number;
  total_volume: number;
  avg_liquidity: number;
}

export interface OverviewStats {
  total_markets: number;
  active_markets: number;
  total_volume_24h: number;
  trending_count: number;
}

export interface VolumeLeader {
  condition_id: string;
  question: string;
  volume_24h: number;
  trade_count: number;
}

export interface ResolvedMarket {
  condition_id: string;
  question: string;
  winning_outcome: string;
  outcome_prices: number[];
  volume_total: number;
  end_date: string;
}

// --- Phase 1 Signal Types ---

export interface OBISignal {
  condition_id: string;
  question: string;
  outcome: string;
  total_bid: number;
  total_ask: number;
  obi: number;            // 0-1, >0.6 bullish, <0.4 bearish
  best_bid: number;
  best_ask: number;
  snapshot_time: string;
}

export interface VolumeAnomaly {
  condition_id: string;
  question: string;
  current_price: number;
  volume_4h: number;
  avg_daily_volume: number;
  volume_ratio: number;   // >2.0 = anomalous
  trade_count: number;
}

export interface LargeTrade {
  condition_id: string;
  question: string;
  outcome: string;
  price: number;
  size: number;
  usd_size: number;
  side: string;
  trade_id: string;
  timestamp: string;
}

export interface TechnicalBar {
  bar_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi: number;           // 0-100
  vwap: number;
  momentum: number;      // % change over 24h
}

export interface SignalsOverview {
  obi_signals: number;
  volume_active_markets: number;
  large_trades_24h: number;
  active_markets: number;
}

// --- Phase 2 Whale/User Types ---

export interface TraderRanking {
  proxy_wallet: string;
  user_name: string;
  profile_image: string;
  rank: number;
  category: string;
  time_period: string;
  order_by: string;
  pnl: number;
  volume: number;
  verified_badge: number;
  x_username: string;
  snapshot_time: string;
}

export interface MarketHolder {
  condition_id: string;
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  outcome_index: number;
  amount: number;
  snapshot_time: string;
}

export interface WalletPosition {
  proxy_wallet: string;
  condition_id: string;
  outcome: string;
  size: number;
  avg_price: number;
  current_value: number;
  cur_price: number;
  cash_pnl: number;
  percent_pnl: number;
  realized_pnl: number;
  title: string;
  market_slug: string;
  updated_at: string;
}

export interface WalletActivity {
  proxy_wallet: string;
  condition_id: string;
  activity_type: string;
  side: string;
  outcome: string;
  size: number;
  usdc_size: number;
  price: number;
  transaction_hash: string;
  title: string;
  timestamp: string;
}

export interface TraderProfile {
  proxy_wallet: string;
  pseudonym: string;
  name: string;
  bio: string;
  profile_image: string;
  x_username: string;
  verified_badge: number;
  first_seen_at: string;
}

export interface WhaleActivityFeed {
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  condition_id: string;
  activity_type: string;
  side: string;
  outcome: string;
  size: number;
  usdc_size: number;
  price: number;
  title: string;
  timestamp: string;
}

export interface SmartMoneyPosition {
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  rank: number;
  condition_id: string;
  outcome: string;
  size: number;
  current_value: number;
  cash_pnl: number;
  percent_pnl: number;
  title: string;
}

export interface PositionConcentration {
  condition_id: string;
  question: string;
  total_holders: number;
  total_amount: number;
  top5_amount: number;
  top5_share: number;
  top_holder_wallet: string;
  top_holder_amount: number;
}

export interface WhalesOverview {
  tracked_wallets: number;
  whale_trades_24h: number;
  total_whale_positions: number;
  unique_markets_held: number;
}

// --- Phase 3 Analytics Types ---

export interface ArbitrageOpportunity {
  condition_id: string;
  event_slug: string;
  arb_type: string;            // 'sum_to_one' | 'related_market'
  expected_sum: number;
  actual_sum: number;
  spread: number;
  related_condition_ids: string[];
  description: string;
  status: string;               // 'open' | 'closed' | 'expired'
  detected_at: string;
  resolved_at: string;
  question?: string;            // Joined from markets
}

export interface WalletCluster {
  cluster_id: string;
  wallets: string[];
  size: number;
  similarity_score: number;
  timing_corr: number;
  market_overlap: number;
  direction_agreement: number;
  common_markets: string[];
  label: string;
  created_at: string;
}

export interface InsiderAlert {
  proxy_wallet: string;
  pseudonym: string;            // Joined from trader_profiles
  profile_image: string;        // Joined from trader_profiles
  score: number;                // 0-100
  freshness_score: number;
  win_rate_score: number;
  niche_score: number;
  size_score: number;
  timing_score: number;
  computed_at: string;
}

export interface CompositeSignal {
  condition_id: string;
  question: string;             // Joined from markets
  score: number;                // -100 to +100
  confidence: number;           // 0-1
  obi_score: number;
  volume_score: number;
  trade_bias_score: number;
  momentum_score: number;
  smart_money_score: number;
  concentration_score: number;
  arbitrage_flag: number;       // 0 or 1
  insider_activity: number;     // 0-100
  computed_at: string;
}

export interface AnalyticsOverview {
  open_arbitrages: number;
  wallet_clusters: number;
  insider_alerts: number;       // Wallets with score > 50
  markets_scored: number;       // Markets with composite signals
  avg_confidence: number;
}
