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
