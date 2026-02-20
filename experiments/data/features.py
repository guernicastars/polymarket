"""Feature engineering for resolved Polymarket contracts.

Implements the 27-feature spec from RESEARCH.md across 5 categories:
  A. Market Structure (5) -- from markets table metadata
  B. Price Signals (6) -- from market_trades + markets pre-computed fields
  C. Trade Microstructure (8) -- from market_trades
  D. Volume & Activity Dynamics (4) -- from markets table + market_trades
  E. Wallet / Smart Money (4) -- from wallet_activity + insider_scores

IMPORTANT: market_prices table is broken (empty condition_id for all rows).
All price features are derived from market_trades or markets metadata.

Class imbalance note: Yes/No markets are 22/78% skewed toward No.
Over/Under markets are roughly balanced (49/51).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ======================================================================
# Category A: Market Structure (5 features)
# Available for ALL resolved markets from the markets table.
# ======================================================================

def extract_market_structure(market: dict) -> dict[str, float]:
    """Features 1-5: market metadata features.

    Parameters
    ----------
    market : dict
        Row from the markets table.
    """
    features: dict[str, float] = {}

    # 1. volume_total (log-scaled to reduce skew)
    features["volume_total"] = float(np.log1p(market.get("volume_total", 0)))

    # 2. liquidity (log-scaled; often 0 for closed markets)
    features["liquidity"] = float(np.log1p(market.get("liquidity", 0)))

    # 3. neg_risk (binary flag)
    features["neg_risk"] = float(market.get("neg_risk", 0))

    # 4. market_duration_days
    start = pd.Timestamp(market.get("start_date"))
    end = pd.Timestamp(market.get("end_date"))
    if pd.notna(start) and pd.notna(end) and end > start:
        features["market_duration_days"] = float(
            (end - start).total_seconds() / 86400
        )
    else:
        features["market_duration_days"] = np.nan

    # 5. num_outcomes
    outcomes = market.get("outcomes", [])
    features["num_outcomes"] = float(len(outcomes)) if outcomes else 2.0

    return features


# ======================================================================
# Category B: Price Signal Features (6 features)
# Derived from market_trades + markets pre-computed Gamma API fields.
# NOT from market_prices (broken table).
# ======================================================================

def extract_price_signals(
    market: dict,
    trades_df: pd.DataFrame,
) -> dict[str, float]:
    """Features 6-11: price signal features.

    Parameters
    ----------
    market : dict
        Row from the markets table (has one_day_price_change, etc.).
    trades_df : pd.DataFrame
        Trades with columns: price, size, side, timestamp.
        Assumed sorted by timestamp ascending.
    """
    features: dict[str, float] = {}

    # 6. last_price -- from last trade, or outcome_prices[0] as fallback
    if not trades_df.empty:
        features["last_price"] = float(trades_df["price"].iloc[-1])
    else:
        outcome_prices = market.get("outcome_prices", [])
        features["last_price"] = (
            float(outcome_prices[0]) if outcome_prices else np.nan
        )

    # 7. one_day_price_change -- from Gamma API (pre-computed on market)
    features["one_day_price_change"] = float(
        market.get("one_day_price_change", 0)
    )

    # 8. one_week_price_change -- from Gamma API
    features["one_week_price_change"] = float(
        market.get("one_week_price_change", 0)
    )

    if not trades_df.empty:
        prices = trades_df["price"].astype(float)
        n = len(prices)

        # 9. price_range -- max - min from trade prices (volatility proxy)
        features["price_range"] = float(prices.max() - prices.min())

        # 10. price_at_75pct_life -- price at 75% of observed trade span
        #     Captures how early the outcome was priced in.
        idx_75 = int(n * 0.75)
        features["price_at_75pct_life"] = float(
            prices.iloc[idx_75] if n > 1 else prices.iloc[0]
        )

        # 11. final_price_velocity -- rate of price change in the last 10%
        #     of observed trades
        last_10pct = max(int(n * 0.10), 2)
        tail = prices.iloc[-last_10pct:]
        if len(tail) >= 2:
            features["final_price_velocity"] = float(
                tail.iloc[-1] - tail.iloc[0]
            )
        else:
            features["final_price_velocity"] = 0.0
    else:
        features["price_range"] = np.nan
        features["price_at_75pct_life"] = np.nan
        features["final_price_velocity"] = np.nan

    return features


# ======================================================================
# Category C: Trade Microstructure (8 features)
# Derived from market_trades.
# ======================================================================

def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values."""
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    total = values.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * total) / (n * total))


_TRADE_NAN_FEATURES = {
    "trade_count": np.nan,
    "avg_trade_size": np.nan,
    "max_trade_size": np.nan,
    "buy_sell_ratio": np.nan,
    "buy_volume_ratio": np.nan,
    "trade_size_gini": np.nan,
    "trades_per_day": np.nan,
    "late_volume_ratio": np.nan,
}


def extract_trade_microstructure(
    trades_df: pd.DataFrame,
    market: dict,
) -> dict[str, float]:
    """Features 12-19: trade microstructure features.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trades with columns: price, size, side, timestamp.
    market : dict
        Market metadata (for market lifetime calculation).
    """
    if trades_df.empty:
        return dict(_TRADE_NAN_FEATURES)

    features: dict[str, float] = {}

    sizes = trades_df["size"].astype(float)
    prices = trades_df["price"].astype(float)
    timestamps = pd.to_datetime(trades_df["timestamp"])

    # USDC value per trade = price * size
    usdc_per_trade = prices * sizes

    # 12. trade_count (log-scaled)
    features["trade_count"] = float(np.log1p(len(trades_df)))

    # 13. avg_trade_size (USDC per trade)
    features["avg_trade_size"] = float(usdc_per_trade.mean())

    # 14. max_trade_size (USDC)
    features["max_trade_size"] = float(usdc_per_trade.max())

    # 15. buy_sell_ratio -- fraction of trades that are buys
    n_buys = (trades_df["side"] == "buy").sum()
    features["buy_sell_ratio"] = float(n_buys / len(trades_df))

    # 16. buy_volume_ratio -- fraction of volume (USDC-weighted) from buys
    buy_vol = usdc_per_trade[trades_df["side"] == "buy"].sum()
    total_vol = usdc_per_trade.sum()
    features["buy_volume_ratio"] = (
        float(buy_vol / total_vol) if total_vol > 0 else 0.5
    )

    # 17. trade_size_gini -- concentration of large traders
    features["trade_size_gini"] = _gini_coefficient(sizes.values)

    # 18. trades_per_day -- average daily trade frequency
    time_span_sec = (timestamps.max() - timestamps.min()).total_seconds()
    if time_span_sec > 0:
        days = time_span_sec / 86400
        features["trades_per_day"] = float(len(trades_df) / max(days, 1e-6))
    else:
        features["trades_per_day"] = float(len(trades_df))

    # 19. late_volume_ratio -- fraction of volume in the last 25% of
    #     market lifetime
    start = pd.Timestamp(market.get("start_date"))
    end = pd.Timestamp(market.get("end_date"))
    if pd.notna(start) and pd.notna(end) and end > start:
        cutoff_75 = start + (end - start) * 0.75
        late_mask = timestamps >= cutoff_75
        late_vol = usdc_per_trade[late_mask].sum()
        features["late_volume_ratio"] = (
            float(late_vol / total_vol) if total_vol > 0 else 0.0
        )
    else:
        # Fall back to last 25% of trades by index
        n = len(trades_df)
        cutoff_idx = int(n * 0.75)
        late_vol = usdc_per_trade.iloc[cutoff_idx:].sum()
        features["late_volume_ratio"] = (
            float(late_vol / total_vol) if total_vol > 0 else 0.0
        )

    return features


# ======================================================================
# Category D: Volume & Activity Dynamics (4 features)
# From markets table pre-computed Gamma API fields + trade data.
# ======================================================================

def extract_volume_dynamics(
    market: dict,
    category_median_volume: float,
) -> dict[str, float]:
    """Features 20-23: volume dynamics features.

    Parameters
    ----------
    market : dict
        Market metadata with volume_24h, volume_1wk fields.
    category_median_volume : float
        Median volume_total for this market's category (pre-computed).
    """
    features: dict[str, float] = {}

    # 20. volume_24h (log-scaled)
    features["volume_24h"] = float(np.log1p(market.get("volume_24h", 0)))

    # 21. volume_1wk (log-scaled)
    features["volume_1wk"] = float(np.log1p(market.get("volume_1wk", 0)))

    # 22. volume_acceleration -- volume_24h / (volume_1wk / 7) spike ratio
    v24 = market.get("volume_24h", 0)
    v1w = market.get("volume_1wk", 0)
    daily_avg = v1w / 7.0 if v1w > 0 else 0
    features["volume_acceleration"] = (
        float(v24 / daily_avg) if daily_avg > 0 else np.nan
    )

    # 23. volume_vs_category_median -- relative volume
    vol_total = market.get("volume_total", 0)
    if category_median_volume > 0:
        features["volume_vs_category_median"] = float(
            vol_total / category_median_volume
        )
    else:
        features["volume_vs_category_median"] = np.nan

    return features


# ======================================================================
# Category E: Wallet / Smart Money (4 features)
# From wallet_activity + insider_scores tables.
# Available for a subset (~2,334 markets with wallet_activity).
# ======================================================================

def extract_wallet_features(wallet_data: dict) -> dict[str, float]:
    """Features 24-27: wallet/smart money features.

    Parameters
    ----------
    wallet_data : dict
        Pre-aggregated wallet data for this market with keys:
        - unique_wallet_count: int
        - whale_buy_ratio: float
        - top_wallet_concentration: float
        - avg_insider_score: float
        All may be None/NaN if wallet data is unavailable.
    """
    return {
        "unique_wallet_count": float(
            wallet_data.get("unique_wallet_count", np.nan)
        ),
        "whale_buy_ratio": float(
            wallet_data.get("whale_buy_ratio", np.nan)
        ),
        "top_wallet_concentration": float(
            wallet_data.get("top_wallet_concentration", np.nan)
        ),
        "avg_insider_score": float(
            wallet_data.get("avg_insider_score", np.nan)
        ),
    }


# ======================================================================
# Aggregation: combine all feature groups for one market
# ======================================================================

def build_feature_vector(
    market: dict,
    trades_df: pd.DataFrame,
    category_median_volume: float,
    wallet_data: dict | None = None,
) -> dict[str, float]:
    """Build a single flat feature dict for one market (up to 27 features).

    NaN values are preserved; downstream code should impute or drop them.
    """
    features: dict[str, float] = {}

    # A: Market Structure (5)
    features.update(extract_market_structure(market))

    # B: Price Signals (6)
    features.update(extract_price_signals(market, trades_df))

    # C: Trade Microstructure (8)
    features.update(extract_trade_microstructure(trades_df, market))

    # D: Volume Dynamics (4)
    features.update(extract_volume_dynamics(market, category_median_volume))

    # E: Wallet / Smart Money (4) -- may be all NaN if unavailable
    features.update(extract_wallet_features(wallet_data or {}))

    return features
