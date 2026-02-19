"""Feature extraction from ClickHouse — builds the 12-feature temporal tensor.

Maps Anastasiia's 12 features to our existing ClickHouse tables:

  F1  log_returns          → market_prices (price changes)
  F2  high_low_spread      → ohlcv_1m or market_prices (high/low within step)
  F3  dist_from_ma         → market_prices (price - SMA_12)
  F4  bid_ask_spread       → orderbook_snapshots (best_ask - best_bid)
  F5  obi                  → orderbook_snapshots (bid_depth - ask_depth) / total
  F6  depth_ratio          → orderbook_snapshots (top5/top1 depth)
  F7  volume_delta         → market_trades (volume acceleration)
  F8  open_interest_change → market_holders (delta in total holdings)
  F9  sentiment_score      → news_sentiment_hourly.weighted_sentiment (primary) with
                              composite_signals.smart_money_score (fallback)
  F10 news_velocity        → news_articles article count (primary) with
                              market_trades large trade count (fallback)
  F11 inv_time_to_expiry   → markets.end_date
  F12 correlation_delta    → market_prices of sibling markets in same event
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .config import FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL templates for each feature
# ---------------------------------------------------------------------------

# F1-F3: Price-based features aggregated into 5-min bars
# Primary: market_prices (tick-level snapshots)
PRICE_FEATURES_SQL = """
SELECT
    toStartOfInterval(timestamp, INTERVAL {step} MINUTE) AS bar_time,
    argMin(price, timestamp) AS open_price,
    max(price) AS high_price,
    min(price) AS low_price,
    argMax(price, timestamp) AS close_price,
    avg(bid) AS avg_bid,
    avg(ask) AS avg_ask
FROM market_prices
WHERE condition_id = {{condition_id:String}}
  AND outcome = 'Yes'
  AND timestamp >= {{start:DateTime64(3)}}
  AND timestamp < {{end:DateTime64(3)}}
GROUP BY bar_time
ORDER BY bar_time
"""

# Fallback: derive OHLC from market_trades when market_prices is empty
PRICE_FROM_TRADES_SQL = """
SELECT
    toStartOfInterval(timestamp, INTERVAL {step} MINUTE) AS bar_time,
    argMin(price, timestamp) AS open_price,
    max(price) AS high_price,
    min(price) AS low_price,
    argMax(price, timestamp) AS close_price,
    0.0 AS avg_bid,
    0.0 AS avg_ask
FROM market_trades
WHERE condition_id = {{condition_id:String}}
  AND outcome = 'Yes'
  AND timestamp >= {{start:DateTime64(3)}}
  AND timestamp < {{end:DateTime64(3)}}
GROUP BY bar_time
ORDER BY bar_time
"""

# F4-F6: Orderbook features (latest snapshot per step)
ORDERBOOK_FEATURES_SQL = """
SELECT
    toStartOfInterval(snapshot_time, INTERVAL {step} MINUTE) AS bar_time,
    argMax(bid_prices, snapshot_time) AS bid_prices,
    argMax(bid_sizes, snapshot_time) AS bid_sizes,
    argMax(ask_prices, snapshot_time) AS ask_prices,
    argMax(ask_sizes, snapshot_time) AS ask_sizes
FROM orderbook_snapshots
WHERE condition_id = {{condition_id:String}}
  AND outcome = 'Yes'
  AND snapshot_time >= {{start:DateTime64(3)}}
  AND snapshot_time < {{end:DateTime64(3)}}
GROUP BY bar_time
ORDER BY bar_time
"""

# F7 + F10: Volume-based features
VOLUME_FEATURES_SQL = """
SELECT
    toStartOfInterval(timestamp, INTERVAL {step} MINUTE) AS bar_time,
    sum(size) AS total_volume,
    countIf(size * price >= 5000) AS large_trade_count
FROM market_trades
WHERE condition_id = {{condition_id:String}}
  AND timestamp >= {{start:DateTime64(3)}}
  AND timestamp < {{end:DateTime64(3)}}
GROUP BY bar_time
ORDER BY bar_time
"""

# F8: Open interest proxy — total holder amounts
HOLDER_DELTA_SQL = """
SELECT
    proxy_wallet,
    sum(amount) AS total_amount,
    max(snapshot_time) AS last_snap
FROM market_holders FINAL
WHERE condition_id = {{condition_id:String}}
GROUP BY proxy_wallet
"""

# F9: Sentiment — weighted sentiment from news data (primary), fallback to smart money score
SENTIMENT_SQL = """
SELECT
    smart_money_score,
    computed_at
FROM composite_signals FINAL
WHERE condition_id = {{condition_id:String}}
ORDER BY computed_at DESC
LIMIT 1
"""

NEWS_SENTIMENT_SQL = """
SELECT
    toStartOfInterval(hour, INTERVAL {step} MINUTE) AS bar_time,
    avg(weighted_sentiment) AS sentiment,
    sum(article_count) AS article_count,
    avg(news_velocity) AS velocity
FROM news_sentiment_hourly
WHERE settlement_id = {{settlement_id:String}}
  AND hour >= {{start:DateTime}}
  AND hour < {{end:DateTime}}
GROUP BY bar_time
ORDER BY bar_time
"""

# F10: News velocity — article count from news data (primary), fallback to large trade count
NEWS_VELOCITY_SQL = """
SELECT
    toStartOfInterval(published_at, INTERVAL {step} MINUTE) AS bar_time,
    count() AS article_count,
    avg(urgency) AS avg_urgency
FROM news_articles
WHERE has(markets_mentioned, {{condition_id:String}})
  AND published_at >= {{start:DateTime64(3)}}
  AND published_at < {{end:DateTime64(3)}}
GROUP BY bar_time
ORDER BY bar_time
"""

# F11: Time to expiry
EXPIRY_SQL = """
SELECT
    end_date
FROM markets FINAL
WHERE condition_id = {{condition_id:String}}
LIMIT 1
"""

# F12: Sibling market prices for correlation
SIBLING_PRICES_SQL = """
SELECT
    m2.condition_id AS sibling_id,
    toStartOfInterval(p.timestamp, INTERVAL {step} MINUTE) AS bar_time,
    argMax(p.price, p.timestamp) AS close_price
FROM market_prices p
JOIN (
    SELECT condition_id
    FROM markets FINAL
    WHERE event_slug = {{event_slug:String}}
      AND condition_id != {{condition_id:String}}
    LIMIT 1
) AS m2 ON p.condition_id = m2.condition_id
WHERE p.outcome = 'Yes'
  AND p.timestamp >= {{start:DateTime64(3)}}
  AND p.timestamp < {{end:DateTime64(3)}}
GROUP BY sibling_id, bar_time
ORDER BY bar_time
"""


class FeatureExtractor:
    """Extracts the 12 temporal features from ClickHouse for a single market.

    Returns a numpy array of shape (window_size, 12).
    Missing values are forward-filled then zero-filled.
    """

    def __init__(self, client, config: Optional[FeatureConfig] = None):
        """
        Args:
            client: clickhouse_connect Client instance
            config: feature extraction configuration
        """
        self.client = client
        self.cfg = config or FeatureConfig()

    def extract(
        self,
        condition_id: str,
        event_slug: str,
        end_time: Optional[datetime] = None,
        settlement_id: Optional[str] = None,
    ) -> np.ndarray:
        """Extract 12-feature window ending at end_time.

        Args:
            condition_id: Market condition ID
            event_slug: Event slug for sibling market correlation
            end_time: Window end time (defaults to now)
            settlement_id: Settlement ID for news sentiment enrichment (optional)

        Returns:
            np.ndarray of shape (window_size, 12), normalized.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        total_minutes = self.cfg.window_size * self.cfg.step_minutes
        start_time = end_time - timedelta(minutes=total_minutes)

        # Generate expected bar times
        bar_times = [
            start_time + timedelta(minutes=i * self.cfg.step_minutes)
            for i in range(self.cfg.window_size)
        ]

        features = np.zeros((self.cfg.window_size, self.cfg.n_features), dtype=np.float32)

        # --- Fetch raw data ---
        prices = self._fetch_prices(condition_id, start_time, end_time)
        orderbooks = self._fetch_orderbooks(condition_id, start_time, end_time)
        volumes = self._fetch_volumes(condition_id, start_time, end_time)
        sentiment = self._fetch_sentiment(condition_id)
        end_date = self._fetch_expiry(condition_id)
        sibling_prices = self._fetch_sibling_prices(
            condition_id, event_slug, start_time, end_time
        )
        news_sentiment = self._fetch_news_sentiment(settlement_id, start_time, end_time) if settlement_id else []
        news_velocity = self._fetch_news_velocity(condition_id, start_time, end_time)

        # --- Align to bar_times and compute features ---
        price_map = {row[0]: row[1:] for row in prices}  # bar_time → (o,h,l,c,bid,ask)
        ob_map = {row[0]: row[1:] for row in orderbooks}
        vol_map = {row[0]: row[1:] for row in volumes}
        sib_map = {row[1]: row[2] for row in sibling_prices}  # bar_time → close
        news_sent_map = {row[0]: row[1:] for row in news_sentiment}  # bar_time → (sentiment, article_count, velocity)
        news_vel_map = {row[0]: row[1:] for row in news_velocity}  # bar_time → (article_count, avg_urgency)

        prev_close = None
        close_history = []
        vol_history = []

        for i, bt in enumerate(bar_times):
            # Snap to nearest bar
            p = price_map.get(bt)
            ob = ob_map.get(bt)
            v = vol_map.get(bt)
            sib_close = sib_map.get(bt)

            # --- F1: Log returns ---
            if p is not None and prev_close is not None and prev_close > 0:
                features[i, 0] = np.log(p[3] / prev_close)
            if p is not None:
                prev_close = p[3]

            # --- F2: High-low spread ---
            if p is not None:
                features[i, 1] = p[1] - p[2]  # high - low

            # --- F3: Distance from 12-step moving average ---
            if p is not None:
                close_history.append(p[3])
                if len(close_history) >= 12:
                    ma = np.mean(close_history[-12:])
                    features[i, 2] = p[3] - ma

            # --- F4: Bid-ask spread ---
            if p is not None:
                features[i, 3] = (p[5] - p[4]) if (p[4] > 0 and p[5] > 0) else 0.0

            # --- F5: Order Book Imbalance ---
            if ob is not None:
                bid_sizes, ask_sizes = ob[1], ob[3]
                if bid_sizes and ask_sizes:
                    total_bid = sum(bid_sizes) if isinstance(bid_sizes, (list, tuple)) else 0
                    total_ask = sum(ask_sizes) if isinstance(ask_sizes, (list, tuple)) else 0
                    denom = total_bid + total_ask
                    features[i, 4] = (total_bid - total_ask) / denom if denom > 0 else 0.0

            # --- F6: Depth ratio (top5 / top1) ---
            if ob is not None:
                bid_sizes = ob[1]
                if bid_sizes and isinstance(bid_sizes, (list, tuple)) and len(bid_sizes) >= 1:
                    top1 = bid_sizes[0] if bid_sizes[0] > 0 else 1.0
                    top5 = sum(bid_sizes[:5]) if len(bid_sizes) >= 5 else sum(bid_sizes)
                    features[i, 5] = top5 / top1

            # --- F7: Volume delta ---
            if v is not None:
                vol_history.append(v[0])
                if len(vol_history) >= 3:
                    recent_avg = np.mean(vol_history[-3:-1]) if len(vol_history) > 1 else 1.0
                    if recent_avg > 0:
                        features[i, 6] = (v[0] / recent_avg) - 1.0

            # --- F8: Open interest change (placeholder — use holder delta) ---
            # Full OI tracking requires per-step holder snapshots; use 0 until available
            features[i, 7] = 0.0

            # --- F9: Sentiment (news sentiment preferred, fallback to smart money score) ---
            news_sent = news_sent_map.get(bt)
            if news_sent is not None:
                # news_sent[0] is weighted_sentiment, assumed to be in [-1, 1] range
                features[i, 8] = news_sent[0]
            elif sentiment is not None:
                features[i, 8] = sentiment / 100.0  # normalize to [-1, 1]

            # --- F10: News velocity (article count preferred, fallback to large trade count) ---
            news_vel = news_vel_map.get(bt)
            if news_vel is not None:
                # news_vel[0] is article_count
                features[i, 9] = float(news_vel[0])
            elif v is not None:
                features[i, 9] = v[1]  # large_trade_count as fallback

            # --- F11: Inverse time to expiry ---
            if end_date is not None:
                days_left = max((end_date - bt).total_seconds() / 86400, 1.0)
                features[i, 10] = 1.0 / days_left

            # --- F12: Correlation delta (sibling market returns) ---
            if sib_close is not None and i > 0:
                prev_sib = sib_map.get(bar_times[i - 1])
                if prev_sib and prev_sib > 0:
                    features[i, 11] = np.log(sib_close / prev_sib)

        # --- Normalize: z-score per feature column ---
        for col in range(self.cfg.n_features):
            std = np.std(features[:, col])
            if std > 1e-8:
                features[:, col] = (features[:, col] - np.mean(features[:, col])) / std

        return features

    # ------------------------------------------------------------------
    # Private fetch helpers
    # ------------------------------------------------------------------

    def _fetch_prices(self, cid, start, end):
        sql = PRICE_FEATURES_SQL.format(step=self.cfg.step_minutes)
        try:
            rows = self.client.query(
                sql,
                parameters={"condition_id": cid, "start": start, "end": end},
            ).result_rows
            if rows:
                return rows
            # Fallback: derive price bars from market_trades
            sql_trades = PRICE_FROM_TRADES_SQL.format(step=self.cfg.step_minutes)
            rows = self.client.query(
                sql_trades,
                parameters={"condition_id": cid, "start": start, "end": end},
            ).result_rows
            if rows:
                logger.debug("Using trade-derived prices for %s (%d bars)", cid, len(rows))
            return rows
        except Exception as e:
            logger.warning("price fetch failed for %s: %s", cid, e)
            return []

    def _fetch_orderbooks(self, cid, start, end):
        sql = ORDERBOOK_FEATURES_SQL.format(step=self.cfg.step_minutes)
        try:
            return self.client.query(
                sql,
                parameters={"condition_id": cid, "start": start, "end": end},
            ).result_rows
        except Exception as e:
            logger.warning("orderbook fetch failed for %s: %s", cid, e)
            return []

    def _fetch_volumes(self, cid, start, end):
        sql = VOLUME_FEATURES_SQL.format(step=self.cfg.step_minutes)
        try:
            return self.client.query(
                sql,
                parameters={"condition_id": cid, "start": start, "end": end},
            ).result_rows
        except Exception as e:
            logger.warning("volume fetch failed for %s: %s", cid, e)
            return []

    def _fetch_sentiment(self, cid) -> Optional[float]:
        try:
            rows = self.client.query(
                SENTIMENT_SQL, parameters={"condition_id": cid}
            ).result_rows
            return rows[0][0] if rows else None
        except Exception:
            return None

    def _fetch_expiry(self, cid) -> Optional[datetime]:
        try:
            rows = self.client.query(
                EXPIRY_SQL, parameters={"condition_id": cid}
            ).result_rows
            return rows[0][0] if rows else None
        except Exception:
            return None

    def _fetch_sibling_prices(self, cid, event_slug, start, end):
        if not event_slug:
            return []
        sql = SIBLING_PRICES_SQL.format(step=self.cfg.step_minutes)
        try:
            return self.client.query(
                sql,
                parameters={
                    "condition_id": cid,
                    "event_slug": event_slug,
                    "start": start,
                    "end": end,
                },
            ).result_rows
        except Exception as e:
            logger.warning("sibling fetch failed for %s: %s", cid, e)
            return []

    def _fetch_news_sentiment(self, settlement_id, start, end):
        """Fetch news sentiment aggregated into 5-minute bars by settlement_id."""
        if not settlement_id:
            return []
        sql = NEWS_SENTIMENT_SQL.format(step=self.cfg.step_minutes)
        try:
            return self.client.query(
                sql,
                parameters={
                    "settlement_id": settlement_id,
                    "start": start,
                    "end": end,
                },
            ).result_rows
        except Exception as e:
            logger.warning("news sentiment fetch failed for settlement %s: %s", settlement_id, e)
            return []

    def _fetch_news_velocity(self, cid, start, end):
        """Fetch news article counts aggregated into 5-minute bars by condition_id."""
        sql = NEWS_VELOCITY_SQL.format(step=self.cfg.step_minutes)
        try:
            return self.client.query(
                sql,
                parameters={
                    "condition_id": cid,
                    "start": start,
                    "end": end,
                },
            ).result_rows
        except Exception as e:
            logger.warning("news velocity fetch failed for %s: %s", cid, e)
            return []
