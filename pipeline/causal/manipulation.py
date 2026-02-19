"""Market manipulation detection for prediction markets.

Identifies suspicious market activity by combining multiple signals:

1. **Wash trading**: same-address buy/sell patterns, round trade sizes,
   high volume with no price impact.
2. **Spoofing**: large orders placed and quickly withdrawn, asymmetric
   orderbook patterns before price moves.
3. **Causal anomalies**: price movements that deviate from what a causal
   model predicts -- residuals exceeding a threshold.
4. **Composite risk scoring**: aggregate all signals into a single 0-1
   manipulation risk score.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class ManipulationDetector:
    """Detect potential market manipulation in prediction markets.

    Queries trades, orderbook snapshots, and prices from ClickHouse to
    identify wash trading, spoofing, and causal anomalies.

    Parameters
    ----------
    host : str, optional
        ClickHouse host. Defaults to config value.
    port : int, optional
        ClickHouse port. Defaults to config value.
    user : str, optional
        ClickHouse user. Defaults to config value.
    password : str, optional
        ClickHouse password. Defaults to config value.
    database : str, optional
        ClickHouse database. Defaults to config value.
    """

    def __init__(
        self,
        host: str = CLICKHOUSE_HOST,
        port: int = CLICKHOUSE_PORT,
        user: str = CLICKHOUSE_USER,
        password: str = CLICKHOUSE_PASSWORD,
        database: str = CLICKHOUSE_DATABASE,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._client: Client | None = None

    def _get_client(self) -> Client:
        """Lazily create and return a ClickHouse client."""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self._host,
                port=self._port,
                username=self._user,
                password=self._password,
                database=self._database,
                secure=True,
            )
        return self._client

    def detect_wash_trading(
        self,
        token_id: str,
        window_hours: int = 24,
    ) -> dict[str, Any]:
        """Detect potential wash trading for a given token.

        Wash trading signals:
        1. **Self-trading**: wallet activity showing buys and sells of the
           same token within a short window (from wallet_activity table).
        2. **Round trade sizes**: suspiciously round amounts (e.g., 100, 500,
           1000) that suggest algorithmic manipulation.
        3. **Volume without impact**: high trade volume with minimal price
           change, suggesting trades that cancel each other out.

        Parameters
        ----------
        token_id : str
            Token ID to analyze.
        window_hours : int
            Lookback window in hours (default 24).

        Returns
        -------
        dict
            Keys:
            - ``self_trade_wallets`` (list[str]): wallets that both bought
              and sold within the window.
            - ``self_trade_count`` (int): number of such wallets.
            - ``round_size_ratio`` (float): fraction of trades with round sizes.
            - ``volume_impact_ratio`` (float): total volume / abs(price change).
              High values suggest wash trading.
            - ``total_trades`` (int): number of trades in the window.
            - ``wash_trading_score`` (float): composite score 0-1.
            - ``signals`` (list[str]): human-readable signal descriptions.
        """
        client = self._get_client()

        # 1. Detect self-trading wallets from wallet_activity
        self_trade_result = client.query(f"""
            SELECT
                proxy_wallet,
                countIf(side = 'BUY') AS buy_count,
                countIf(side = 'SELL') AS sell_count,
                sumIf(usdc_size, side = 'BUY') AS buy_volume,
                sumIf(usdc_size, side = 'SELL') AS sell_volume
            FROM wallet_activity
            WHERE asset = '{token_id}'
              AND timestamp >= now() - INTERVAL {window_hours} HOUR
              AND activity_type = 'TRADE'
            GROUP BY proxy_wallet
            HAVING buy_count > 0 AND sell_count > 0
        """)

        self_trade_wallets = []
        for row in self_trade_result.result_rows:
            wallet = str(row[0])
            buy_vol = float(row[3])
            sell_vol = float(row[4])
            # Flag if buy and sell volumes are suspiciously close
            if min(buy_vol, sell_vol) > 0:
                ratio = min(buy_vol, sell_vol) / max(buy_vol, sell_vol)
                if ratio > 0.7:  # within 30% of each other
                    self_trade_wallets.append(wallet)

        # 2. Round trade sizes
        trades_result = client.query(f"""
            SELECT
                size,
                price,
                side
            FROM market_trades
            WHERE token_id = '{token_id}'
              AND timestamp >= now() - INTERVAL {window_hours} HOUR
        """)

        total_trades = len(trades_result.result_rows)
        round_count = 0
        sizes = []
        for row in trades_result.result_rows:
            size = float(row[0])
            sizes.append(size)
            # Check if size is "suspiciously round" (multiple of 10, 50, 100, etc.)
            if size > 0 and (size % 100 == 0 or size % 50 == 0 or size % 10 == 0):
                round_count += 1

        round_ratio = round_count / total_trades if total_trades > 0 else 0.0

        # 3. Volume without price impact
        volume_impact_result = client.query(f"""
            SELECT
                sum(size * price) AS total_volume,
                max(price) - min(price) AS price_range,
                first_value(price) AS first_price,
                last_value(price) AS last_price
            FROM market_trades
            WHERE token_id = '{token_id}'
              AND timestamp >= now() - INTERVAL {window_hours} HOUR
        """)

        total_volume = 0.0
        price_change = 0.0
        if volume_impact_result.result_rows:
            row = volume_impact_result.result_rows[0]
            total_volume = float(row[0]) if row[0] else 0.0
            price_change = abs(float(row[3]) - float(row[2])) if row[2] and row[3] else 0.0

        # Volume/impact ratio: high = suspicious
        volume_impact_ratio = total_volume / max(price_change, 0.001)

        # Composite wash trading score
        signals: list[str] = []
        score = 0.0

        # Self-trading contributes up to 0.4
        if self_trade_wallets:
            self_trade_score = min(len(self_trade_wallets) / 5.0, 1.0) * 0.4
            score += self_trade_score
            signals.append(
                f"{len(self_trade_wallets)} wallet(s) with matching buy/sell activity"
            )

        # Round sizes contribute up to 0.2
        if round_ratio > 0.3:
            round_score = min((round_ratio - 0.3) / 0.4, 1.0) * 0.2
            score += round_score
            signals.append(
                f"{round_ratio:.0%} of trades have round sizes"
            )

        # Volume without impact contributes up to 0.4
        # Normalize: consider > 10000 volume per 0.01 price change as high
        if total_volume > 0 and price_change < 0.01:
            vi_score = min(total_volume / 50000, 1.0) * 0.4
            score += vi_score
            signals.append(
                f"${total_volume:,.0f} volume with only {price_change:.4f} price change"
            )

        logger.info(
            "wash_trading_analysis",
            extra={
                "token": token_id,
                "self_traders": len(self_trade_wallets),
                "total_trades": total_trades,
                "score": round(score, 4),
            },
        )

        return {
            "self_trade_wallets": self_trade_wallets,
            "self_trade_count": len(self_trade_wallets),
            "round_size_ratio": round(round_ratio, 4),
            "volume_impact_ratio": round(volume_impact_ratio, 2),
            "total_trades": total_trades,
            "wash_trading_score": round(min(score, 1.0), 4),
            "signals": signals,
        }

    def detect_spoofing(
        self,
        token_id: str,
        window_hours: int = 6,
    ) -> dict[str, Any]:
        """Detect potential spoofing behavior from orderbook snapshots.

        Spoofing signals:
        1. **Disappearing liquidity**: large orderbook depth that vanishes
           between consecutive snapshots.
        2. **Asymmetric book before moves**: one-sided depth buildup before
           a price move in the opposite direction.

        Parameters
        ----------
        token_id : str
            Token ID to analyze.
        window_hours : int
            Lookback window in hours (default 6).

        Returns
        -------
        dict
            Keys:
            - ``depth_volatility_bid`` (float): std/mean of bid depth.
            - ``depth_volatility_ask`` (float): std/mean of ask depth.
            - ``asymmetry_events`` (int): count of asymmetric-book-before-move events.
            - ``spoofing_score`` (float): composite score 0-1.
            - ``signals`` (list[str]): human-readable signal descriptions.
        """
        client = self._get_client()

        # Fetch orderbook snapshots
        ob_result = client.query(f"""
            SELECT
                snapshot_time,
                bid_sizes,
                ask_sizes,
                bid_prices,
                ask_prices
            FROM orderbook_snapshots
            WHERE token_id = '{token_id}'
              AND snapshot_time >= now() - INTERVAL {window_hours} HOUR
            ORDER BY snapshot_time
        """)

        if len(ob_result.result_rows) < 3:
            return {
                "depth_volatility_bid": 0.0,
                "depth_volatility_ask": 0.0,
                "asymmetry_events": 0,
                "spoofing_score": 0.0,
                "signals": [],
            }

        bid_depths = []
        ask_depths = []
        bid_ask_ratios = []

        for row in ob_result.result_rows:
            bid_sizes = row[1] if row[1] else []
            ask_sizes = row[2] if row[2] else []
            bid_total = sum(float(s) for s in bid_sizes) if bid_sizes else 0.0
            ask_total = sum(float(s) for s in ask_sizes) if ask_sizes else 0.0
            bid_depths.append(bid_total)
            ask_depths.append(ask_total)
            total = bid_total + ask_total
            if total > 0:
                bid_ask_ratios.append(bid_total / total)
            else:
                bid_ask_ratios.append(0.5)

        bid_depths_arr = np.array(bid_depths)
        ask_depths_arr = np.array(ask_depths)

        # 1. Depth volatility (high = orders appearing/disappearing)
        bid_mean = np.mean(bid_depths_arr)
        ask_mean = np.mean(ask_depths_arr)
        bid_vol = float(np.std(bid_depths_arr) / max(bid_mean, 1e-6))
        ask_vol = float(np.std(ask_depths_arr) / max(ask_mean, 1e-6))

        # 2. Asymmetric book before price moves
        # Fetch price data for the same period
        price_result = client.query(f"""
            SELECT
                toStartOfInterval(timestamp, INTERVAL 1 MINUTE) AS bucket,
                avg(price) AS avg_price
            FROM market_prices
            WHERE token_id = '{token_id}'
              AND timestamp >= now() - INTERVAL {window_hours} HOUR
            GROUP BY bucket
            ORDER BY bucket
        """)

        prices = [float(row[1]) for row in price_result.result_rows]
        price_changes = np.diff(prices) if len(prices) > 1 else np.array([])

        # Count asymmetry events: large bid/ask imbalance followed by move
        # in the opposite direction (classic spoofing pattern)
        asymmetry_events = 0
        n_ratios = len(bid_ask_ratios)
        for i in range(min(n_ratios - 1, len(price_changes))):
            ratio = bid_ask_ratios[i]
            if i < len(price_changes):
                change = price_changes[i] if i < len(price_changes) else 0
                # Heavy bids (ratio > 0.7) followed by price DOWN
                if ratio > 0.7 and change < -0.005:
                    asymmetry_events += 1
                # Heavy asks (ratio < 0.3) followed by price UP
                elif ratio < 0.3 and change > 0.005:
                    asymmetry_events += 1

        # Composite spoofing score
        signals: list[str] = []
        score = 0.0

        # Depth volatility contributes up to 0.4
        max_vol = max(bid_vol, ask_vol)
        if max_vol > 0.5:
            vol_score = min((max_vol - 0.5) / 1.0, 1.0) * 0.4
            score += vol_score
            signals.append(
                f"High depth volatility: bid CV={bid_vol:.2f}, ask CV={ask_vol:.2f}"
            )

        # Asymmetry events contribute up to 0.6
        if asymmetry_events > 0:
            asym_score = min(asymmetry_events / 5.0, 1.0) * 0.6
            score += asym_score
            signals.append(
                f"{asymmetry_events} asymmetric book events before opposing price moves"
            )

        logger.info(
            "spoofing_analysis",
            extra={
                "token": token_id,
                "bid_vol": round(bid_vol, 4),
                "ask_vol": round(ask_vol, 4),
                "asymmetry_events": asymmetry_events,
                "score": round(score, 4),
            },
        )

        return {
            "depth_volatility_bid": round(bid_vol, 4),
            "depth_volatility_ask": round(ask_vol, 4),
            "asymmetry_events": asymmetry_events,
            "spoofing_score": round(min(score, 1.0), 4),
            "signals": signals,
        }

    def detect_causal_anomalies(
        self,
        token_id: str,
        causal_model: dict[str, Any],
        threshold_sigma: float = 3.0,
        window_hours: int = 24,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Detect price movements not explained by a causal model.

        Given a causal model (adjacency matrix + token IDs), fits a linear
        model of the target token on its causal parents, then flags
        time points where the residual exceeds ``threshold_sigma``
        standard deviations.

        Parameters
        ----------
        token_id : str
            Target token to check for anomalies.
        causal_model : dict
            Output of CrossMarketAnalyzer's DAG discovery methods.
            Must contain ``token_ids`` and ``adjacency``.
        threshold_sigma : float
            Number of standard deviations for anomaly detection.
        window_hours : int
            Lookback window in hours.
        resample_minutes : int
            Time bin width for price resampling.

        Returns
        -------
        dict
            Keys:
            - ``anomaly_count`` (int): number of anomalous time points.
            - ``anomalies`` (list[dict]): each with index, residual, sigma_score.
            - ``residual_std`` (float): standard deviation of residuals.
            - ``r2`` (float): R-squared of the causal model fit.
            - ``parent_tokens`` (list[str]): causal parents used.
        """
        # Find causal parents of token_id in the model
        token_ids = causal_model.get("token_ids", [])
        adjacency = causal_model.get("adjacency", [])

        if token_id not in token_ids:
            logger.warning("causal_anomaly_token_not_in_model", extra={"token": token_id})
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": [],
            }

        target_idx = token_ids.index(token_id)
        parent_indices = [
            i for i in range(len(token_ids))
            if i != target_idx and adjacency[i][target_idx] == 1
        ]
        parent_tokens = [token_ids[i] for i in parent_indices]

        if not parent_tokens:
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": [],
            }

        # Fetch price data
        client = self._get_client()
        all_tokens = [token_id] + parent_tokens
        token_list = ", ".join(f"'{t}'" for t in all_tokens)

        query = f"""
            SELECT
                token_id,
                toStartOfInterval(timestamp, INTERVAL {resample_minutes} MINUTE) AS bucket,
                avg(price) AS avg_price
            FROM market_prices
            WHERE token_id IN ({token_list})
              AND timestamp >= now() - INTERVAL {window_hours} HOUR
            GROUP BY token_id, bucket
            ORDER BY token_id, bucket
        """

        result = client.query(query)

        raw: dict[str, dict[datetime, float]] = {}
        for row in result.result_rows:
            tid = str(row[0])
            bucket = row[1]
            price = float(row[2])
            raw.setdefault(tid, {})[bucket] = price

        # Common time axis
        all_buckets: set[datetime] = set()
        for buckets in raw.values():
            all_buckets.update(buckets.keys())
        time_axis = sorted(all_buckets)

        if len(time_axis) < 10:
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": parent_tokens,
            }

        # Build aligned arrays
        def _align(tid: str) -> np.ndarray | None:
            if tid not in raw:
                return None
            prices = []
            last = None
            for t in time_axis:
                if t in raw[tid]:
                    last = raw[tid][t]
                prices.append(last if last is not None else np.nan)
            arr = np.array(prices, dtype=np.float64)
            mask = np.isnan(arr)
            if mask.all():
                return None
            fv = int(np.argmax(~mask))
            arr[:fv] = arr[fv]
            rem = np.isnan(arr)
            if rem.any():
                idx = np.where(~rem, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
            return np.diff(arr)  # Use changes for stationarity

        y = _align(token_id)
        if y is None or len(y) < 5:
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": parent_tokens,
            }

        X_cols = []
        used_parents = []
        for pt in parent_tokens:
            x = _align(pt)
            if x is not None and len(x) == len(y):
                X_cols.append(x)
                used_parents.append(pt)

        if not X_cols:
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": parent_tokens,
            }

        X = np.column_stack(X_cols)
        X_aug = np.column_stack([np.ones(len(y)), X])

        try:
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return {
                "anomaly_count": 0,
                "anomalies": [],
                "residual_std": 0.0,
                "r2": 0.0,
                "parent_tokens": used_parents,
            }

        y_hat = X_aug @ beta
        residuals = y - y_hat
        residual_std = float(np.std(residuals))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Flag anomalies
        anomalies: list[dict[str, Any]] = []
        if residual_std > 1e-10:
            for i, res in enumerate(residuals):
                sigma_score = abs(float(res)) / residual_std
                if sigma_score >= threshold_sigma:
                    anomalies.append({
                        "index": i,
                        "residual": round(float(res), 6),
                        "sigma_score": round(sigma_score, 4),
                        "direction": "up" if res > 0 else "down",
                    })

        logger.info(
            "causal_anomaly_detection",
            extra={
                "token": token_id,
                "parents": len(used_parents),
                "anomalies": len(anomalies),
                "r2": round(r2, 4),
            },
        )

        return {
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "residual_std": round(residual_std, 6),
            "r2": round(r2, 4),
            "parent_tokens": used_parents,
        }

    def score_manipulation_risk(
        self,
        token_id: str,
        causal_model: dict[str, Any] | None = None,
        window_hours: int = 24,
    ) -> dict[str, Any]:
        """Compute an aggregate manipulation risk score for a token.

        Combines wash trading, spoofing, and (optionally) causal anomaly
        signals into a single 0-1 risk score.

        Parameters
        ----------
        token_id : str
            Token ID to score.
        causal_model : dict, optional
            If provided, also checks for causal anomalies.
        window_hours : int
            Lookback window in hours.

        Returns
        -------
        dict
            Keys:
            - ``risk_score`` (float): aggregate risk 0-1.
            - ``wash_trading`` (dict): wash trading sub-analysis.
            - ``spoofing`` (dict): spoofing sub-analysis.
            - ``causal_anomalies`` (dict | None): causal anomaly sub-analysis.
            - ``all_signals`` (list[str]): combined human-readable signals.
        """
        wash = self.detect_wash_trading(token_id, window_hours)
        spoof = self.detect_spoofing(token_id, min(window_hours, 6))

        causal = None
        causal_score = 0.0
        if causal_model is not None:
            causal = self.detect_causal_anomalies(
                token_id, causal_model, window_hours=window_hours
            )
            if causal["anomaly_count"] > 0:
                causal_score = min(causal["anomaly_count"] / 5.0, 1.0)

        # Weighted aggregate
        # Wash trading: 0.4, Spoofing: 0.4, Causal anomalies: 0.2
        if causal_model is not None:
            risk_score = (
                0.4 * wash["wash_trading_score"]
                + 0.4 * spoof["spoofing_score"]
                + 0.2 * causal_score
            )
        else:
            # Without causal model, reweight wash and spoof
            risk_score = (
                0.5 * wash["wash_trading_score"]
                + 0.5 * spoof["spoofing_score"]
            )

        risk_score = min(risk_score, 1.0)

        all_signals = wash["signals"] + spoof["signals"]
        if causal and causal["anomaly_count"] > 0:
            all_signals.append(
                f"{causal['anomaly_count']} unexplained price movements (causal anomalies)"
            )

        logger.info(
            "manipulation_risk_scored",
            extra={
                "token": token_id,
                "risk_score": round(risk_score, 4),
                "wash_score": wash["wash_trading_score"],
                "spoof_score": spoof["spoofing_score"],
                "causal_score": round(causal_score, 4),
            },
        )

        return {
            "risk_score": round(risk_score, 4),
            "wash_trading": wash,
            "spoofing": spoof,
            "causal_anomalies": causal,
            "all_signals": all_signals,
        }
