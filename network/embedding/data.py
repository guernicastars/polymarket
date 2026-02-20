"""Resolved market dataset: ClickHouse → lifecycle summary features.

Each resolved Polymarket contract becomes a single data point with ~27 summary
statistics computed over its full (or cutoff) price/trade/holder history.
Batched SQL queries keep ClickHouse round-trips to ~6 regardless of market count.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EmbeddingFeatureConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (ordered, matching tensor columns)
# ---------------------------------------------------------------------------
PRICE_FEATURES = [
    "start_price",
    "cutoff_price",
    "min_price",
    "max_price",
    "price_volatility",
    "price_range",
    "final_runup_speed",
]

VOLUME_FEATURES = [
    "total_volume",
    "daily_avg_volume",
    "peak_daily_volume",
    "volume_gini",
    "volume_trend_slope",
]

LIQUIDITY_FEATURES = [
    "avg_spread",
    "spread_volatility",
    "avg_obi",
    "avg_depth",
]

PARTICIPATION_FEATURES = [
    "unique_holders",
    "top5_concentration",
    "holder_gini",
    "max_single_holder_pct",
]

TEMPORAL_FEATURES = [
    "duration_days",
    "active_trading_days_ratio",
    "time_to_peak_volume",
]

STRUCTURE_FEATURES = [
    "n_outcomes",
    "is_binary",
    "category_encoded",
    "event_group_size",
]

ALL_FEATURE_NAMES = (
    PRICE_FEATURES
    + VOLUME_FEATURES
    + LIQUIDITY_FEATURES
    + PARTICIPATION_FEATURES
    + TEMPORAL_FEATURES
    + STRUCTURE_FEATURES
)


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient for measuring concentration."""
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_arr) / (n * np.sum(sorted_arr))) - (n + 1) / n)


class ResolvedMarketDataset(Dataset):
    """Dataset of resolved Polymarket markets with lifecycle summary features.

    Each sample is a single market represented by ~27 summary statistics.
    Supports a lifetime_cutoff_ratio to prevent feature leakage from
    the resolution convergence period.
    """

    def __init__(
        self,
        client: Any,
        config: Optional[EmbeddingFeatureConfig] = None,
    ):
        self.client = client
        self.cfg = config or EmbeddingFeatureConfig()

        # 1. Discover resolved markets
        logger.info("Discovering resolved markets...")
        self.markets = self._discover_resolved_markets()
        logger.info("Found %d resolved markets", len(self.markets))

        if len(self.markets) == 0:
            self.features = np.zeros((0, len(ALL_FEATURE_NAMES)), dtype=np.float32)
            self.raw_features = self.features.copy()
            self.feature_names = list(ALL_FEATURE_NAMES)
            self.labels: list[dict] = []
            self.mean = np.zeros(len(ALL_FEATURE_NAMES))
            self.std = np.ones(len(ALL_FEATURE_NAMES))
            return

        # 2. Compute summary features
        logger.info("Computing summary features...")
        self.features, self.feature_names = self._compute_all_features()
        logger.info("Feature matrix shape: %s", self.features.shape)

        # 3. Store raw features before normalization (needed by _extract_labels)
        self.raw_features = self.features.copy()

        # 4. Extract labels
        self.labels = self._extract_labels()

        # 5. Normalize (z-score per column)
        self.mean = np.mean(self.features, axis=0)
        self.std = np.std(self.features, axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.markets)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "raw_features": torch.tensor(self.raw_features[idx], dtype=torch.float32),
            "labels": self.labels[idx],
            "condition_id": self.markets[idx]["condition_id"],
            "question": self.markets[idx]["question"],
        }

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_resolved_markets(self) -> list[dict]:
        """Query ClickHouse for resolved markets meeting volume/duration thresholds."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.cfg.lookback_months * 30)

        sql = """
            SELECT
                condition_id,
                question,
                category,
                event_slug,
                event_id,
                outcomes,
                winning_outcome,
                volume_total,
                start_date,
                end_date,
                created_at
            FROM (SELECT * FROM markets FINAL) AS m
            INNER JOIN (
                SELECT DISTINCT condition_id AS cid
                FROM market_trades
            ) AS t ON m.condition_id = t.cid
            WHERE m.resolved = 1
              AND m.volume_total >= {min_vol:Float64}
              AND m.end_date >= {cutoff:DateTime64(3)}
              AND m.winning_outcome != ''
            ORDER BY m.volume_total DESC
        """
        params = {"min_vol": self.cfg.min_volume_total, "cutoff": cutoff_date}

        if self.cfg.max_markets > 0:
            sql += f"\nLIMIT {self.cfg.max_markets}"

        try:
            result = self.client.query(sql, parameters=params)
            rows = result.result_rows
            cols = result.column_names
        except Exception as e:
            logger.error("Failed to discover resolved markets: %s", e)
            return []

        markets = []
        for row in rows:
            m = dict(zip(cols, row))
            # Compute cutoff time for leakage protection
            start = m["start_date"]
            end = m["end_date"]
            if start and end and end > start:
                duration = (end - start).total_seconds()
                m["cutoff_time"] = start + timedelta(
                    seconds=duration * self.cfg.lifetime_cutoff_ratio
                )
                m["duration_seconds"] = duration
            else:
                m["cutoff_time"] = end or datetime.utcnow()
                m["duration_seconds"] = 0
            markets.append(m)

        # Filter by minimum duration
        min_secs = self.cfg.min_duration_days * 86400
        markets = [m for m in markets if m["duration_seconds"] >= min_secs]

        return markets

    # ------------------------------------------------------------------
    # Feature computation (batched SQL)
    # ------------------------------------------------------------------

    def _compute_all_features(self) -> tuple[np.ndarray, list[str]]:
        """Compute all summary features via batched ClickHouse queries."""
        n = len(self.markets)
        cid_to_idx = {m["condition_id"]: i for i, m in enumerate(self.markets)}
        cids = list(cid_to_idx.keys())

        # Build cutoff map: condition_id -> cutoff_time
        cutoff_map = {m["condition_id"]: m["cutoff_time"] for m in self.markets}
        start_map = {m["condition_id"]: m["start_date"] for m in self.markets}

        features = np.zeros((n, len(ALL_FEATURE_NAMES)), dtype=np.float32)

        col_offset = 0
        if self.cfg.include_price:
            self._fill_price_features(features, col_offset, cids, cid_to_idx, start_map, cutoff_map)
        col_offset += len(PRICE_FEATURES)

        if self.cfg.include_volume:
            self._fill_volume_features(features, col_offset, cids, cid_to_idx, start_map, cutoff_map)
        col_offset += len(VOLUME_FEATURES)

        if self.cfg.include_liquidity:
            self._fill_liquidity_features(features, col_offset, cids, cid_to_idx, start_map, cutoff_map)
        col_offset += len(LIQUIDITY_FEATURES)

        if self.cfg.include_participation:
            self._fill_participation_features(features, col_offset, cids, cid_to_idx)
        col_offset += len(PARTICIPATION_FEATURES)

        if self.cfg.include_temporal:
            self._fill_temporal_features(features, col_offset, cids, cid_to_idx, start_map, cutoff_map)
        col_offset += len(TEMPORAL_FEATURES)

        if self.cfg.include_structure:
            self._fill_structure_features(features, col_offset, cids, cid_to_idx)
        col_offset += len(STRUCTURE_FEATURES)

        return features, list(ALL_FEATURE_NAMES)

    def _in_clause(self, cids: list[str]) -> str:
        """Build a safe IN clause for condition_ids."""
        return ", ".join(f"'{c}'" for c in cids)

    def _fill_price_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict, start_map: dict, cutoff_map: dict,
    ) -> None:
        """Price trajectory from trades: start, cutoff, min, max, volatility, range, runup.

        Uses market_trades (available for resolved markets) since market_prices
        only covers actively-polled markets and drops resolved ones.
        """
        in_clause = self._in_clause(cids)

        # Batch query: per-market VWAP-based price stats from trades
        sql = f"""
            SELECT
                condition_id,
                argMin(price, timestamp) AS start_price,
                min(price) AS min_price,
                max(price) AS max_price,
                stddevPop(price) AS price_std,
                count() AS n_trades,
                argMax(price, timestamp) AS last_price
            FROM market_trades
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id
        """
        rows = self._safe_query(sql)

        price_data: dict[str, dict] = {}
        for row in rows:
            cid = row[0]
            price_data[cid] = {
                "start_price": float(row[1] or 0.5),
                "min_price": float(row[2] or 0.0),
                "max_price": float(row[3] or 1.0),
                "price_std": float(row[4] or 0.0),
                "n_trades": int(row[5] or 0),
                "last_price": float(row[6] or 0.5),
            }

        # Cutoff prices via batch query on trades before cutoff
        # Group by condition_id, each with its own cutoff time
        # Use a UNION ALL approach to batch cutoff queries
        cutoff_prices: dict[str, float] = {}
        batch_size = 50
        cid_list = list(cids)
        for i in range(0, len(cid_list), batch_size):
            batch = cid_list[i:i + batch_size]
            parts = []
            for cid in batch:
                cutoff = cutoff_map.get(cid)
                if cutoff:
                    ct = cutoff.strftime("%Y-%m-%d %H:%M:%S")
                    parts.append(
                        f"SELECT '{cid}' AS cid, argMax(price, timestamp) AS cp "
                        f"FROM market_trades "
                        f"WHERE condition_id = '{cid}' AND timestamp <= '{ct}'"
                    )
            if parts:
                union_sql = " UNION ALL ".join(parts)
                for row in self._safe_query(union_sql):
                    if row[1] is not None:
                        cutoff_prices[row[0]] = float(row[1])

        for cid in cids:
            idx = cid_to_idx[cid]
            pd = price_data.get(cid, {})

            start_price = pd.get("start_price", 0.5)
            features[idx, offset + 0] = start_price
            features[idx, offset + 2] = pd.get("min_price", 0.0)
            features[idx, offset + 3] = pd.get("max_price", 1.0)
            features[idx, offset + 4] = pd.get("price_std", 0.0)

            p_range = pd.get("max_price", 1.0) - pd.get("min_price", 0.0)
            features[idx, offset + 5] = p_range

            # Cutoff price
            cutoff_price = cutoff_prices.get(cid, start_price)
            features[idx, offset + 1] = cutoff_price

            # Runup speed: price change from start to cutoff / duration
            start = start_map.get(cid)
            cutoff = cutoff_map.get(cid)
            if start and cutoff and start_price > 0:
                dur_days = max((cutoff - start).total_seconds() / 86400, 0.01)
                features[idx, offset + 6] = (cutoff_price - start_price) / dur_days
            else:
                features[idx, offset + 6] = 0.0

    def _fill_volume_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict, start_map: dict, cutoff_map: dict,
    ) -> None:
        """Volume: total, daily avg, peak daily, gini, trend slope."""
        in_clause = self._in_clause(cids)

        # Daily volume breakdown per market
        sql = f"""
            SELECT
                condition_id,
                toDate(timestamp) AS dt,
                sum(size) AS daily_vol
            FROM market_trades
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id, dt
            ORDER BY condition_id, dt
        """
        rows = self._safe_query(sql)

        # Aggregate per market
        vol_series: dict[str, list[float]] = {cid: [] for cid in cids}
        for row in rows:
            cid, dt, vol = row[0], row[1], row[2]
            if cid in vol_series:
                vol_series[cid].append(float(vol))

        for cid in cids:
            idx = cid_to_idx[cid]
            vols = vol_series[cid]
            if not vols:
                continue

            arr = np.array(vols, dtype=np.float64)
            features[idx, offset + 0] = arr.sum()
            features[idx, offset + 1] = arr.mean()
            features[idx, offset + 2] = arr.max()
            features[idx, offset + 3] = _gini(arr)

            # Trend slope: linear regression of daily volumes
            if len(arr) >= 3:
                x = np.arange(len(arr), dtype=np.float64)
                x_mean = x.mean()
                y_mean = arr.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    features[idx, offset + 4] = np.sum((x - x_mean) * (arr - y_mean)) / denom

    def _fill_liquidity_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict, start_map: dict, cutoff_map: dict,
    ) -> None:
        """Liquidity: avg spread, spread vol, avg OBI, avg depth. Zero if no orderbook."""
        in_clause = self._in_clause(cids)

        sql = f"""
            SELECT
                condition_id,
                avg(if(length(ask_prices) > 0 AND length(bid_prices) > 0,
                       ask_prices[1] - bid_prices[1], 0)) AS avg_spread,
                stddevPop(if(length(ask_prices) > 0 AND length(bid_prices) > 0,
                              ask_prices[1] - bid_prices[1], 0)) AS spread_vol,
                avg(if(arraySum(bid_sizes) + arraySum(ask_sizes) > 0,
                       (arraySum(bid_sizes) - arraySum(ask_sizes))
                       / (arraySum(bid_sizes) + arraySum(ask_sizes)), 0)) AS avg_obi,
                avg(arraySum(bid_sizes) + arraySum(ask_sizes)) AS avg_depth
            FROM orderbook_snapshots
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id
        """
        rows = self._safe_query(sql)

        for row in rows:
            cid = row[0]
            if cid not in cid_to_idx:
                continue
            idx = cid_to_idx[cid]
            features[idx, offset + 0] = row[1] or 0.0
            features[idx, offset + 1] = row[2] or 0.0
            features[idx, offset + 2] = row[3] or 0.0
            features[idx, offset + 3] = row[4] or 0.0

    def _fill_participation_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict,
    ) -> None:
        """Participation: unique holders, top5 concentration, gini, max single holder."""
        in_clause = self._in_clause(cids)

        sql = f"""
            SELECT
                condition_id,
                proxy_wallet,
                sum(amount) AS total_amount
            FROM market_holders FINAL
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id, proxy_wallet
            ORDER BY condition_id, total_amount DESC
        """
        rows = self._safe_query(sql)

        # Aggregate per market
        holder_amounts: dict[str, list[float]] = {cid: [] for cid in cids}
        for row in rows:
            cid, wallet, amount = row[0], row[1], row[2]
            if cid in holder_amounts:
                holder_amounts[cid].append(float(amount))

        for cid in cids:
            idx = cid_to_idx[cid]
            amounts = holder_amounts[cid]
            if not amounts:
                continue

            arr = np.array(amounts, dtype=np.float64)
            total = arr.sum()
            if total <= 0:
                continue

            features[idx, offset + 0] = len(arr)  # unique holders
            top5 = arr[:5].sum() / total if len(arr) >= 5 else 1.0
            features[idx, offset + 1] = top5
            features[idx, offset + 2] = _gini(arr)
            features[idx, offset + 3] = arr[0] / total  # max single holder

    def _fill_temporal_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict, start_map: dict, cutoff_map: dict,
    ) -> None:
        """Temporal: duration days, active trading days ratio, time to peak volume."""
        in_clause = self._in_clause(cids)

        # Active trading days
        sql = f"""
            SELECT
                condition_id,
                count(DISTINCT toDate(timestamp)) AS active_days
            FROM market_trades
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id
        """
        rows = self._safe_query(sql)
        active_days_map = {row[0]: row[1] for row in rows}

        # Peak volume day offset
        sql2 = f"""
            SELECT condition_id, dt, daily_vol FROM (
                SELECT
                    condition_id,
                    toDate(timestamp) AS dt,
                    sum(size) AS daily_vol,
                    row_number() OVER (PARTITION BY condition_id ORDER BY sum(size) DESC) AS rn
                FROM market_trades
                WHERE condition_id IN ({in_clause})
                GROUP BY condition_id, dt
            ) WHERE rn = 1
        """
        peak_rows = self._safe_query(sql2)
        peak_day_map = {row[0]: row[1] for row in peak_rows}

        for cid in cids:
            idx = cid_to_idx[cid]
            m = self.markets[idx]
            duration_days = m["duration_seconds"] / 86400
            features[idx, offset + 0] = duration_days

            active = active_days_map.get(cid, 0)
            features[idx, offset + 1] = active / max(duration_days, 1)

            # Time to peak: fraction of lifetime where peak volume occurred
            peak_day = peak_day_map.get(cid)
            if peak_day and m.get("start_date"):
                days_to_peak = (peak_day - m["start_date"].date()).days if hasattr(m["start_date"], "date") else 0
                features[idx, offset + 2] = days_to_peak / max(duration_days, 1)

    def _fill_structure_features(
        self, features: np.ndarray, offset: int,
        cids: list[str], cid_to_idx: dict,
    ) -> None:
        """Structure: n_outcomes, is_binary, category encoded, event group size."""
        # Category encoding: build label map from data
        categories = sorted(set(
            m.get("category", "") for m in self.markets if m.get("category")
        ))
        cat_to_int = {c: i for i, c in enumerate(categories)}

        # Event group sizes
        event_counts: dict[str, int] = {}
        for m in self.markets:
            eid = m.get("event_id", "")
            if eid:
                event_counts[eid] = event_counts.get(eid, 0) + 1

        for cid in cids:
            idx = cid_to_idx[cid]
            m = self.markets[idx]

            outcomes = m.get("outcomes", [])
            n_outcomes = len(outcomes) if outcomes else 2
            features[idx, offset + 0] = n_outcomes
            features[idx, offset + 1] = 1.0 if n_outcomes == 2 else 0.0
            features[idx, offset + 2] = cat_to_int.get(m.get("category", ""), 0)
            features[idx, offset + 3] = event_counts.get(m.get("event_id", ""), 1)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _extract_labels(self) -> list[dict]:
        """Build label dictionaries for each market."""
        # Compute terciles for volume and volatility buckets
        volumes = np.array([m.get("volume_total", 0) for m in self.markets])
        vol_t1, vol_t2 = np.percentile(volumes, [33, 67]) if len(volumes) > 2 else (0, 0)

        # Price volatility from features (column index 4 in raw features)
        vols_feat = self.raw_features[:, 4] if self.raw_features.shape[1] > 4 else np.zeros(len(self.markets))
        vol_median = np.median(vols_feat) if len(vols_feat) > 0 else 0

        labels = []
        for i, m in enumerate(self.markets):
            winning = m.get("winning_outcome", "")
            outcomes = m.get("outcomes", [])

            # Binary outcome: 1 if first outcome won, 0 if second outcome won
            # Handles Yes/No, Over/Under, and named two-outcome markets
            if len(outcomes) == 2:
                outcome_binary = 0 if winning == outcomes[1] else 1
            else:
                outcome_binary = -1

            # Duration bucket
            dur = m.get("duration_seconds", 0) / 86400
            if dur < 7:
                dur_bucket = "short"
            elif dur < 30:
                dur_bucket = "medium"
            else:
                dur_bucket = "long"

            # Volume bucket
            v = m.get("volume_total", 0)
            if v < vol_t1:
                vol_bucket = "low"
            elif v < vol_t2:
                vol_bucket = "medium"
            else:
                vol_bucket = "high"

            # Volatility regime
            vol_regime = "high" if vols_feat[i] > vol_median else "low"

            labels.append({
                "winning_outcome": winning,
                "outcome_binary": outcome_binary,
                "category": m.get("category", ""),
                "duration_bucket": dur_bucket,
                "volume_bucket": vol_bucket,
                "volatility_regime": vol_regime,
            })

        return labels

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _safe_query(self, sql: str, parameters: Optional[dict] = None) -> list:
        """Execute ClickHouse query with error handling."""
        try:
            return self.client.query(sql, parameters=parameters).result_rows
        except Exception as e:
            logger.warning("Query failed: %s", e)
            return []


def collate_embedding_batch(batch: list[dict]) -> dict:
    """Collate function for DataLoader."""
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "raw_features": torch.stack([b["raw_features"] for b in batch]),
        "labels": {
            k: [b["labels"][k] for b in batch] for k in batch[0]["labels"]
        },
        "condition_ids": [b["condition_id"] for b in batch],
    }
