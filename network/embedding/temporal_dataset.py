"""Temporal market dataset: ClickHouse → variable-length hourly feature sequences.

Each market becomes a time series of shape (n_bars, 12) where n_bars varies
by market lifetime. The 12 features are adapted from the GNN feature extractor
(network/gnn/features.py) but aggregated to 1-hour bars instead of 5-minute.

Supports two modes:
  - Pre-training: ALL markets (active + resolved), no labels needed, lower volume filter
  - Fine-tuning: resolved markets only, with labels for probe evaluation

SQL queries are batched in chunks of 2000 condition IDs to stay within
ClickHouse's max_query_size (~256KB). Each chunk runs 4 queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TransformerConfig

logger = logging.getLogger(__name__)

# Feature names (ordered, matching tensor columns — same as GNN's 12 features)
TEMPORAL_FEATURE_NAMES = [
    "log_returns",           # F1
    "high_low_spread",       # F2
    "dist_from_ma",          # F3
    "bid_ask_spread",        # F4
    "obi",                   # F5
    "depth_ratio",           # F6
    "volume_delta",          # F7
    "open_interest_change",  # F8
    "sentiment_score",       # F9
    "news_velocity",         # F10
    "inv_time_to_expiry",    # F11
    "correlation_delta",     # F12
]


class TemporalMarketDataset(Dataset):
    """Variable-length hourly time series per market for transformer training.

    Each sample is a dict with:
      - features: (n_bars, 12) float32 tensor — z-score normalized per feature
      - padding_mask: (n_bars,) bool tensor — True for padded positions
      - relative_positions: (n_patches,) float32 — position fraction [0, 1]
      - condition_id: str
      - labels: dict (only for resolved markets, empty dict otherwise)
    """

    def __init__(
        self,
        client: Any,
        cfg: Optional[TransformerConfig] = None,
        mode: str = "pretrain",
    ):
        """
        Args:
            client: clickhouse_connect Client instance
            cfg: transformer configuration
            mode: 'pretrain' (all markets) or 'finetune' (resolved only)
        """
        self.client = client
        self.cfg = cfg or TransformerConfig()
        self.mode = mode

        # 1. Discover markets
        logger.info("Discovering markets (mode=%s)...", mode)
        self.markets = self._discover_markets()
        logger.info("Found %d markets", len(self.markets))

        if not self.markets:
            self.sequences: list[np.ndarray] = []
            self.labels: list[dict] = []
            self._global_mean = np.zeros(12)
            self._global_std = np.ones(12)
            return

        # 2. Batch-fetch hourly feature data
        logger.info("Fetching hourly bar data...")
        raw_sequences = self._fetch_all_bars()
        logger.info("Built %d sequences", len(raw_sequences))

        # 3. Compute per-feature z-score stats across all bars
        all_bars = np.concatenate([s for s in raw_sequences if len(s) > 0], axis=0)
        self._global_mean = np.mean(all_bars, axis=0) if len(all_bars) > 0 else np.zeros(12)
        self._global_std = np.std(all_bars, axis=0) if len(all_bars) > 0 else np.ones(12)
        self._global_std[self._global_std < 1e-8] = 1.0

        # 4. Normalize and store
        self.sequences = []
        for seq in raw_sequences:
            if len(seq) >= self.cfg.min_bars:
                normed = (seq - self._global_mean) / self._global_std
                self.sequences.append(normed.astype(np.float32))

        # 5. Extract labels for resolved markets
        self.labels = self._extract_labels() if mode == "finetune" else [{}] * len(self.sequences)

        # Filter out markets that didn't have enough bars
        valid_cids = {self.markets[i]["condition_id"] for i in range(len(self.sequences))}
        self.markets = [m for m in self.markets if m["condition_id"] in valid_cids]

        logger.info("Final dataset: %d markets, features shape per market varies", len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        n_bars = len(seq)
        n_patches = n_bars // self.cfg.patch_size

        # Relative positions for each patch: fraction through the market's life
        rel_pos = np.linspace(0, 1, n_patches, dtype=np.float32) if n_patches > 0 else np.array([], dtype=np.float32)

        return {
            "features": torch.tensor(seq, dtype=torch.float32),
            "padding_mask": torch.zeros(n_bars, dtype=torch.bool),
            "relative_positions": torch.tensor(rel_pos, dtype=torch.float32),
            "condition_id": self.markets[idx]["condition_id"],
            "labels": self.labels[idx] if idx < len(self.labels) else {},
        }

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def _discover_markets(self) -> list[dict]:
        """Query ClickHouse for markets matching mode criteria."""
        min_vol = self.cfg.min_volume_pretrain if self.mode == "pretrain" else self.cfg.min_volume_finetune

        if self.mode == "pretrain":
            # All markets with trade data and sufficient volume
            sql = """
                SELECT
                    m.condition_id,
                    m.question,
                    m.category,
                    m.event_slug,
                    m.outcomes,
                    m.winning_outcome,
                    m.volume_total,
                    m.start_date,
                    m.end_date,
                    m.resolved
                FROM (SELECT * FROM markets FINAL) AS m
                INNER JOIN (SELECT DISTINCT condition_id AS cid FROM market_trades) AS t
                    ON m.condition_id = t.cid
                WHERE m.volume_total >= {min_vol:Float64}
                ORDER BY m.volume_total DESC
            """
        else:
            # Resolved markets only
            sql = """
                SELECT
                    m.condition_id,
                    m.question,
                    m.category,
                    m.event_slug,
                    m.outcomes,
                    m.winning_outcome,
                    m.volume_total,
                    m.start_date,
                    m.end_date,
                    m.resolved
                FROM (SELECT * FROM markets FINAL) AS m
                INNER JOIN (SELECT DISTINCT condition_id AS cid FROM market_trades) AS t
                    ON m.condition_id = t.cid
                WHERE m.resolved = 1
                  AND m.winning_outcome != ''
                  AND m.volume_total >= {min_vol:Float64}
                ORDER BY m.volume_total DESC
            """

        try:
            result = self.client.query(sql, parameters={"min_vol": min_vol})
            rows = result.result_rows
            cols = result.column_names
        except Exception as e:
            logger.error("Market discovery failed: %s", e)
            return []

        markets = []
        for row in rows:
            m = dict(zip(cols, row))
            markets.append(m)

        return markets

    # ------------------------------------------------------------------
    # Batch feature extraction (hourly bars)
    # ------------------------------------------------------------------

    def _fetch_all_bars(self) -> list[np.ndarray]:
        """Fetch hourly OHLCV + orderbook features for all markets in batched queries.

        Returns list of (n_bars, 12) arrays, one per market (same order as self.markets).
        """
        cids = [m["condition_id"] for m in self.markets]
        cid_to_idx = {cid: i for i, cid in enumerate(cids)}
        n = len(cids)

        # Pre-allocate: dict of condition_id → list of (bar_time, 12-feature vector)
        market_bars: dict[str, dict[datetime, list[float]]] = {cid: {} for cid in cids}
        expiry_map: dict[str, datetime] = {}

        # Batch condition IDs to avoid exceeding ClickHouse max_query_size (~256KB).
        # Each hex condition_id is ~66 chars; 2000 * 69 bytes ≈ 138KB, well within limit.
        batch_size = 2000
        for batch_start in range(0, n, batch_size):
            batch_cids = cids[batch_start:batch_start + batch_size]
            in_clause = ", ".join(f"'{cid}'" for cid in batch_cids)
            logger.info(
                "Fetching bars for markets %d-%d of %d...",
                batch_start, min(batch_start + batch_size, n), n,
            )

            self._fetch_price_bars(in_clause, market_bars)
            self._fetch_orderbook_bars(in_clause, market_bars)
            self._fetch_volume_bars(in_clause, market_bars)
            expiry_map.update(self._fetch_expiries(in_clause))

        # --- Compute derived features and assemble arrays ---
        sequences = []
        for i, cid in enumerate(cids):
            bars = market_bars[cid]
            if not bars:
                sequences.append(np.zeros((0, 12), dtype=np.float32))
                continue

            # Sort by time
            sorted_times = sorted(bars.keys())

            # Cap at max_bars
            if len(sorted_times) > self.cfg.max_bars:
                sorted_times = sorted_times[-self.cfg.max_bars:]

            arr = np.zeros((len(sorted_times), 12), dtype=np.float32)
            prev_close = None
            close_history: list[float] = []
            vol_history: list[float] = []

            end_date = expiry_map.get(cid)

            for t_idx, bt in enumerate(sorted_times):
                raw = bars[bt]  # list of partial feature values

                close = raw[0] if len(raw) > 0 and raw[0] != 0 else None
                high = raw[1] if len(raw) > 1 else 0.0
                low = raw[2] if len(raw) > 2 else 0.0
                bid = raw[3] if len(raw) > 3 else 0.0
                ask = raw[4] if len(raw) > 4 else 0.0
                ob_bid_depth = raw[5] if len(raw) > 5 else 0.0
                ob_ask_depth = raw[6] if len(raw) > 6 else 0.0
                ob_bid_top1 = raw[7] if len(raw) > 7 else 0.0
                ob_bid_top5 = raw[8] if len(raw) > 8 else 0.0
                volume = raw[9] if len(raw) > 9 else 0.0
                large_count = raw[10] if len(raw) > 10 else 0.0

                # F1: Log returns
                if close is not None and prev_close is not None and prev_close > 0:
                    arr[t_idx, 0] = np.log(close / prev_close)
                if close is not None:
                    prev_close = close

                # F2: High-low spread
                arr[t_idx, 1] = high - low

                # F3: Distance from 12-bar MA
                if close is not None:
                    close_history.append(close)
                    if len(close_history) >= 12:
                        ma = np.mean(close_history[-12:])
                        arr[t_idx, 2] = close - ma

                # F4: Bid-ask spread
                if bid > 0 and ask > 0:
                    arr[t_idx, 3] = ask - bid

                # F5: OBI
                total_depth = ob_bid_depth + ob_ask_depth
                if total_depth > 0:
                    arr[t_idx, 4] = (ob_bid_depth - ob_ask_depth) / total_depth

                # F6: Depth ratio
                if ob_bid_top1 > 0:
                    arr[t_idx, 5] = ob_bid_top5 / ob_bid_top1

                # F7: Volume delta
                vol_history.append(volume)
                if len(vol_history) >= 3:
                    recent_avg = np.mean(vol_history[-3:-1])
                    if recent_avg > 0:
                        arr[t_idx, 6] = (volume / recent_avg) - 1.0

                # F8: OI change — placeholder (0.0)
                arr[t_idx, 7] = 0.0

                # F9: Sentiment — placeholder (0.0)
                arr[t_idx, 8] = 0.0

                # F10: News velocity proxy (large trade count)
                arr[t_idx, 9] = large_count

                # F11: Inverse time to expiry
                if end_date is not None:
                    days_left = max((end_date - bt).total_seconds() / 86400, 1.0)
                    arr[t_idx, 10] = 1.0 / days_left

                # F12: Correlation delta — placeholder (0.0, requires sibling data)
                arr[t_idx, 11] = 0.0

            sequences.append(arr)

        return sequences

    def _fetch_price_bars(self, in_clause: str, market_bars: dict) -> None:
        """Fetch hourly OHLC from market_trades."""
        sql = f"""
            SELECT
                condition_id,
                toStartOfHour(timestamp) AS bar_time,
                argMax(price, timestamp) AS close_price,
                max(price) AS high_price,
                min(price) AS low_price
            FROM market_trades
            WHERE condition_id IN ({in_clause})
              AND outcome = 'Yes'
            GROUP BY condition_id, bar_time
            ORDER BY condition_id, bar_time
        """
        for row in self._safe_query(sql):
            cid, bt, close, high, low = row[0], row[1], row[2], row[3], row[4]
            if cid in market_bars:
                if bt not in market_bars[cid]:
                    market_bars[cid][bt] = [0.0] * 11
                bars = market_bars[cid][bt]
                bars[0] = float(close or 0)
                bars[1] = float(high or 0)
                bars[2] = float(low or 0)

    def _fetch_orderbook_bars(self, in_clause: str, market_bars: dict) -> None:
        """Fetch hourly orderbook features from orderbook_snapshots."""
        sql = f"""
            SELECT
                condition_id,
                toStartOfHour(snapshot_time) AS bar_time,
                avg(if(length(bid_prices) > 0, bid_prices[1], 0)) AS avg_bid,
                avg(if(length(ask_prices) > 0, ask_prices[1], 0)) AS avg_ask,
                avg(arraySum(bid_sizes)) AS avg_bid_depth,
                avg(arraySum(ask_sizes)) AS avg_ask_depth,
                avg(if(length(bid_sizes) > 0, bid_sizes[1], 0)) AS avg_bid_top1,
                avg(if(length(bid_sizes) >= 5,
                    bid_sizes[1] + bid_sizes[2] + bid_sizes[3] + bid_sizes[4] + bid_sizes[5],
                    arraySum(bid_sizes))) AS avg_bid_top5
            FROM orderbook_snapshots
            WHERE condition_id IN ({in_clause})
              AND outcome = 'Yes'
            GROUP BY condition_id, bar_time
            ORDER BY condition_id, bar_time
        """
        for row in self._safe_query(sql):
            cid, bt = row[0], row[1]
            if cid in market_bars:
                if bt not in market_bars[cid]:
                    market_bars[cid][bt] = [0.0] * 11
                bars = market_bars[cid][bt]
                bars[3] = float(row[2] or 0)   # bid
                bars[4] = float(row[3] or 0)   # ask
                bars[5] = float(row[4] or 0)   # bid depth
                bars[6] = float(row[5] or 0)   # ask depth
                bars[7] = float(row[6] or 0)   # bid top1
                bars[8] = float(row[7] or 0)   # bid top5

    def _fetch_volume_bars(self, in_clause: str, market_bars: dict) -> None:
        """Fetch hourly volume + large trade count from market_trades."""
        sql = f"""
            SELECT
                condition_id,
                toStartOfHour(timestamp) AS bar_time,
                sum(size) AS total_volume,
                countIf(size * price >= 5000) AS large_trade_count
            FROM market_trades
            WHERE condition_id IN ({in_clause})
            GROUP BY condition_id, bar_time
            ORDER BY condition_id, bar_time
        """
        for row in self._safe_query(sql):
            cid, bt = row[0], row[1]
            if cid in market_bars:
                if bt not in market_bars[cid]:
                    market_bars[cid][bt] = [0.0] * 11
                bars = market_bars[cid][bt]
                bars[9] = float(row[2] or 0)   # volume
                bars[10] = float(row[3] or 0)  # large trade count

    def _fetch_expiries(self, in_clause: str) -> dict[str, datetime]:
        """Fetch end_date per market for F11 computation."""
        sql = f"""
            SELECT condition_id, end_date
            FROM markets FINAL
            WHERE condition_id IN ({in_clause})
              AND end_date < '2090-01-01'
        """
        expiry_map = {}
        for row in self._safe_query(sql):
            if row[1] is not None:
                expiry_map[row[0]] = row[1]
        return expiry_map

    # ------------------------------------------------------------------
    # Labels (for fine-tuning mode)
    # ------------------------------------------------------------------

    def _extract_labels(self) -> list[dict]:
        """Build label dicts for resolved markets (same schema as ResolvedMarketDataset)."""
        # Volume terciles
        volumes = np.array([m.get("volume_total", 0) for m in self.markets])
        vol_t1, vol_t2 = np.percentile(volumes, [33, 67]) if len(volumes) > 2 else (0, 0)

        labels = []
        for i, m in enumerate(self.markets):
            if i >= len(self.sequences):
                break

            winning = m.get("winning_outcome", "")
            outcomes = m.get("outcomes", [])

            # Binary outcome
            if isinstance(outcomes, list) and len(outcomes) == 2:
                outcome_binary = 0 if winning == outcomes[1] else 1
            else:
                outcome_binary = -1

            # Duration bucket
            start = m.get("start_date")
            end = m.get("end_date")
            if start and end and end > start:
                dur_days = (end - start).total_seconds() / 86400
            else:
                dur_days = len(self.sequences[i]) / 24  # approximate from bar count

            if dur_days < 7:
                dur_bucket = "short"
            elif dur_days < 30:
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

            # Volatility regime from price std
            seq = self.sequences[i]
            price_std = np.std(seq[:, 0]) if len(seq) > 0 else 0  # log returns std
            vol_regime = "high" if price_std > np.median(
                [np.std(s[:, 0]) for s in self.sequences if len(s) > 0]
            ) else "low"

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
        try:
            return self.client.query(sql, parameters=parameters).result_rows
        except Exception as e:
            logger.warning("Query failed: %s", e)
            return []

    @property
    def feature_names(self) -> list[str]:
        return list(TEMPORAL_FEATURE_NAMES)

    @property
    def normalization_params(self) -> tuple[np.ndarray, np.ndarray]:
        return self._global_mean, self._global_std


def collate_temporal_batch(batch: list[dict]) -> dict:
    """Collate variable-length sequences into a padded batch.

    Pads all sequences to the longest in the batch.
    Returns:
      - features: (B, max_bars, 12)
      - padding_mask: (B, max_bars) — True for padding positions
      - relative_positions: (B, max_patches)
      - condition_ids: list of str
      - labels: dict of lists
    """
    max_bars = max(b["features"].shape[0] for b in batch)
    n_features = batch[0]["features"].shape[1]
    ps = 24  # patch_size

    # Compute max_patches directly from batch items — don't estimate from bar count,
    # because n_bars // n_patches gives wrong ps when n_bars isn't a multiple of 24.
    max_patches = max(b["relative_positions"].shape[0] for b in batch)

    # Pad bars to exactly max_patches * ps. Trailing bars beyond the last complete
    # patch boundary are clipped — PatchEmbedding truncates to n_patches * ps anyway,
    # so those bars would be unused and would create phantom patches with no positions.
    max_bars_padded = max_patches * ps

    B = len(batch)
    features = torch.zeros(B, max_bars_padded, n_features)
    padding_mask = torch.ones(B, max_bars_padded, dtype=torch.bool)  # True = padding
    relative_positions = torch.zeros(B, max_patches)

    for i, b in enumerate(batch):
        n = min(b["features"].shape[0], max_bars_padded)
        features[i, :n, :] = b["features"][:n]
        padding_mask[i, :n] = False
        n_p = b["relative_positions"].shape[0]
        if n_p > 0:
            relative_positions[i, :n_p] = b["relative_positions"]

    # Collate labels
    has_labels = any(b["labels"] for b in batch)
    if has_labels:
        label_keys = [k for k in batch[0]["labels"].keys()] if batch[0]["labels"] else []
        labels_collated = {k: [b["labels"].get(k, None) for b in batch] for k in label_keys}
    else:
        labels_collated = {}

    return {
        "features": features,
        "padding_mask": padding_mask,
        "relative_positions": relative_positions,
        "condition_ids": [b["condition_id"] for b in batch],
        "labels": labels_collated,
    }
