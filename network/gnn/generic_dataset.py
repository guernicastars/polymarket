"""Generic market dataset: any set of condition_ids → temporal graph tensors.

Unlike DonbasTemporalDataset, this doesn't require geographic settlement
data. It takes a list of Polymarket condition IDs and an adjacency matrix
(built by MarketGraphBuilder or pre-computed) and produces the same
(N, W, F) tensor format the GNN-TCN model expects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GNNConfig
from .features import FeatureExtractor

logger = logging.getLogger(__name__)


class GenericMarketDataset(Dataset):
    """Temporal graph dataset for arbitrary Polymarket markets.

    Each sample is a (N_nodes × window_size × 12_features) tensor with
    a (n_targets,) label vector. Compatible with the same collator and
    model as DonbasTemporalDataset.
    """

    def __init__(
        self,
        client: Any,
        condition_ids: list[str],
        adjacency: np.ndarray,
        config: Optional[GNNConfig] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        stride_minutes: int = 30,
        target_ids: Optional[list[str]] = None,
    ):
        """
        Args:
            client: clickhouse_connect Client
            condition_ids: Market condition IDs (graph nodes).
            adjacency: Pre-built (N, N) adjacency matrix.
            config: GNN config.
            start_date: Data range start.
            end_date: Data range end.
            stride_minutes: Sliding window stride.
            target_ids: Subset of condition_ids to predict. If None, all are targets.
        """
        self.client = client
        self.cfg = config or GNNConfig()
        self.stride = stride_minutes
        self.extractor = FeatureExtractor(client, self.cfg.features)

        # Node mapping (sorted for deterministic ordering)
        self.node_ids = sorted(condition_ids)
        self.node_to_idx = {cid: i for i, cid in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)

        # Adjacency
        assert adjacency.shape == (self.n_nodes, self.n_nodes), (
            f"Adjacency shape {adjacency.shape} doesn't match {self.n_nodes} nodes"
        )
        self.adj = adjacency.astype(np.float32)

        # Targets: predict all markets or a subset
        if target_ids is not None:
            self.target_ids = [t for t in target_ids if t in self.node_to_idx]
        else:
            self.target_ids = list(self.node_ids)  # predict all
        self.target_indices = [self.node_to_idx[t] for t in self.target_ids]

        # Update model config with actual target count
        self.cfg.model.n_targets = len(self.target_ids)

        # Fetch market metadata (event_slug, category for feature F12)
        self.market_info = self._fetch_market_info()

        # Build sample timestamps
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        window_minutes = self.cfg.features.window_size * self.cfg.features.step_minutes
        self.sample_times: list[datetime] = []
        t = start_date + timedelta(minutes=window_minutes)
        while t <= end_date:
            self.sample_times.append(t)
            t += timedelta(minutes=self.stride)

        logger.info(
            "GenericMarketDataset: %d nodes, %d targets, %d samples, stride=%dmin",
            self.n_nodes, len(self.target_ids), len(self.sample_times), self.stride,
        )

    def __len__(self) -> int:
        return len(self.sample_times)

    def __getitem__(self, idx: int) -> dict:
        """Returns a single temporal graph snapshot.

        Returns dict with:
            x: (N_nodes, window_size, 12) float tensor
            adj: (N_nodes, N_nodes) float tensor
            y: (n_targets,) float tensor
            target_indices: list[int]
            timestamp: str
        """
        end_time = self.sample_times[idx]

        # Extract features for each node
        x = np.zeros(
            (self.n_nodes, self.cfg.features.window_size, self.cfg.features.n_features),
            dtype=np.float32,
        )

        for cid in self.node_ids:
            node_idx = self.node_to_idx[cid]
            info = self.market_info.get(cid, {})
            eslug = info.get("event_slug", "")
            try:
                x[node_idx] = self.extractor.extract(cid, eslug, end_time)
            except Exception as e:
                logger.debug("Feature extraction failed for %s: %s", cid, e)

        # Labels: next-step price for target nodes
        y = self._fetch_next_prices(end_time)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "adj": torch.tensor(self.adj, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "target_indices": self.target_indices,
            "timestamp": end_time.isoformat(),
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _fetch_market_info(self) -> dict[str, dict]:
        """Fetch event_slug and category for each condition_id."""
        info: dict[str, dict] = {}
        for cid in self.node_ids:
            info[cid] = {"condition_id": cid, "event_slug": "", "category": ""}

        try:
            placeholders = ", ".join(f"'{c}'" for c in self.node_ids)
            rows = self.client.query(f"""
                SELECT condition_id, event_slug, category
                FROM markets FINAL
                WHERE condition_id IN ({placeholders})
            """).result_rows

            for cid, eslug, cat in rows:
                if cid in info:
                    info[cid]["event_slug"] = eslug or ""
                    info[cid]["category"] = cat or ""
        except Exception as e:
            logger.warning("Could not fetch market info: %s", e)

        return info

    def _fetch_next_prices(self, end_time: datetime) -> np.ndarray:
        """Fetch next-step price for each target (supervised label).

        Tries market_prices first, falls back to market_trades.
        """
        step_min = self.cfg.features.step_minutes
        next_time = end_time + timedelta(minutes=step_min)
        y = np.full(len(self.target_ids), 0.5, dtype=np.float32)

        for i, cid in enumerate(self.target_ids):
            try:
                # Try market_prices
                rows = self.client.query("""
                    SELECT price FROM market_prices
                    WHERE condition_id = {cid:String}
                      AND outcome = 'Yes'
                      AND timestamp >= {start:DateTime64(3)}
                      AND timestamp < {end:DateTime64(3)}
                    ORDER BY timestamp ASC LIMIT 1
                """, parameters={
                    "cid": cid, "start": end_time,
                    "end": next_time + timedelta(minutes=step_min),
                }).result_rows

                if rows:
                    y[i] = rows[0][0]
                    continue

                # Fallback: market_trades
                rows = self.client.query("""
                    SELECT price FROM market_trades
                    WHERE condition_id = {cid:String}
                      AND outcome = 'Yes'
                      AND timestamp >= {start:DateTime64(3)}
                      AND timestamp < {end:DateTime64(3)}
                    ORDER BY timestamp ASC LIMIT 1
                """, parameters={
                    "cid": cid, "start": end_time,
                    "end": next_time + timedelta(minutes=step_min),
                }).result_rows

                if rows:
                    y[i] = rows[0][0]
            except Exception:
                pass

        return y


def create_dataset(
    client: Any,
    config: GNNConfig,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    stride_minutes: int = 30,
) -> Dataset:
    """Factory: create the appropriate dataset based on config.graph_type.

    - 'settlement': DonbasTemporalDataset (geographic graph from JSON files)
    - 'market': GenericMarketDataset (similarity graph from ClickHouse)
    """
    if config.graph_type == "settlement":
        from .dataset import DonbasTemporalDataset
        return DonbasTemporalDataset(
            client, config, start_date=start_date,
            end_date=end_date, stride_minutes=stride_minutes,
        )

    elif config.graph_type == "market":
        from .graph_builder import MarketGraphBuilder
        # Fetch top markets by volume
        top_n = config.graph.top_markets
        try:
            rows = client.query(f"""
                SELECT condition_id
                FROM markets FINAL
                WHERE active = 1 AND closed = 0
                  AND volume_24h > 0
                ORDER BY volume_24h DESC
                LIMIT {top_n}
            """).result_rows
            condition_ids = [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.error("Failed to fetch top markets: %s", e)
            raise

        if len(condition_ids) < 10:
            raise ValueError(f"Only {len(condition_ids)} active markets found, need >= 10")

        logger.info("Building market graph for %d markets", len(condition_ids))
        builder = MarketGraphBuilder(client, config.graph)

        # Try cached graph first, fall back to on-demand
        result = builder.build_from_cached(condition_ids)

        # Update model config
        config.model.n_targets = len(condition_ids)

        return GenericMarketDataset(
            client,
            condition_ids,
            adjacency=result.adjacency,
            config=config,
            start_date=start_date,
            end_date=end_date,
            stride_minutes=stride_minutes,
        )

    else:
        raise ValueError(f"Unknown graph_type: {config.graph_type}")
