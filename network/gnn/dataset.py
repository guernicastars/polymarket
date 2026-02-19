"""Dataset: ClickHouse → temporal graph snapshots → PyTorch tensors.

Builds the adjacency matrix from the Donbas settlement graph and
extracts 12-feature windows per node from ClickHouse data.
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GNNConfig
from .features import FeatureExtractor

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class DonbasTemporalDataset(Dataset):
    """Temporal graph dataset for the Donbas settlement network.

    Each sample is a (window_size × N_nodes × 12_features) tensor
    with a (n_targets,) label vector (next-step price direction or resolution).

    The adjacency matrix is static (graph structure doesn't change per sample).
    """

    def __init__(
        self,
        client,
        config: Optional[GNNConfig] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        stride_minutes: int = 30,
    ):
        """
        Args:
            client: clickhouse_connect Client
            config: full GNN config
            start_date: beginning of data range
            end_date: end of data range
            stride_minutes: how far to slide the window between samples
        """
        self.client = client
        self.cfg = config or GNNConfig()
        self.stride = stride_minutes
        self.extractor = FeatureExtractor(client, self.cfg.features)

        # Load graph structure
        self.settlements = self._load_settlements()
        self.node_ids = sorted(self.settlements.keys())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)

        # Identify PM target indices
        self.target_ids = [
            s["id"] for s in self.settlements.values()
            if s.get("is_polymarket_target", False)
        ]
        self.target_indices = [self.node_to_idx[tid] for tid in self.target_ids]

        # Load adjacency
        self.adj = self._build_adjacency()

        # Load condition_id and event_slug mapping for each PM target
        self.market_info = self._load_market_info()

        # Build sample timestamps
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        window_minutes = self.cfg.features.window_size * self.cfg.features.step_minutes
        self.sample_times = []
        t = start_date + timedelta(minutes=window_minutes)
        while t <= end_date:
            self.sample_times.append(t)
            t += timedelta(minutes=self.stride)

        logger.info(
            "Dataset: %d nodes, %d targets, %d samples, stride=%dmin",
            self.n_nodes, len(self.target_ids), len(self.sample_times), self.stride,
        )

    def __len__(self) -> int:
        return len(self.sample_times)

    def __getitem__(self, idx: int) -> dict:
        """Returns a single temporal graph snapshot.

        Returns dict with:
            x: (N_nodes, window_size, 12) float tensor
            adj: (N_nodes, N_nodes) float tensor
            y: (n_targets,) float tensor — next-step price for targets
            target_indices: list[int]
            timestamp: str
        """
        end_time = self.sample_times[idx]

        # Extract features for each node
        x = np.zeros(
            (self.n_nodes, self.cfg.features.window_size, self.cfg.features.n_features),
            dtype=np.float32,
        )

        for nid in self.target_ids:
            node_idx = self.node_to_idx[nid]
            info = self.market_info.get(nid, {})
            cid = info.get("condition_id", "")
            eslug = info.get("event_slug", "")
            if cid:
                try:
                    x[node_idx] = self.extractor.extract(cid, eslug, end_time)
                except Exception as e:
                    logger.warning("Feature extraction failed for %s: %s", nid, e)

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
    # Graph construction
    # ------------------------------------------------------------------

    def _load_settlements(self) -> dict:
        path = DATA_DIR / "settlements.json"
        with open(path) as f:
            raw = json.load(f)
        return {s["id"]: s for s in raw}

    def _build_adjacency(self) -> np.ndarray:
        """Build dense adjacency matrix from edges.json with self-loops."""
        adj = np.eye(self.n_nodes, dtype=np.float32)  # self-loops

        path = DATA_DIR / "edges.json"
        with open(path) as f:
            edges = json.load(f)

        for e in edges:
            src = e["source"]
            tgt = e["target"]
            if src in self.node_to_idx and tgt in self.node_to_idx:
                i, j = self.node_to_idx[src], self.node_to_idx[tgt]
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # undirected

        # Normalize: D^{-1/2} A D^{-1/2}
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj = adj * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

        return adj

    def _load_market_info(self) -> dict:
        """Load condition_id mapping for PM target settlements."""
        mapping_path = DATA_DIR / "polymarket_mapping.json"
        with open(mapping_path) as f:
            raw = json.load(f)

        info = {}
        for sid, data in raw.items():
            info[sid] = {
                "slug": data.get("slug", ""),
                "condition_id": data.get("condition_id", ""),
                "event_slug": "",  # filled from ClickHouse if available
            }

        # Try to resolve condition_ids from ClickHouse
        self._resolve_condition_ids(info)
        return info

    def _resolve_condition_ids(self, info: dict) -> None:
        """Look up condition_ids from ClickHouse markets table by slug."""
        slugs = [v["slug"] for v in info.values() if v["slug"]]
        if not slugs:
            return

        try:
            placeholders = ", ".join(f"'{s}'" for s in slugs)
            sql = f"""
                SELECT market_slug, condition_id, event_slug
                FROM markets FINAL
                WHERE market_slug IN ({placeholders})
            """
            rows = self.client.query(sql).result_rows
            slug_to_row = {r[0]: r for r in rows}

            for sid, data in info.items():
                row = slug_to_row.get(data["slug"])
                if row:
                    data["condition_id"] = row[1]
                    data["event_slug"] = row[2]
                    logger.info("Resolved %s → condition_id=%s", sid, row[1])
        except Exception as e:
            logger.warning("Could not resolve condition IDs: %s", e)

    def _fetch_next_prices(self, end_time: datetime) -> np.ndarray:
        """Fetch the next-step price for each target (label for supervised learning).

        Tries market_prices first, falls back to market_trades.
        """
        step_min = self.cfg.features.step_minutes
        next_time = end_time + timedelta(minutes=step_min)
        y = np.full(len(self.target_ids), 0.5, dtype=np.float32)

        for i, sid in enumerate(self.target_ids):
            info = self.market_info.get(sid, {})
            cid = info.get("condition_id", "")
            if not cid:
                continue
            try:
                # Try market_prices first
                rows = self.client.query(
                    """
                    SELECT price FROM market_prices
                    WHERE condition_id = {condition_id:String}
                      AND outcome = 'Yes'
                      AND timestamp >= {start:DateTime64(3)}
                      AND timestamp < {end:DateTime64(3)}
                    ORDER BY timestamp ASC
                    LIMIT 1
                    """,
                    parameters={
                        "condition_id": cid,
                        "start": end_time,
                        "end": next_time + timedelta(minutes=step_min),
                    },
                ).result_rows
                if rows:
                    y[i] = rows[0][0]
                    continue

                # Fallback: use trade prices
                rows = self.client.query(
                    """
                    SELECT price FROM market_trades
                    WHERE condition_id = {condition_id:String}
                      AND outcome = 'Yes'
                      AND timestamp >= {start:DateTime64(3)}
                      AND timestamp < {end:DateTime64(3)}
                    ORDER BY timestamp ASC
                    LIMIT 1
                    """,
                    parameters={
                        "condition_id": cid,
                        "start": end_time,
                        "end": next_time + timedelta(minutes=step_min),
                    },
                ).result_rows
                if rows:
                    y[i] = rows[0][0]
            except Exception:
                pass

        return y


def collate_graph_batch(batch: list[dict]) -> dict:
    """Custom collator for DataLoader — stacks graph samples."""
    return {
        "x": torch.stack([b["x"] for b in batch]),        # (B, N, W, F)
        "adj": batch[0]["adj"],                             # (N, N) — shared
        "y": torch.stack([b["y"] for b in batch]),          # (B, n_targets)
        "target_indices": batch[0]["target_indices"],
        "timestamps": [b["timestamp"] for b in batch],
    }
