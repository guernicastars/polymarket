"""Build market similarity graphs from ClickHouse data.

Constructs adjacency matrices for the GNN-TCN model using multiple
similarity measures between Polymarket markets. Supports both
pre-computed (from pipeline job) and on-demand graph construction.

Usage:
    from network.gnn.graph_builder import MarketGraphBuilder
    from network.gnn.config import GraphConfig

    builder = MarketGraphBuilder(client, GraphConfig())
    result = builder.build(condition_ids, method="combined")
    adj = result["adjacency"]  # (N, N) numpy array
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from network.gnn.config import GraphConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    """Result of graph construction."""

    adjacency: np.ndarray           # (N, N) float32
    node_to_idx: dict[str, int]
    idx_to_node: dict[int, str]
    edge_count: int
    method: str
    timestamp: datetime


class MarketGraphBuilder:
    """Builds market similarity adjacency matrices from ClickHouse.

    Supports 5 individual similarity methods and a weighted combination:
      - event: binary (same event_slug)
      - whale: Jaccard of top holders
      - correlation: Pearson of hourly log returns
      - signal: cosine similarity of composite signal components
      - category: binary (same category tag)
      - combined: weighted sum of all above
    """

    def __init__(
        self,
        client: Any,  # clickhouse_connect.driver.Client
        config: Optional[GraphConfig] = None,
    ) -> None:
        self.client = client
        self.cfg = config or GraphConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        condition_ids: list[str],
        method: Optional[str] = None,
    ) -> GraphResult:
        """Build adjacency matrix for the given markets.

        Args:
            condition_ids: List of market condition IDs (nodes).
            method: Similarity method override. Defaults to config.method.

        Returns:
            GraphResult with adjacency matrix and node mappings.
        """
        method = method or self.cfg.method
        n = len(condition_ids)
        if n == 0:
            raise ValueError("No condition_ids provided")

        node_to_idx = {cid: i for i, cid in enumerate(sorted(condition_ids))}
        idx_to_node = {i: cid for cid, i in node_to_idx.items()}
        cids_sorted = sorted(condition_ids)

        logger.info("Building %s graph for %d markets", method, n)

        dispatch = {
            "event": self._event_similarity,
            "whale": self._whale_overlap,
            "correlation": self._price_correlation,
            "signal": self._signal_fingerprint,
            "category": self._category_similarity,
            "combined": self._combined,
        }

        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}. Options: {list(dispatch)}")

        adj = dispatch[method](cids_sorted, node_to_idx)

        # Post-processing
        adj = self._threshold(adj, self.cfg.min_similarity)
        if self.cfg.add_self_loops:
            np.fill_diagonal(adj, 1.0)
        if self.cfg.symmetric:
            adj = np.maximum(adj, adj.T)
        adj = self._symmetric_normalize(adj)

        edge_count = int((adj > 0).sum()) - n  # exclude self-loops

        logger.info(
            "Graph built: %d nodes, %d edges, density=%.3f",
            n, edge_count, edge_count / max(n * (n - 1), 1),
        )

        return GraphResult(
            adjacency=adj.astype(np.float32),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            edge_count=edge_count,
            method=method,
            timestamp=datetime.now(timezone.utc),
        )

    def build_from_cached(
        self,
        condition_ids: list[str],
    ) -> GraphResult:
        """Build adjacency from pre-computed market_similarity_graph table.

        Falls back to on-demand computation if cache is empty.
        """
        n = len(condition_ids)
        node_to_idx = {cid: i for i, cid in enumerate(sorted(condition_ids))}
        idx_to_node = {i: cid for cid, i in node_to_idx.items()}
        cids_sorted = sorted(condition_ids)

        try:
            placeholders = ", ".join(f"'{c}'" for c in cids_sorted)
            rows = self.client.query(f"""
                SELECT source_id, target_id, similarity_score
                FROM market_similarity_graph FINAL
                WHERE source_id IN ({placeholders})
                  AND target_id IN ({placeholders})
                  AND computed_at >= now() - INTERVAL 4 HOUR
            """).result_rows
        except Exception as e:
            logger.warning("Cache read failed: %s, falling back to on-demand", e)
            return self.build(condition_ids)

        if not rows:
            logger.info("No cached edges, computing on-demand")
            return self.build(condition_ids)

        adj = np.zeros((n, n), dtype=np.float32)
        for src, tgt, score in rows:
            if src in node_to_idx and tgt in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[tgt]
                adj[i, j] = score
                adj[j, i] = score

        np.fill_diagonal(adj, 1.0)
        adj = self._symmetric_normalize(adj)
        edge_count = int((adj > 0).sum()) - n

        return GraphResult(
            adjacency=adj,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            edge_count=edge_count,
            method="cached",
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Similarity methods
    # ------------------------------------------------------------------

    def _event_similarity(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Binary similarity: 1.0 if markets share the same event_slug."""
        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)

        placeholders = ", ".join(f"'{c}'" for c in cids)
        try:
            rows = self.client.query(f"""
                SELECT condition_id, event_slug
                FROM markets FINAL
                WHERE condition_id IN ({placeholders})
                  AND event_slug != ''
            """).result_rows
        except Exception as e:
            logger.warning("event_similarity query failed: %s", e)
            return adj

        # Group by event_slug
        event_groups: dict[str, list[str]] = defaultdict(list)
        for cid, eslug in rows:
            if cid in node_to_idx:
                event_groups[eslug].append(cid)

        # All pairs within same event get weight 1.0
        for eslug, group_cids in event_groups.items():
            for i_idx in range(len(group_cids)):
                for j_idx in range(i_idx + 1, len(group_cids)):
                    i = node_to_idx[group_cids[i_idx]]
                    j = node_to_idx[group_cids[j_idx]]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj

    def _whale_overlap(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Jaccard similarity of top-20 holder sets per market."""
        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)

        placeholders = ", ".join(f"'{c}'" for c in cids)
        try:
            rows = self.client.query(f"""
                SELECT condition_id, proxy_wallet
                FROM market_holders FINAL
                WHERE condition_id IN ({placeholders})
            """).result_rows
        except Exception as e:
            logger.warning("whale_overlap query failed: %s", e)
            return adj

        # Build holder sets
        holders: dict[str, set[str]] = defaultdict(set)
        for cid, wallet in rows:
            if cid in node_to_idx:
                holders[cid].add(wallet)

        # Pairwise Jaccard
        cid_list = [c for c in cids if c in holders and len(holders[c]) > 0]
        for i_idx in range(len(cid_list)):
            for j_idx in range(i_idx + 1, len(cid_list)):
                ci, cj = cid_list[i_idx], cid_list[j_idx]
                si, sj = holders[ci], holders[cj]
                intersection = len(si & sj)
                union = len(si | sj)
                if union > 0:
                    jaccard = intersection / union
                    i = node_to_idx[ci]
                    j = node_to_idx[cj]
                    adj[i, j] = jaccard
                    adj[j, i] = jaccard

        return adj

    def _price_correlation(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Pearson correlation of hourly log returns over lookback period."""
        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)
        days = self.cfg.price_lookback_days
        min_pts = self.cfg.min_data_points

        placeholders = ", ".join(f"'{c}'" for c in cids)
        try:
            rows = self.client.query(f"""
                SELECT condition_id,
                       bar_time,
                       argMinState(open, bar_time) AS open_s,
                       argMaxState(close, bar_time) AS close_s
                FROM ohlcv_1h
                WHERE condition_id IN ({placeholders})
                  AND outcome = 'Yes'
                  AND bar_time >= now() - INTERVAL {days} DAY
                GROUP BY condition_id, bar_time
                ORDER BY condition_id, bar_time
            """).result_rows
        except Exception:
            # Fallback: use market_trades for price
            try:
                rows = self.client.query(f"""
                    SELECT condition_id,
                           toStartOfHour(timestamp) AS bar_time,
                           argMin(price, timestamp) AS open_p,
                           argMax(price, timestamp) AS close_p
                    FROM market_trades
                    WHERE condition_id IN ({placeholders})
                      AND outcome = 'Yes'
                      AND timestamp >= now() - INTERVAL {days} DAY
                    GROUP BY condition_id, bar_time
                    HAVING count() >= 2
                    ORDER BY condition_id, bar_time
                """).result_rows
            except Exception as e:
                logger.warning("price_correlation query failed: %s", e)
                return adj

        # Build time series per market
        prices: dict[str, dict[datetime, float]] = defaultdict(dict)
        for row in rows:
            cid = row[0]
            bar_time = row[1]
            close_val = float(row[-1]) if row[-1] else 0.0
            if cid in node_to_idx and close_val > 0:
                prices[cid][bar_time] = close_val

        # Compute log returns
        returns: dict[str, dict[datetime, float]] = {}
        for cid, price_series in prices.items():
            sorted_times = sorted(price_series.keys())
            if len(sorted_times) < min_pts:
                continue
            ret = {}
            for k in range(1, len(sorted_times)):
                t = sorted_times[k]
                p_prev = price_series[sorted_times[k - 1]]
                p_curr = price_series[t]
                if p_prev > 0 and p_curr > 0:
                    ret[t] = np.log(p_curr / p_prev)
            if len(ret) >= min_pts // 2:
                returns[cid] = ret

        # Pairwise Pearson correlation
        cid_list = [c for c in cids if c in returns]
        for i_idx in range(len(cid_list)):
            for j_idx in range(i_idx + 1, len(cid_list)):
                ci, cj = cid_list[i_idx], cid_list[j_idx]
                common_times = set(returns[ci].keys()) & set(returns[cj].keys())
                if len(common_times) < min_pts // 2:
                    continue
                times = sorted(common_times)
                ri = np.array([returns[ci][t] for t in times])
                rj = np.array([returns[cj][t] for t in times])
                # Pearson correlation
                if np.std(ri) > 1e-10 and np.std(rj) > 1e-10:
                    corr = np.corrcoef(ri, rj)[0, 1]
                    # Use absolute correlation (both positive and negative are signals)
                    sim = abs(corr)
                    i = node_to_idx[ci]
                    j = node_to_idx[cj]
                    adj[i, j] = sim
                    adj[j, i] = sim

        return adj

    def _signal_fingerprint(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Cosine similarity of 8 composite signal components."""
        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)

        placeholders = ", ".join(f"'{c}'" for c in cids)
        try:
            rows = self.client.query(f"""
                SELECT condition_id,
                       obi_score, volume_anomaly_score, large_trade_score,
                       momentum_score, smart_money_score, concentration_score,
                       arbitrage_flag, insider_score
                FROM composite_signals FINAL
                WHERE condition_id IN ({placeholders})
            """).result_rows
        except Exception as e:
            logger.warning("signal_fingerprint query failed: %s", e)
            return adj

        # Build signal vectors
        vectors: dict[str, np.ndarray] = {}
        for row in rows:
            cid = row[0]
            if cid in node_to_idx:
                vec = np.array([float(v) for v in row[1:]], dtype=np.float32)
                vectors[cid] = vec

        # Pairwise cosine similarity
        cid_list = [c for c in cids if c in vectors]
        for i_idx in range(len(cid_list)):
            for j_idx in range(i_idx + 1, len(cid_list)):
                ci, cj = cid_list[i_idx], cid_list[j_idx]
                vi, vj = vectors[ci], vectors[cj]
                norm_i = np.linalg.norm(vi)
                norm_j = np.linalg.norm(vj)
                if norm_i > 1e-10 and norm_j > 1e-10:
                    cosine = np.dot(vi, vj) / (norm_i * norm_j)
                    # Map from [-1, 1] to [0, 1]
                    sim = (cosine + 1.0) / 2.0
                    i = node_to_idx[ci]
                    j = node_to_idx[cj]
                    adj[i, j] = sim
                    adj[j, i] = sim

        return adj

    def _category_similarity(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Binary similarity: 1.0 if markets have the same category."""
        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)

        placeholders = ", ".join(f"'{c}'" for c in cids)
        try:
            rows = self.client.query(f"""
                SELECT condition_id, category
                FROM markets FINAL
                WHERE condition_id IN ({placeholders})
                  AND category != ''
            """).result_rows
        except Exception as e:
            logger.warning("category_similarity query failed: %s", e)
            return adj

        # Group by category
        cat_groups: dict[str, list[str]] = defaultdict(list)
        for cid, cat in rows:
            if cid in node_to_idx:
                cat_groups[cat].append(cid)

        for cat, group_cids in cat_groups.items():
            for i_idx in range(len(group_cids)):
                for j_idx in range(i_idx + 1, len(group_cids)):
                    i = node_to_idx[group_cids[i_idx]]
                    j = node_to_idx[group_cids[j_idx]]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj

    def _combined(
        self,
        cids: list[str],
        node_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Weighted combination of all similarity methods."""
        components = {}
        weights = {}

        if self.cfg.event_weight > 0:
            components["event"] = self._event_similarity(cids, node_to_idx)
            weights["event"] = self.cfg.event_weight

        if self.cfg.whale_weight > 0:
            components["whale"] = self._whale_overlap(cids, node_to_idx)
            weights["whale"] = self.cfg.whale_weight

        if self.cfg.correlation_weight > 0:
            components["correlation"] = self._price_correlation(cids, node_to_idx)
            weights["correlation"] = self.cfg.correlation_weight

        if self.cfg.signal_weight > 0:
            components["signal"] = self._signal_fingerprint(cids, node_to_idx)
            weights["signal"] = self.cfg.signal_weight

        if self.cfg.category_weight > 0:
            components["category"] = self._category_similarity(cids, node_to_idx)
            weights["category"] = self.cfg.category_weight

        n = len(cids)
        adj = np.zeros((n, n), dtype=np.float32)

        total_weight = sum(weights.values())
        for name, mat in components.items():
            w = weights[name] / total_weight if total_weight > 0 else 0
            adj += w * mat
            populated = int((mat > 0).sum())
            logger.debug("Component %s: %d non-zero entries, weight=%.2f", name, populated, w)

        return adj

    # ------------------------------------------------------------------
    # Post-processing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _threshold(adj: np.ndarray, min_weight: float) -> np.ndarray:
        """Zero out edges below minimum weight."""
        adj[adj < min_weight] = 0.0
        return adj

    @staticmethod
    def _symmetric_normalize(adj: np.ndarray) -> np.ndarray:
        """D^{-1/2} A D^{-1/2} normalization (standard GCN)."""
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        return adj * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

    # ------------------------------------------------------------------
    # Edge extraction (for pipeline job persistence)
    # ------------------------------------------------------------------

    def extract_edges(
        self,
        adj: np.ndarray,
        idx_to_node: dict[int, str],
        min_weight: float = 0.0,
    ) -> list[tuple[str, str, float, str]]:
        """Extract edge list from adjacency matrix.

        Returns list of (source_id, target_id, weight, components_json).
        Only returns upper triangle (undirected).
        """
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w > min_weight:
                    edges.append((
                        idx_to_node[i],
                        idx_to_node[j],
                        float(w),
                        "{}",  # components populated by pipeline job
                    ))
        return edges
