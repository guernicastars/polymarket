"""Cross-market causal discovery using the PC algorithm and Granger tests.

Discovers the causal directed acyclic graph (DAG) among a panel of
prediction markets. Combines two complementary methods:

1. **PC algorithm** (constraint-based): Tests conditional independences
   via partial correlation to determine which edges to keep and orient.
   Captures contemporaneous causal structure.

2. **Granger DAG** (time-series-based): Builds a DAG from Granger
   causality tests. Captures lagged causal structure.

Both DAGs can be merged to produce a combined evidence DAG.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client
from scipy import stats as sp_stats

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class CrossMarketAnalyzer:
    """Discover causal DAG structure among prediction markets.

    Uses partial-correlation-based PC algorithm and Granger causality
    to identify which markets causally influence others.

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

    def fetch_panel_data(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Fetch aligned price-change panel data for a set of tokens.

        Returns first-differences (price changes) rather than levels,
        because causal discovery methods assume stationarity.

        Parameters
        ----------
        token_ids : list[str]
            Token IDs to include.
        start_date : datetime
            Start of the observation window.
        end_date : datetime
            End of the observation window.
        resample_minutes : int
            Width of each time bin in minutes.

        Returns
        -------
        dict
            Keys:
            - ``token_ids`` (list[str]): tokens with available data.
            - ``data`` (np.ndarray): (T, N) matrix of price changes.
            - ``n_obs`` (int): number of time observations.
        """
        client = self._get_client()
        token_list = ", ".join(f"'{t}'" for t in token_ids)
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
            SELECT
                token_id,
                toStartOfInterval(timestamp, INTERVAL {resample_minutes} MINUTE) AS bucket,
                avg(price) AS avg_price
            FROM market_prices
            WHERE token_id IN ({token_list})
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
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

        if not raw:
            return {"token_ids": [], "data": np.array([]), "n_obs": 0}

        # Common time axis
        all_buckets: set[datetime] = set()
        for buckets in raw.values():
            all_buckets.update(buckets.keys())
        time_axis = sorted(all_buckets)

        # Build aligned price matrix and take first-differences
        available: list[str] = []
        columns: list[np.ndarray] = []

        for tid in token_ids:
            if tid not in raw:
                continue
            prices = []
            last_price = None
            for t in time_axis:
                if t in raw[tid]:
                    last_price = raw[tid][t]
                prices.append(last_price if last_price is not None else np.nan)
            arr = np.array(prices, dtype=np.float64)

            # Forward fill
            mask = np.isnan(arr)
            if mask.all():
                continue
            first_valid = int(np.argmax(~mask))
            arr[:first_valid] = arr[first_valid]
            remaining = np.isnan(arr)
            if remaining.any():
                idx = np.where(~remaining, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]

            # First difference for stationarity
            diff = np.diff(arr)
            available.append(tid)
            columns.append(diff)

        if not columns:
            return {"token_ids": [], "data": np.array([]), "n_obs": 0}

        data = np.column_stack(columns)  # (T-1, N)

        logger.info(
            "panel_data_fetched",
            extra={
                "tokens": len(available),
                "observations": data.shape[0],
            },
        )

        return {
            "token_ids": available,
            "data": data,
            "n_obs": data.shape[0],
        }

    @staticmethod
    def _partial_correlation(
        data: np.ndarray, i: int, j: int, conditioning: list[int]
    ) -> tuple[float, float]:
        """Compute partial correlation between columns i and j given conditioning set.

        Uses recursive formula for partial correlation:
        rho(i,j | Z) computed via regression residuals.

        Parameters
        ----------
        data : np.ndarray
            (T, N) data matrix.
        i : int
            Column index of first variable.
        j : int
            Column index of second variable.
        conditioning : list[int]
            Column indices to condition on.

        Returns
        -------
        tuple[float, float]
            (partial_correlation, p_value).
        """
        n = data.shape[0]

        if not conditioning:
            # Simple Pearson correlation
            r, p = sp_stats.pearsonr(data[:, i], data[:, j])
            return float(r), float(p)

        # Regress i and j on the conditioning set, use residuals
        Z = data[:, conditioning]
        # Add intercept
        Z_aug = np.column_stack([np.ones(n), Z])

        try:
            # Residuals of i on Z
            beta_i, _, _, _ = np.linalg.lstsq(Z_aug, data[:, i], rcond=None)
            resid_i = data[:, i] - Z_aug @ beta_i

            # Residuals of j on Z
            beta_j, _, _, _ = np.linalg.lstsq(Z_aug, data[:, j], rcond=None)
            resid_j = data[:, j] - Z_aug @ beta_j

            # Correlation of residuals
            if np.std(resid_i) < 1e-12 or np.std(resid_j) < 1e-12:
                return 0.0, 1.0

            r, p = sp_stats.pearsonr(resid_i, resid_j)
            return float(r), float(p)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    def discover_dag_pc(
        self,
        panel_data: dict[str, Any],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Discover a causal DAG using a simplified PC algorithm.

        The PC algorithm works in two phases:
        1. **Skeleton discovery**: start with a complete undirected graph,
           remove edges (i, j) if they become conditionally independent
           given some subset of their neighbors.
        2. **Orientation**: orient edges using v-structures (colliders).

        This is a simplified implementation that conditions on up to 2
        variables at a time, suitable for moderate-sized panels.

        Parameters
        ----------
        panel_data : dict
            Output of ``fetch_panel_data``.
        alpha : float
            Significance level for conditional independence tests (default 0.05).

        Returns
        -------
        dict
            Keys:
            - ``token_ids`` (list[str]): ordered tokens (node labels).
            - ``adjacency`` (list[list[int]]): NxN adjacency matrix.
              ``adjacency[i][j] = 1`` means an edge from i to j.
            - ``edges`` (list[dict]): list of directed edges with metadata.
            - ``n_edges`` (int): total directed edges found.
        """
        token_ids = panel_data["token_ids"]
        data = panel_data["data"]
        n = len(token_ids)

        if n < 2 or data.shape[0] < 10:
            logger.warning("pc_insufficient_data", extra={"n": n, "obs": data.shape[0]})
            return {
                "token_ids": token_ids,
                "adjacency": [[0] * n for _ in range(n)],
                "edges": [],
                "n_edges": 0,
            }

        # Phase 1: Skeleton discovery
        # Start with complete undirected graph
        skeleton = [[1] * n for _ in range(n)]
        for i in range(n):
            skeleton[i][i] = 0

        # Separation sets: sep[i][j] = set of variables that make i _||_ j
        sep_sets: dict[tuple[int, int], list[int]] = {}

        # Depth 0: test unconditional independence
        for i in range(n):
            for j in range(i + 1, n):
                if skeleton[i][j] == 0:
                    continue
                r, p = self._partial_correlation(data, i, j, [])
                if p > alpha:
                    skeleton[i][j] = 0
                    skeleton[j][i] = 0
                    sep_sets[(i, j)] = []
                    sep_sets[(j, i)] = []

        # Depth 1: condition on each single neighbor
        for i in range(n):
            for j in range(i + 1, n):
                if skeleton[i][j] == 0:
                    continue
                # Neighbors of i (excluding j)
                neighbors_i = [k for k in range(n) if k != i and k != j and skeleton[i][k] == 1]
                removed = False
                for k in neighbors_i:
                    r, p = self._partial_correlation(data, i, j, [k])
                    if p > alpha:
                        skeleton[i][j] = 0
                        skeleton[j][i] = 0
                        sep_sets[(i, j)] = [k]
                        sep_sets[(j, i)] = [k]
                        removed = True
                        break
                if removed:
                    continue
                # Also try neighbors of j
                neighbors_j = [k for k in range(n) if k != i and k != j and skeleton[j][k] == 1]
                for k in neighbors_j:
                    if k in neighbors_i:
                        continue  # Already tested
                    r, p = self._partial_correlation(data, i, j, [k])
                    if p > alpha:
                        skeleton[i][j] = 0
                        skeleton[j][i] = 0
                        sep_sets[(i, j)] = [k]
                        sep_sets[(j, i)] = [k]
                        break

        # Depth 2: condition on pairs of neighbors (only if many edges remain)
        remaining_edges = sum(skeleton[i][j] for i in range(n) for j in range(i + 1, n))
        if remaining_edges > 0 and n >= 4:
            for i in range(n):
                for j in range(i + 1, n):
                    if skeleton[i][j] == 0:
                        continue
                    neighbors_i = [k for k in range(n) if k != i and k != j and skeleton[i][k] == 1]
                    if len(neighbors_i) < 2:
                        continue
                    removed = False
                    for a in range(len(neighbors_i)):
                        for b in range(a + 1, len(neighbors_i)):
                            cond = [neighbors_i[a], neighbors_i[b]]
                            r, p = self._partial_correlation(data, i, j, cond)
                            if p > alpha:
                                skeleton[i][j] = 0
                                skeleton[j][i] = 0
                                sep_sets[(i, j)] = cond
                                sep_sets[(j, i)] = cond
                                removed = True
                                break
                        if removed:
                            break

        # Phase 2: Orient edges (v-structure detection)
        # For each triple i - k - j where i and j are not adjacent:
        # If k is NOT in sep(i, j), then orient as i -> k <- j (collider)
        adjacency = [[0] * n for _ in range(n)]

        # Start with undirected skeleton
        for i in range(n):
            for j in range(n):
                adjacency[i][j] = skeleton[i][j]

        # Find and orient v-structures
        for k in range(n):
            # Find pairs (i, j) that are both neighbors of k but not of each other
            neighbors_k = [m for m in range(n) if skeleton[k][m] == 1]
            for idx_a in range(len(neighbors_k)):
                for idx_b in range(idx_a + 1, len(neighbors_k)):
                    i = neighbors_k[idx_a]
                    j = neighbors_k[idx_b]
                    # i and j must not be adjacent
                    if skeleton[i][j] == 1:
                        continue
                    # Check if k is in the separation set of (i, j)
                    sep = sep_sets.get((i, j), sep_sets.get((j, i), None))
                    if sep is not None and k not in sep:
                        # Orient as i -> k <- j (collider)
                        adjacency[i][k] = 1
                        adjacency[k][i] = 0
                        adjacency[j][k] = 1
                        adjacency[k][j] = 0

        # Build edge list
        edges: list[dict[str, Any]] = []
        for i in range(n):
            for j in range(n):
                if adjacency[i][j] == 1 and adjacency[j][i] == 0:
                    # Directed edge i -> j
                    r, _ = self._partial_correlation(data, i, j, [])
                    edges.append({
                        "source": token_ids[i],
                        "target": token_ids[j],
                        "type": "directed",
                        "correlation": round(r, 4),
                    })
                elif adjacency[i][j] == 1 and adjacency[j][i] == 1 and i < j:
                    # Undirected edge i -- j
                    r, _ = self._partial_correlation(data, i, j, [])
                    edges.append({
                        "source": token_ids[i],
                        "target": token_ids[j],
                        "type": "undirected",
                        "correlation": round(r, 4),
                    })

        logger.info(
            "pc_dag_complete",
            extra={
                "nodes": n,
                "edges_found": len(edges),
            },
        )

        return {
            "token_ids": token_ids,
            "adjacency": adjacency,
            "edges": edges,
            "n_edges": len(edges),
        }

    def discover_dag_granger(
        self,
        panel_data: dict[str, Any],
        max_lag: int = 5,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Build a DAG from pairwise Granger causality tests.

        An edge i -> j exists if token i Granger-causes token j at the
        given significance level.

        Parameters
        ----------
        panel_data : dict
            Output of ``fetch_panel_data``.
        max_lag : int
            Maximum lag for Granger tests.
        alpha : float
            Significance threshold.

        Returns
        -------
        dict
            Same structure as ``discover_dag_pc``.
        """
        from pipeline.causal.granger import GrangerCausalityAnalyzer

        token_ids = panel_data["token_ids"]
        data = panel_data["data"]
        n = len(token_ids)

        if n < 2 or data.shape[0] < max_lag + 3:
            return {
                "token_ids": token_ids,
                "adjacency": [[0] * n for _ in range(n)],
                "edges": [],
                "n_edges": 0,
            }

        analyzer = GrangerCausalityAnalyzer(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
        )

        adjacency = [[0] * n for _ in range(n)]
        edges: list[dict[str, Any]] = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result = analyzer.pairwise_granger(
                    data[:, i], data[:, j], max_lag
                )
                if result["significant"]:
                    adjacency[i][j] = 1
                    edges.append({
                        "source": token_ids[i],
                        "target": token_ids[j],
                        "type": "granger",
                        "f_stat": result["best_f_stat"],
                        "p_value": result["best_p_value"],
                        "lag": result["best_lag"],
                    })

        logger.info(
            "granger_dag_complete",
            extra={
                "nodes": n,
                "edges_found": len(edges),
            },
        )

        return {
            "token_ids": token_ids,
            "adjacency": adjacency,
            "edges": edges,
            "n_edges": len(edges),
        }

    @staticmethod
    def merge_evidence(
        pc_dag: dict[str, Any],
        granger_dag: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge evidence from PC and Granger DAGs into a combined DAG.

        An edge exists in the merged DAG if it appears in EITHER the PC
        or the Granger DAG. Edges found by both methods are marked with
        higher confidence.

        Parameters
        ----------
        pc_dag : dict
            Output of ``discover_dag_pc``.
        granger_dag : dict
            Output of ``discover_dag_granger``.

        Returns
        -------
        dict
            Keys:
            - ``token_ids`` (list[str]): ordered tokens.
            - ``adjacency`` (list[list[int]]): merged adjacency matrix.
            - ``edges`` (list[dict]): edges with method attribution.
            - ``n_edges`` (int): total edges.
        """
        # Ensure both DAGs have the same token ordering
        token_ids = pc_dag["token_ids"]
        n = len(token_ids)

        # Build index map for granger_dag in case ordering differs
        granger_idx: dict[str, int] = {}
        for idx, tid in enumerate(granger_dag["token_ids"]):
            granger_idx[tid] = idx

        adjacency = [[0] * n for _ in range(n)]
        edge_evidence: dict[tuple[int, int], list[str]] = {}

        # Add PC edges
        for i in range(n):
            for j in range(n):
                if pc_dag["adjacency"][i][j] == 1:
                    adjacency[i][j] = 1
                    edge_evidence.setdefault((i, j), []).append("pc")

        # Add Granger edges
        for edge in granger_dag["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            if src in granger_idx and tgt in granger_idx:
                # Map to pc_dag token ordering
                try:
                    i = token_ids.index(src)
                    j = token_ids.index(tgt)
                    adjacency[i][j] = 1
                    edge_evidence.setdefault((i, j), []).append("granger")
                except ValueError:
                    pass

        # Build merged edge list
        edges: list[dict[str, Any]] = []
        for (i, j), methods in edge_evidence.items():
            edges.append({
                "source": token_ids[i],
                "target": token_ids[j],
                "methods": methods,
                "confidence": len(methods) / 2.0,  # 0.5 if one method, 1.0 if both
                "type": "directed" if adjacency[j][i] == 0 else "bidirectional",
            })

        logger.info(
            "merged_dag_complete",
            extra={
                "nodes": n,
                "edges": len(edges),
                "both_methods": sum(1 for e in edges if len(e["methods"]) == 2),
            },
        )

        return {
            "token_ids": token_ids,
            "adjacency": adjacency,
            "edges": edges,
            "n_edges": len(edges),
        }

    @staticmethod
    def to_networkx(dag: dict[str, Any]) -> Any:
        """Convert a DAG result to a NetworkX DiGraph.

        Parameters
        ----------
        dag : dict
            Output of ``discover_dag_pc``, ``discover_dag_granger``,
            or ``merge_evidence``.

        Returns
        -------
        networkx.DiGraph
            Directed graph with token_ids as nodes and edge metadata
            as edge attributes.
        """
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(dag["token_ids"])

        for edge in dag["edges"]:
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            G.add_edge(edge["source"], edge["target"], **attrs)

        return G
