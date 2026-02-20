"""Job: pre-compute market similarity graph edges hourly.

Fetches top markets by volume, builds a combined similarity graph via
MarketGraphBuilder, and writes edges + graph metrics to ClickHouse.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import clickhouse_connect
import numpy as np

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)

# Tuning — these mirror GraphConfig defaults but are used at pipeline level
SIMILARITY_TOP_MARKETS = 500
SIMILARITY_MIN_WEIGHT = 0.15


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    """Create a read-only ClickHouse client for queries."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


async def run_similarity_scorer() -> None:
    """Compute pairwise market similarity and write to ClickHouse.

    Steps:
      1. Fetch top N markets by 24h volume from `markets FINAL`
      2. Build combined similarity graph via MarketGraphBuilder
      3. Extract edges above threshold → market_similarity_graph
      4. Compute graph-level metrics → market_graph_metrics
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # 1. Fetch top markets by volume
        query = """
            SELECT condition_id
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            ORDER BY volume_24h DESC
            LIMIT {top_n:UInt32}
        """
        result = await asyncio.to_thread(
            client.query, query, parameters={"top_n": SIMILARITY_TOP_MARKETS}
        )
        condition_ids = [row[0] for row in result.result_rows]

        if len(condition_ids) < 5:
            logger.warning(
                "similarity_scorer_skip",
                extra={"reason": "too_few_markets", "count": len(condition_ids)},
            )
            return

        logger.info(
            "similarity_scorer_start",
            extra={"markets": len(condition_ids)},
        )

        # 2. Build combined similarity graph
        # Import here to avoid circular imports at module load
        from network.gnn.config import GraphConfig
        from network.gnn.graph_builder import MarketGraphBuilder

        cfg = GraphConfig(
            method="combined",
            min_similarity=SIMILARITY_MIN_WEIGHT,
            top_markets=SIMILARITY_TOP_MARKETS,
        )
        builder = MarketGraphBuilder(client, cfg)
        graph_result = await asyncio.to_thread(builder.build, condition_ids, "combined")

        adj = graph_result.adjacency
        node_to_idx = graph_result.node_to_idx
        idx_to_node = graph_result.idx_to_node
        n = adj.shape[0]

        # 3. Extract edges above threshold and write to market_similarity_graph
        edge_rows = []
        for i in range(n):
            for j in range(i + 1, n):
                score = float(adj[i, j])
                if score > 0:
                    source_id = idx_to_node[i]
                    target_id = idx_to_node[j]

                    # Get per-component breakdown by building individual methods
                    components = {}
                    for method_name in ("event", "whale", "correlation", "signal", "category"):
                        try:
                            single_result = builder.build([source_id, target_id], method_name)
                            components[method_name] = round(float(single_result.adjacency[0, 1]), 4)
                        except Exception:
                            components[method_name] = 0.0

                    edge_rows.append([
                        source_id,       # source_id
                        target_id,       # target_id
                        round(score, 6), # similarity_score
                        json.dumps(components),  # components JSON
                        "combined",      # method
                        now,             # computed_at
                    ])

        if edge_rows:
            await writer.write("market_similarity_graph", edge_rows)
            logger.info(
                "similarity_edges_written",
                extra={"edges": len(edge_rows)},
            )

        # 4. Compute graph-level metrics
        edge_count = graph_result.edge_count
        density = (2 * edge_count) / (n * (n - 1)) if n > 1 else 0.0
        degrees = (adj > 0).sum(axis=1).astype(float)
        avg_degree = float(degrees.mean()) if n > 0 else 0.0

        # Clustering coefficient: fraction of closed triplets
        clustering = _compute_clustering_coefficient(adj)

        # Edge weight stats
        nonzero_weights = adj[adj > 0]
        min_w = float(nonzero_weights.min()) if len(nonzero_weights) > 0 else 0.0
        max_w = float(nonzero_weights.max()) if len(nonzero_weights) > 0 else 0.0
        median_w = float(np.median(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0

        metrics_row = [
            "combined",           # method
            now,                  # refresh_time
            n,                    # node_count
            edge_count,           # edge_count
            round(density, 6),    # density
            round(avg_degree, 4), # avg_degree
            round(clustering, 6), # clustering_coeff
            round(min_w, 6),      # min_weight
            round(max_w, 6),      # max_weight
            round(median_w, 6),   # median_weight
            now,                  # computed_at
        ]
        await writer.write("market_graph_metrics", [metrics_row])

        logger.info(
            "similarity_scorer_complete",
            extra={
                "nodes": n,
                "edges": edge_count,
                "density": round(density, 4),
                "avg_degree": round(avg_degree, 2),
                "clustering": round(clustering, 4),
            },
        )

    except Exception:
        logger.error("similarity_scorer_error", exc_info=True)


def _compute_clustering_coefficient(adj: "np.ndarray") -> float:
    """Compute average clustering coefficient from adjacency matrix."""
    import numpy as np

    n = adj.shape[0]
    if n < 3:
        return 0.0

    binary = (adj > 0).astype(float)
    total_cc = 0.0
    count = 0

    for i in range(n):
        neighbors = np.where(binary[i] > 0)[0]
        neighbors = neighbors[neighbors != i]  # exclude self-loops
        k = len(neighbors)
        if k < 2:
            continue

        # Count edges between neighbors
        submat = binary[np.ix_(neighbors, neighbors)]
        actual_edges = submat.sum() / 2  # undirected
        possible_edges = k * (k - 1) / 2
        total_cc += actual_edges / possible_edges
        count += 1

    return total_cc / count if count > 0 else 0.0
