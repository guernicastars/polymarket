"""Job: incremental GNN-TCN update cycle + prediction.

Two functions:
  - run_online_gnn_update(): Every 15 min — pull recent data, incremental SGD
  - run_online_gnn_predict(): Every 5 min — forward pass, write predictions
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timezone

import numpy as np
import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)

# Module-level singleton
_learner = None
_extractor = None
_initialized = False
_target_cids: list[str] = []
_target_names: list[str] = []

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


async def _initialize() -> None:
    """Initialize the online learner on first run."""
    global _learner, _extractor, _initialized, _target_cids, _target_names

    from network.gnn.config import GNNConfig, OnlineLearningConfig
    from network.gnn.online_learner import OnlineLearner
    from network.gnn.features import FeatureExtractor

    cfg = GNNConfig()
    online_cfg = OnlineLearningConfig()
    client = _get_read_client()

    # Load market graph info
    data_dir = PROJECT_ROOT / "network" / "data"
    settlements_path = data_dir / "settlements.json"
    edges_path = data_dir / "edges.json"
    mapping_path = data_dir / "polymarket_mapping.json"

    if not settlements_path.exists():
        logger.warning("No settlements.json — online GNN disabled")
        return

    settlements = json.loads(settlements_path.read_text())
    node_ids = sorted(s["id"] for s in settlements)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # Identify target nodes
    target_ids = [
        s["id"] for s in settlements if s.get("is_polymarket_target", False)
    ]
    target_indices = [node_to_idx[tid] for tid in target_ids]
    cfg.model.n_targets = len(target_indices)

    # Build adjacency
    adj = np.eye(n_nodes, dtype=np.float32)
    if edges_path.exists():
        edges = json.loads(edges_path.read_text())
        for e in edges:
            src, tgt = e["source"], e["target"]
            if src in node_to_idx and tgt in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[tgt]
                adj[i, j] = adj[j, i] = 1.0
        # Normalize
        deg = adj.sum(axis=1)
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj = adj * dinv[:, None] * dinv[None, :]

    # Resolve condition IDs for targets
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        for tid in target_ids:
            info = mapping.get(tid, {})
            cid = info.get("condition_id", "")
            if cid:
                _target_cids.append(cid)
                _target_names.append(tid)

    # Try to resolve from ClickHouse if mapping incomplete
    if len(_target_cids) < len(target_ids):
        try:
            slugs = []
            for tid in target_ids:
                info = mapping.get(tid, {})
                slug = info.get("slug", "")
                if slug:
                    slugs.append((tid, slug))
            if slugs:
                placeholders = ", ".join(f"'{s}'" for _, s in slugs)
                rows = client.query(
                    f"SELECT market_slug, condition_id FROM markets FINAL "
                    f"WHERE market_slug IN ({placeholders})"
                ).result_rows
                slug_to_cid = {r[0]: r[1] for r in rows}
                _target_cids = []
                _target_names = []
                for tid, slug in slugs:
                    cid = slug_to_cid.get(slug, "")
                    if cid:
                        _target_cids.append(cid)
                        _target_names.append(tid)
        except Exception as e:
            logger.warning("Could not resolve condition IDs: %s", e)

    checkpoint_dir = str(PROJECT_ROOT / "network" / "gnn" / "checkpoints")

    _learner = OnlineLearner(
        cfg=cfg,
        online_cfg=online_cfg,
        n_nodes=n_nodes,
        target_indices=target_indices,
        adj=adj,
        checkpoint_dir=checkpoint_dir,
    )
    _learner.warm_start()

    _extractor = FeatureExtractor(client, cfg.features)
    _initialized = True

    logger.info(
        "Online GNN initialized: %d nodes, %d targets, warm=%s",
        n_nodes, len(target_indices), _learner.is_warm,
    )


async def run_online_gnn_predict() -> None:
    """Run GNN-TCN prediction cycle using online model. Every 5 min."""
    if not _initialized:
        await _initialize()
    if _learner is None or _extractor is None:
        return

    client = _get_read_client()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        # Extract features for all nodes
        n_nodes = _learner.n_nodes
        features = np.zeros(
            (n_nodes, _learner.cfg.features.window_size, _learner.cfg.features.n_features),
            dtype=np.float32,
        )

        for i, cid in enumerate(_target_cids):
            if i >= len(_learner.target_indices):
                break
            idx = _learner.target_indices[i]
            try:
                features[idx] = await asyncio.to_thread(
                    _extractor.extract, cid, "", now,
                )
            except Exception as e:
                logger.debug("Feature extraction failed for %s: %s", cid, e)

        # Run prediction
        predictions = _learner.predict(features, _target_cids)

        # Get current market prices
        prices = await _fetch_market_prices(client, _target_cids)

        # Write to gnn_predictions table
        rows = []
        for cid, pred in predictions.items():
            mkt_price = prices.get(cid, 0.5)
            edge = pred.calibrated_prob - mkt_price
            direction = "BUY" if edge > 0.02 else ("SELL" if edge < -0.02 else "HOLD")
            confidence = min(abs(edge) / 0.15, 1.0) if not pred.is_cold_start else 0.0

            rows.append([
                cid,                         # condition_id
                "",                          # settlement_id
                "",                          # market_slug
                pred.raw_logit,              # raw_logit
                pred.calibrated_prob,        # calibrated_prob
                mkt_price,                   # market_price
                edge,                        # edge
                direction,                   # direction
                0.0,                         # kelly_fraction (computed by Bayesian layer)
                0.0,                         # position_size_usd
                0.0,                         # hurdle_rate
                "online-v1",                 # model_version
                _learner.cfg.features.window_size,
                _learner.cfg.features.step_minutes,
                confidence,
                "{}",                        # top_features
                now,                         # predicted_at
                now,                         # target_time
            ])

        if rows:
            await writer.write("gnn_predictions", rows)
            await writer.flush_all()
            logger.info("Online GNN: %d predictions written", len(rows))

    except Exception:
        logger.error("online_gnn_predict_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def run_online_gnn_update() -> None:
    """Run incremental GNN-TCN learning cycle. Every 15 min."""
    if not _initialized:
        await _initialize()
    if _learner is None or _extractor is None:
        return

    client = _get_read_client()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        # Build training samples from recent data
        # Each sample: (N_nodes, window_size, n_features) + (n_targets,) labels
        n_nodes = _learner.n_nodes
        window = _learner.cfg.features.window_size
        n_features = _learner.cfg.features.n_features
        n_targets = len(_learner.target_indices)

        features = np.zeros(
            (1, n_nodes, window, n_features), dtype=np.float32,
        )
        labels = np.zeros((1, n_targets), dtype=np.float32)

        # Extract current features
        for i, cid in enumerate(_target_cids):
            if i >= len(_learner.target_indices):
                break
            idx = _learner.target_indices[i]
            try:
                features[0, idx] = await asyncio.to_thread(
                    _extractor.extract, cid, "", now,
                )
            except Exception:
                pass

        # Labels = current market prices (MSE target)
        prices = await _fetch_market_prices(client, _target_cids)
        for i, cid in enumerate(_target_cids):
            if i >= n_targets:
                break
            labels[0, i] = prices.get(cid, 0.5)

        # Run incremental update
        result = _learner.update(features, labels)

        # Write metrics to online_learning_state
        if result.n_gradient_steps > 0:
            metrics = _learner.get_metrics()
            rows = [[
                _learner.n_updates,            # update_id
                "online-v1",                   # model_version
                result.n_samples,              # n_samples
                result.n_gradient_steps,       # n_gradient_steps
                result.avg_loss,               # avg_loss
                result.max_grad_norm,          # max_grad_norm
                result.learning_rate,          # learning_rate
                result.ema_decay_used,         # ema_decay
                metrics.get("platt_a", 1.0),   # platt_a
                metrics.get("platt_b", 0.0),   # platt_b
                now,                           # updated_at
            ]]
            await writer.write("online_learning_state", rows)
            await writer.flush_all()

            logger.info(
                "Online GNN update #%d: %d steps, loss=%.6f",
                _learner.n_updates, result.n_gradient_steps, result.avg_loss,
            )

    except Exception:
        logger.error("online_gnn_update_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def _fetch_market_prices(
    client, condition_ids: list[str]
) -> dict[str, float]:
    """Get current prices for target markets."""
    if not condition_ids:
        return {}
    try:
        placeholders = ", ".join(f"'{c}'" for c in condition_ids)
        rows = await asyncio.to_thread(
            client.query,
            f"""SELECT condition_id, price
                FROM market_latest_price FINAL
                WHERE condition_id IN ({placeholders}) AND outcome = 'Yes'""",
        )
        return {r[0]: float(r[1]) for r in rows.result_rows}
    except Exception as e:
        logger.warning("Could not fetch market prices: %s", e)
        return {}
