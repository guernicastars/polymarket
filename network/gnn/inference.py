"""Production inference pipeline — runs predictions and writes to ClickHouse.

Designed to be called from a scheduler (APScheduler) or CLI:
    python -m network.gnn.inference --checkpoint checkpoints/best.pt

Flow:
  1. Load trained model + Platt scaling params
  2. Extract latest features from ClickHouse
  3. Run forward pass → calibrated probabilities
  4. Compare to market prices → generate signals
  5. Write predictions to gnn_predictions table
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from .config import GNNConfig
from .model import GNNTCN, PlattScaling
from .backtest import kelly_criterion, DynamicHurdle
from .features import FeatureExtractor

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class InferencePipeline:
    """Production inference: model → predictions → ClickHouse."""

    def __init__(
        self,
        client,
        checkpoint_dir: str,
        config: Optional[GNNConfig] = None,
        model_version: str = "v1.0",
    ):
        self.client = client
        self.cfg = config or GNNConfig()
        self.model_version = model_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hurdle = DynamicHurdle(self.cfg.backtest)

        # Load graph info
        self.settlements = self._load_settlements()
        self.node_ids = sorted(self.settlements.keys())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        self.target_ids = [
            s["id"] for s in self.settlements.values()
            if s.get("is_polymarket_target", False)
        ]
        self.target_indices = [self.node_to_idx[tid] for tid in self.target_ids]

        # Build adjacency
        self.adj = self._build_adjacency()

        # Load market info
        self.market_info = self._load_market_info()

        # Load model
        cp_dir = pathlib.Path(checkpoint_dir)
        self.model = GNNTCN(
            cfg=self.cfg.model,
            n_nodes=len(self.node_ids),
            target_indices=self.target_indices,
        )
        model_path = cp_dir / "best.pt"
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning("No checkpoint found at %s — using random weights", model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load Platt scaling
        self.platt = PlattScaling(n_targets=len(self.target_indices))
        platt_path = cp_dir / "platt.pt"
        if platt_path.exists():
            platt_data = torch.load(platt_path, map_location=self.device, weights_only=True)
            self.platt.a.data = platt_data["a"]
            self.platt.b.data = platt_data["b"]
            logger.info("Loaded Platt scaling from %s", platt_path)
        self.platt.eval()

        self.extractor = FeatureExtractor(client, self.cfg.features)

    def run(self) -> list[dict]:
        """Execute one prediction cycle.

        Returns:
            list of prediction dicts (one per target settlement)
        """
        now = datetime.utcnow()
        logger.info("Running inference at %s", now.isoformat())

        # 1. Extract features for all nodes
        n_nodes = len(self.node_ids)
        x = np.zeros(
            (n_nodes, self.cfg.features.window_size, self.cfg.features.n_features),
            dtype=np.float32,
        )

        for sid in self.target_ids:
            idx = self.node_to_idx[sid]
            info = self.market_info.get(sid, {})
            cid = info.get("condition_id", "")
            eslug = info.get("event_slug", "")
            if cid:
                try:
                    x[idx] = self.extractor.extract(cid, eslug, now)
                except Exception as e:
                    logger.warning("Feature extraction failed for %s: %s", sid, e)

        # 2. Forward pass
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        adj_tensor = torch.tensor(self.adj, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor, adj_tensor)
            calibrated = self.platt(logits.cpu()).squeeze(0).numpy()
            raw_logits = logits.cpu().squeeze(0).numpy()

        # 3. Get current market prices
        market_prices = self._fetch_market_prices()

        # 4. Generate predictions
        predictions = []
        for i, sid in enumerate(self.target_ids):
            info = self.market_info.get(sid, {})
            cid = info.get("condition_id", "")
            slug = info.get("slug", "")
            mkt_price = market_prices.get(cid, 0.5)

            prob = float(calibrated[i])
            edge = prob - mkt_price
            direction = "BUY" if edge > 0.02 else ("SELL" if edge < -0.02 else "HOLD")
            kf = kelly_criterion(prob, mkt_price) if direction != "HOLD" else 0.0

            # Dynamic hurdle and position sizing
            liq = self._get_liquidity(cid)
            hurdle = self.hurdle.compute_hurdle(1000, liq)
            size = 0.0
            if abs(edge) > hurdle and direction != "HOLD":
                size = self.hurdle.optimal_size(edge, kf, 10_000, liq)

            pred = {
                "condition_id": cid,
                "settlement_id": sid,
                "market_slug": slug,
                "raw_logit": float(raw_logits[i]),
                "calibrated_prob": prob,
                "market_price": mkt_price,
                "edge": edge,
                "direction": direction,
                "kelly_fraction": kf,
                "position_size_usd": size,
                "hurdle_rate": hurdle,
                "model_version": self.model_version,
                "window_size": self.cfg.features.window_size,
                "step_minutes": self.cfg.features.step_minutes,
                "confidence": min(abs(edge) / 0.15, 1.0),
                "top_features": "{}",
                "predicted_at": now,
                "target_time": now,
            }
            predictions.append(pred)

            logger.info(
                "  %s: P(model)=%.3f P(market)=%.3f edge=%+.3f → %s kelly=%.3f size=$%.0f",
                sid, prob, mkt_price, edge, direction, kf, size,
            )

        # 5. Write to ClickHouse
        self._write_predictions(predictions)

        return predictions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_settlements(self) -> dict:
        path = DATA_DIR / "settlements.json"
        with open(path) as f:
            return {s["id"]: s for s in json.load(f)}

    def _build_adjacency(self) -> np.ndarray:
        n = len(self.node_ids)
        adj = np.eye(n, dtype=np.float32)
        path = DATA_DIR / "edges.json"
        with open(path) as f:
            edges = json.load(f)
        for e in edges:
            src, tgt = e["source"], e["target"]
            if src in self.node_to_idx and tgt in self.node_to_idx:
                i, j = self.node_to_idx[src], self.node_to_idx[tgt]
                adj[i, j] = adj[j, i] = 1.0
        deg = adj.sum(axis=1)
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        return adj * dinv[:, None] * dinv[None, :]

    def _load_market_info(self) -> dict:
        path = DATA_DIR / "polymarket_mapping.json"
        with open(path) as f:
            raw = json.load(f)
        info = {}
        for sid, data in raw.items():
            info[sid] = {
                "slug": data.get("slug", ""),
                "condition_id": data.get("condition_id", ""),
                "event_slug": "",
            }
        # Resolve condition IDs
        try:
            slugs = [v["slug"] for v in info.values() if v["slug"]]
            if slugs:
                placeholders = ", ".join(f"'{s}'" for s in slugs)
                rows = self.client.query(
                    f"SELECT market_slug, condition_id, event_slug FROM markets FINAL WHERE market_slug IN ({placeholders})"
                ).result_rows
                for slug, cid, eslug in rows:
                    for sid, d in info.items():
                        if d["slug"] == slug:
                            d["condition_id"] = cid
                            d["event_slug"] = eslug
        except Exception as e:
            logger.warning("Could not resolve condition IDs: %s", e)
        return info

    def _fetch_market_prices(self) -> dict[str, float]:
        """Get current prices for all PM target markets."""
        prices = {}
        cids = [v["condition_id"] for v in self.market_info.values() if v["condition_id"]]
        if not cids:
            return prices
        try:
            placeholders = ", ".join(f"'{c}'" for c in cids)
            rows = self.client.query(
                f"""SELECT condition_id, price FROM market_latest_price FINAL
                    WHERE condition_id IN ({placeholders}) AND outcome = 'Yes'"""
            ).result_rows
            for cid, price in rows:
                prices[cid] = price
        except Exception as e:
            logger.warning("Could not fetch market prices: %s", e)
        return prices

    def _get_liquidity(self, condition_id: str) -> float:
        """Get market liquidity for position sizing."""
        if not condition_id:
            return 10_000.0
        try:
            rows = self.client.query(
                "SELECT liquidity FROM markets FINAL WHERE condition_id = {cid:String}",
                parameters={"cid": condition_id},
            ).result_rows
            return rows[0][0] if rows else 10_000.0
        except Exception:
            return 10_000.0

    def _write_predictions(self, predictions: list[dict]) -> None:
        """Write predictions to ClickHouse gnn_predictions table."""
        if not predictions:
            return

        try:
            columns = [
                "condition_id", "settlement_id", "market_slug",
                "raw_logit", "calibrated_prob", "market_price", "edge",
                "direction", "kelly_fraction", "position_size_usd", "hurdle_rate",
                "model_version", "window_size", "step_minutes", "confidence",
                "top_features", "predicted_at", "target_time",
            ]
            rows = []
            for p in predictions:
                rows.append([p[c] for c in columns])

            self.client.insert(
                "gnn_predictions",
                rows,
                column_names=columns,
            )
            logger.info("Wrote %d predictions to gnn_predictions", len(rows))
        except Exception as e:
            logger.error("Failed to write predictions: %s", e)
