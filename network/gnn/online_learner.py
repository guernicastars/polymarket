"""Online/incremental GNN-TCN learner with EMA shadow model.

Wraps the batch-trained GNNTCN with:
  - Incremental SGD updates on recent data (every 15 min)
  - Exponential Moving Average shadow model for stable inference
  - Feature masking for unreliable features (F8-F10)
  - Graceful cold-start (returns 0.5 when no checkpoint)
  - Automatic Platt recalibration as data accumulates
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import GNNConfig, OnlineLearningConfig, ModelConfig
from .model import GNNTCN, PlattScaling

logger = logging.getLogger(__name__)


@dataclass
class OnlinePrediction:
    """Single prediction from the online GNN-TCN model."""

    condition_id: str
    raw_logit: float
    calibrated_prob: float
    model_uncertainty: float
    features_used: int
    is_cold_start: bool


@dataclass
class OnlineUpdateResult:
    """Result of an incremental learning step."""

    n_samples: int
    n_gradient_steps: int
    avg_loss: float
    max_grad_norm: float
    learning_rate: float
    ema_decay_used: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class OnlineLearner:
    """Online GNN-TCN with EMA shadow model for stable inference.

    Maintains two copies of the model:
      - live_model: updated by incremental SGD
      - ema_model: exponential moving average of live (used for predictions)

    The EMA model provides stability — even if a single SGD update is bad,
    the shadow model barely changes (0.5% weight from bad update at decay=0.995).
    """

    def __init__(
        self,
        cfg: GNNConfig,
        online_cfg: OnlineLearningConfig,
        n_nodes: int,
        target_indices: list[int],
        adj: np.ndarray,
        checkpoint_dir: str = "network/gnn/checkpoints",
    ) -> None:
        self.cfg = cfg
        self.online_cfg = online_cfg
        self.n_nodes = n_nodes
        self.target_indices = target_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Adjacency matrix (pre-normalized)
        self.adj = torch.tensor(adj, dtype=torch.float32, device=self.device)

        # Models
        self.live_model = GNNTCN(
            cfg=cfg.model, n_nodes=n_nodes, target_indices=target_indices
        ).to(self.device)
        self.ema_model = GNNTCN(
            cfg=cfg.model, n_nodes=n_nodes, target_indices=target_indices
        ).to(self.device)

        # Platt scaling
        self.platt = PlattScaling(n_targets=len(target_indices)).to(self.device)

        # Optimizer (initialized after warm_start)
        self.optimizer: Optional[torch.optim.AdamW] = None

        # Tracking
        self.n_updates = 0
        self.is_warm = False
        self._batch_checkpoint_hash: Optional[str] = None

        # Recalibration buffer: stores (logit, label) pairs
        self._recal_buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=online_cfg.recalibration_buffer_size
        )

    def warm_start(self) -> bool:
        """Load pretrained checkpoint if available. Returns True if loaded."""
        model_path = self.checkpoint_dir / "best.pt"

        # Check for online checkpoint first (resume from previous session)
        online_path = self.checkpoint_dir / "online_ema.pt"
        live_path = self.checkpoint_dir / "online_live.pt"
        meta_path = self.checkpoint_dir / "online_meta.json"

        # Check if batch checkpoint changed (triggers full reload)
        new_hash = self._file_hash(model_path) if model_path.exists() else None
        batch_changed = (
            new_hash is not None
            and self._batch_checkpoint_hash is not None
            and new_hash != self._batch_checkpoint_hash
        )

        if batch_changed:
            logger.info("Batch checkpoint changed — reinitializing from best.pt")
            self._load_batch_checkpoint(model_path)
            self._batch_checkpoint_hash = new_hash
            self.is_warm = True
            return True

        # Try to resume online state
        if online_path.exists() and live_path.exists() and not batch_changed:
            try:
                self.ema_model.load_state_dict(
                    torch.load(online_path, map_location=self.device, weights_only=True)
                )
                self.live_model.load_state_dict(
                    torch.load(live_path, map_location=self.device, weights_only=True)
                )
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self.n_updates = meta.get("n_updates", 0)
                    self._batch_checkpoint_hash = meta.get("batch_checkpoint_hash")
                self._init_optimizer()
                # Load optimizer state if exists
                opt_path = self.checkpoint_dir / "online_optimizer.pt"
                if opt_path.exists():
                    self.optimizer.load_state_dict(
                        torch.load(opt_path, map_location=self.device, weights_only=True)
                    )
                self.is_warm = True
                logger.info("Resumed online state (n_updates=%d)", self.n_updates)
                return True
            except Exception as e:
                logger.warning("Failed to resume online state: %s", e)

        # Fall back to batch checkpoint
        if model_path.exists():
            self._load_batch_checkpoint(model_path)
            self._batch_checkpoint_hash = new_hash
            self.is_warm = True
            return True

        # No checkpoint at all — cold start
        self._init_optimizer()
        self.is_warm = False
        logger.warning("No checkpoint found — cold start with random weights")
        return False

    def predict(
        self,
        features: np.ndarray,
        condition_ids: list[str],
    ) -> dict[str, OnlinePrediction]:
        """Run inference using EMA shadow model.

        Args:
            features: (N_nodes, window_size, n_features) array
            condition_ids: list of condition IDs corresponding to target indices

        Returns:
            dict mapping condition_id → OnlinePrediction
        """
        if not self.is_warm:
            return {
                cid: OnlinePrediction(
                    condition_id=cid,
                    raw_logit=0.0,
                    calibrated_prob=self.online_cfg.cold_start_prob,
                    model_uncertainty=1.0,
                    features_used=0,
                    is_cold_start=True,
                )
                for cid in condition_ids
            }

        # Apply feature mask (zero out unreliable features)
        masked = features.copy()
        all_features = set(range(self.cfg.features.n_features))
        active = set(self.online_cfg.feature_mask)
        for fi in all_features - active:
            masked[:, :, fi] = 0.0

        x = torch.tensor(masked, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.ema_model.eval()
        self.platt.eval()
        with torch.no_grad():
            logits = self.ema_model(x, self.adj)  # (1, n_targets)
            calibrated = self.platt(logits.cpu()).squeeze(0).numpy()
            raw = logits.cpu().squeeze(0).numpy()

        results = {}
        for i, cid in enumerate(condition_ids):
            if i >= len(raw):
                break
            results[cid] = OnlinePrediction(
                condition_id=cid,
                raw_logit=float(raw[i]),
                calibrated_prob=float(np.clip(calibrated[i], 0.01, 0.99)),
                model_uncertainty=0.1,  # TODO: MC dropout estimate
                features_used=len(self.online_cfg.feature_mask),
                is_cold_start=False,
            )

        return results

    def update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> OnlineUpdateResult:
        """Perform incremental SGD steps on recent data.

        Args:
            features: (N_samples, N_nodes, window_size, n_features)
            labels: (N_samples, n_targets) — market prices as targets

        Returns:
            OnlineUpdateResult with metrics
        """
        n_samples = features.shape[0]
        if n_samples < self.online_cfg.min_samples_before_update and self.is_warm:
            return OnlineUpdateResult(
                n_samples=n_samples,
                n_gradient_steps=0,
                avg_loss=0.0,
                max_grad_norm=0.0,
                learning_rate=self.online_cfg.online_lr,
                ema_decay_used=self.online_cfg.ema_decay,
            )

        if self.optimizer is None:
            self._init_optimizer()

        # Apply feature mask
        masked = features.copy()
        all_features = set(range(self.cfg.features.n_features))
        active = set(self.online_cfg.feature_mask)
        for fi in all_features - active:
            masked[:, :, :, fi] = 0.0

        x = torch.tensor(masked, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        # Determine number of gradient steps
        max_steps = self.online_cfg.max_gradient_steps
        if not self.is_warm:
            max_steps = self.online_cfg.cold_start_max_steps

        self.live_model.train()
        criterion = nn.MSELoss()
        total_loss = 0.0
        max_grad = 0.0
        steps = 0

        for step in range(min(max_steps, n_samples)):
            # Mini-batch: single sample at a time for online update
            idx = step % n_samples
            x_batch = x[idx : idx + 1]
            y_batch = y[idx : idx + 1]

            self.optimizer.zero_grad()
            logits = self.live_model(x_batch, self.adj)
            preds = torch.sigmoid(logits)
            loss = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.live_model.parameters(), self.online_cfg.online_grad_clip
            )
            max_grad = max(max_grad, grad_norm.item())

            self.optimizer.step()
            total_loss += loss.item()
            steps += 1

            # Store for recalibration
            with torch.no_grad():
                self._recal_buffer.append(
                    (logits.cpu().numpy().flatten(), y_batch.cpu().numpy().flatten())
                )

        # EMA update
        self._ema_update()
        self.n_updates += 1
        self.is_warm = True

        # Periodic Platt recalibration
        if self.n_updates % self.online_cfg.platt_recalibrate_interval == 0:
            self._recalibrate_platt()

        # Periodic checkpoint
        if self.n_updates % self.online_cfg.checkpoint_save_interval == 0:
            self.save_checkpoint()

        avg_loss = total_loss / max(steps, 1)
        lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            "Online update #%d: %d steps, loss=%.6f, grad=%.4f, lr=%.2e",
            self.n_updates, steps, avg_loss, max_grad, lr,
        )

        return OnlineUpdateResult(
            n_samples=n_samples,
            n_gradient_steps=steps,
            avg_loss=avg_loss,
            max_grad_norm=max_grad,
            learning_rate=lr,
            ema_decay_used=self.online_cfg.ema_decay,
        )

    def save_checkpoint(self) -> str:
        """Save live model, EMA model, optimizer, Platt params, and metadata."""
        torch.save(
            self.live_model.state_dict(), self.checkpoint_dir / "online_live.pt"
        )
        torch.save(
            self.ema_model.state_dict(), self.checkpoint_dir / "online_ema.pt"
        )
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                self.checkpoint_dir / "online_optimizer.pt",
            )
        torch.save(
            {"a": self.platt.a.data, "b": self.platt.b.data},
            self.checkpoint_dir / "online_platt.pt",
        )

        meta = {
            "n_updates": self.n_updates,
            "batch_checkpoint_hash": self._batch_checkpoint_hash,
            "last_save": datetime.now(timezone.utc).isoformat(),
        }
        (self.checkpoint_dir / "online_meta.json").write_text(json.dumps(meta))

        logger.info("Saved online checkpoint (n_updates=%d)", self.n_updates)
        return str(self.checkpoint_dir)

    def get_metrics(self) -> dict:
        """Return current online learning metrics."""
        return {
            "n_updates": self.n_updates,
            "is_warm": self.is_warm,
            "ema_decay": self.online_cfg.ema_decay,
            "recal_buffer_size": len(self._recal_buffer),
            "platt_a": self.platt.a.data.mean().item(),
            "platt_b": self.platt.b.data.mean().item(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_batch_checkpoint(self, model_path: pathlib.Path) -> None:
        """Load batch-trained checkpoint into both live and EMA models."""
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.live_model.load_state_dict(state)
        self.ema_model.load_state_dict(state)

        # Load Platt scaling if available
        platt_path = self.checkpoint_dir / "platt.pt"
        if platt_path.exists():
            platt_data = torch.load(
                platt_path, map_location=self.device, weights_only=True
            )
            self.platt.a.data = platt_data["a"]
            self.platt.b.data = platt_data["b"]

        self._init_optimizer()
        self.n_updates = 0
        logger.info("Loaded batch checkpoint from %s", model_path)

    def _init_optimizer(self) -> None:
        """Initialize AdamW optimizer for online updates."""
        self.optimizer = torch.optim.AdamW(
            self.live_model.parameters(),
            lr=self.online_cfg.online_lr,
            weight_decay=self.online_cfg.online_weight_decay,
        )

    def _ema_update(self) -> None:
        """Update EMA shadow: shadow = decay * shadow + (1 - decay) * live."""
        decay = self.online_cfg.ema_decay
        with torch.no_grad():
            for ema_p, live_p in zip(
                self.ema_model.parameters(), self.live_model.parameters()
            ):
                ema_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)

    def _recalibrate_platt(self) -> None:
        """Re-fit Platt scaling on recalibration buffer."""
        if len(self._recal_buffer) < 30:
            return

        logits_list = [pair[0] for pair in self._recal_buffer]
        labels_list = [pair[1] for pair in self._recal_buffer]

        logits_t = torch.tensor(np.array(logits_list), dtype=torch.float32)
        labels_t = torch.tensor(np.array(labels_list), dtype=torch.float32)

        # Truncate to match Platt n_targets
        n_targets = self.platt.a.shape[0]
        logits_t = logits_t[:, :n_targets]
        labels_t = labels_t[:, :n_targets]

        losses = self.platt.fit(
            logits_t, labels_t, lr=0.01, epochs=50
        )
        logger.info(
            "Platt recalibrated on %d samples, final loss=%.6f",
            len(logits_list), losses[-1] if losses else 0.0,
        )

    @staticmethod
    def _file_hash(path: pathlib.Path) -> Optional[str]:
        """Compute MD5 hash of a file for change detection."""
        if not path.exists():
            return None
        h = hashlib.md5()
        h.update(path.read_bytes())
        return h.hexdigest()
