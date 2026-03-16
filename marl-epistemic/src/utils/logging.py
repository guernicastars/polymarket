"""Weights & Biases integration for experiment tracking."""

import os
from typing import Any, Dict, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ExperimentLogger:
    """
    Wrapper around W&B for experiment tracking.

    Falls back to console logging if W&B is not available or disabled.
    """

    def __init__(
        self,
        project: str = "marl-epistemic",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        tags: Optional[list] = None,
    ):
        self.enabled = enabled and HAS_WANDB
        self._step = 0

        if self.enabled:
            wandb.init(
                project=project,
                name=experiment_name,
                config=config or {},
                tags=tags,
                reinit=True,
            )
        else:
            print(f"[Logger] Experiment: {experiment_name}")
            if config:
                for k, v in config.items():
                    print(f"  {k}: {v}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if step is None:
            step = self._step
            self._step += 1

        if self.enabled:
            wandb.log(metrics, step=step)
        else:
            items = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items())
            print(f"[Step {step}] {items}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics (shown in W&B overview)."""
        if self.enabled:
            for k, v in metrics.items():
                wandb.run.summary[k] = v
        else:
            print(f"[Summary] {metrics}")

    def finish(self):
        """Close the logger."""
        if self.enabled:
            wandb.finish()
