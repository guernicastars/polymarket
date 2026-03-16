"""
Keynesian loss: L_K = L(y, y_hat) + mu * V^{-1}

The first term pushes accuracy.
The second term pushes the agent to SEEK MORE EVIDENCE (increase V),
which means directing attention toward data-rich regions or requesting
more data in data-poor regions.

The gradient decomposes as:
    grad_theta L_K = grad_theta L(y, y_hat) - mu * V^{-2} * grad_theta V

The second term is the "evidence-seeking gradient":
it adjusts theta to increase V, not just to reduce prediction error.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def keynesian_loss(
    y_true: Tensor,
    y_pred: Tensor,
    evidence: Tensor,
    mu: float = 0.1,
    base_loss: str = "mse",
) -> Tensor:
    """
    Keynesian loss combining prediction accuracy with evidence seeking.

    Args:
        y_true: Ground truth targets. Shape: (batch, output_dim) or (batch,).
        y_pred: Model predictions. Shape: same as y_true.
        evidence: Weight of evidence V_i(x) per sample. Shape: (batch,). Must be > 0.
        mu: Tradeoff hyperparameter. Higher mu = stronger evidence seeking.
        base_loss: "mse" or "bce" for the prediction loss component.

    Returns:
        Scalar loss value.
    """
    if base_loss == "mse":
        prediction_loss = F.mse_loss(y_pred, y_true, reduction="none")
    elif base_loss == "bce":
        prediction_loss = F.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction="none"
        )
    else:
        raise ValueError(f"Unknown base_loss: {base_loss}")

    # Reduce over output dimensions if needed
    if prediction_loss.dim() > 1:
        prediction_loss = prediction_loss.mean(dim=-1)  # (batch,)

    # Evidence penalty: mu / V
    evidence_penalty = mu / (evidence + 1e-8)

    return (prediction_loss + evidence_penalty).mean()


class KeynesianLossFunction:
    """
    Stateful wrapper for Keynesian loss with configurable parameters.

    Supports scheduling mu over training (e.g., warm up evidence-seeking
    after initial convergence of the prediction loss).
    """

    def __init__(
        self,
        mu: float = 0.1,
        base_loss: str = "mse",
        mu_schedule: str = "constant",
        mu_warmup_steps: int = 0,
    ):
        self.mu = mu
        self.base_loss = base_loss
        self.mu_schedule = mu_schedule
        self.mu_warmup_steps = mu_warmup_steps
        self._step = 0

    def get_mu(self) -> float:
        if self.mu_schedule == "constant":
            return self.mu
        elif self.mu_schedule == "warmup":
            if self._step < self.mu_warmup_steps:
                return self.mu * (self._step / max(self.mu_warmup_steps, 1))
            return self.mu
        elif self.mu_schedule == "cosine":
            if self.mu_warmup_steps > 0:
                import math
                phase = min(self._step / self.mu_warmup_steps, 1.0)
                return self.mu * 0.5 * (1.0 + math.cos(math.pi * (1.0 - phase)))
            return self.mu
        return self.mu

    def __call__(
        self, y_true: Tensor, y_pred: Tensor, evidence: Tensor
    ) -> Tensor:
        loss = keynesian_loss(
            y_true, y_pred, evidence, mu=self.get_mu(), base_loss=self.base_loss
        )
        self._step += 1
        return loss

    def reset(self):
        self._step = 0
