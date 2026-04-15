"""Abstract base class for all epistemic agents."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BaseAgent(ABC):
    """
    Each agent i has:
    - hypothesis_class_name: str identifying H_i
    - model: nn.Module (the approximator f_hat_theta_i)
    - parameters theta_i
    - A method to compute weight of evidence V_i(x)
    - A method to compute blind spot membership (empirical)

    The agent represents one formal system in the epistemic diversity framework.
    Different hypothesis classes (Linear, MLP, CNN, Attention) have provably
    different expressiveness, leading to different blind spots B_i.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hypothesis_class_name: str,
        mc_dropout_samples: int = 20,
        dropout_rate: float = 0.1,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hypothesis_class_name = hypothesis_class_name
        self.mc_dropout_samples = mc_dropout_samples
        self.dropout_rate = dropout_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._training_points: Optional[Tensor] = None
        self._calibration_scale: float = 1.0  # set via calibrate_evidence()

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        return self._model

    @property
    def parameters(self):
        return self.model.parameters()

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the nn.Module for this hypothesis class."""

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Return prediction y_hat. Shape: (batch, output_dim)."""

    def weight_of_evidence(self, x: Tensor, method: str = "mc_dropout") -> Tensor:
        """
        Compute V_i(x) -- the Keynesian weight of evidence.

        V_i is NOT prediction confidence (softmax entropy).
        V_i measures how much DATA supports the prediction at x.

        Methods:
            mc_dropout: Inverse predictive variance under MC dropout.
                V_i(x) = 1 / Var[f_hat(x; theta_i + noise)]
            kernel: Effective sample size in a neighbourhood of x.
                V_i(x) = sum of kernel weights K(x, x_j) for training points x_j

        Returns: (batch,) tensor of evidence weights, always > 0.
        """
        if method == "mc_dropout":
            return self._evidence_mc_dropout(x)
        elif method == "calibrated":
            return self._evidence_calibrated(x)
        elif method == "kernel":
            return self._evidence_kernel(x)
        else:
            raise ValueError(f"Unknown evidence method: {method}")

    def _evidence_mc_dropout(self, x: Tensor) -> Tensor:
        """MC dropout: run multiple forward passes, measure inverse variance."""
        self.model.train()  # enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                pred = self.predict(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (S, batch, output_dim)
        variance = predictions.var(dim=0).mean(dim=-1)  # (batch,)
        evidence = 1.0 / (variance + 1e-8)

        self.model.eval()
        return evidence

    def calibrate_evidence(self, x_val: Tensor, y_val: Tensor):
        """Calibrate evidence using validation accuracy.

        Scales MC dropout evidence by (1 / val_mse), so agents that are
        confidently WRONG get downweighted. Call after training, before
        using method='calibrated'.
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.predict(x_val)
            mse = (y_pred - y_val).pow(2).mean().item()
        # Scale: good models (low MSE) get high multiplier
        self._calibration_scale = 1.0 / (mse + 1e-6)

    def _evidence_calibrated(self, x: Tensor) -> Tensor:
        """MC dropout evidence scaled by validation accuracy.

        Fixes the 'confidently wrong' problem: raw MC dropout gives high
        evidence to models with low variance regardless of accuracy.
        Multiplying by 1/val_mse penalizes models that are confident but wrong.
        """
        raw_evidence = self._evidence_mc_dropout(x)
        return raw_evidence * self._calibration_scale

    def _evidence_kernel(self, x: Tensor, bandwidth: float = 1.0) -> Tensor:
        """Kernel density: effective sample size in neighbourhood of x."""
        if self._training_points is None:
            return torch.ones(x.shape[0], device=x.device)

        # x: (batch, input_dim), training_points: (N, input_dim)
        x_flat = x.view(x.shape[0], -1)
        tp_flat = self._training_points.view(self._training_points.shape[0], -1)

        # Pairwise squared distances
        dists_sq = torch.cdist(x_flat.float(), tp_flat.float()).pow(2)
        # Gaussian kernel weights
        weights = torch.exp(-dists_sq / (2.0 * bandwidth ** 2))
        evidence = weights.sum(dim=-1)  # (batch,)
        return evidence

    def blind_spot_score(self, x: Tensor, y_true: Tensor) -> Tensor:
        """
        Empirical measure of whether x is in B_i.
        High score = x is likely in the blind spot of H_i.

        After training the agent to convergence, the irreducible error
        at x reflects the expressiveness limit of the hypothesis class.
        If error is high despite sufficient data, x is in B_i.

        Returns: (batch,) tensor of blind spot scores in [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.predict(x)
            error = (y_pred - y_true).pow(2).mean(dim=-1)  # MSE per sample
            # Normalize to [0, 1] via sigmoid
            score = torch.sigmoid(error * 5.0 - 1.0)
        return score

    def store_training_points(self, x: Tensor):
        """Store training data for kernel-based evidence computation."""
        self._training_points = x.detach().clone().to(self.device)

    def setup_optimizer(self, lr: float = 1e-3, weight_decay: float = 0.0):
        """Create optimizer for this agent."""
        self._optimizer = torch.optim.Adam(
            self.parameters, lr=lr, weight_decay=weight_decay
        )
        return self._optimizer

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise RuntimeError("Optimizer not set. Call setup_optimizer() first.")
        return self._optimizer

    def to(self, device: str) -> "BaseAgent":
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        if self._training_points is not None:
            self._training_points = self._training_points.to(device)
        return self

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters)
        return (
            f"{self.__class__.__name__}("
            f"H={self.hypothesis_class_name}, "
            f"in={self.input_dim}, out={self.output_dim}, "
            f"params={n_params:,})"
        )
