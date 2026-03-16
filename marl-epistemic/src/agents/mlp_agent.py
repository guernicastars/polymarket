"""MLP agent -- medium hypothesis class, can represent piecewise linear functions."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseAgent


class MLPModel(nn.Module):
    """2-layer MLP with ReLU. Piecewise linear function space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], -1)
        return self.net(x)


class MLPAgent(BaseAgent):
    """
    MLP agent: 2-layer ReLU network.

    Hypothesis class H_MLP is the set of piecewise linear functions.
    With finite width, it cannot represent smooth functions exactly,
    and has difficulty with periodic or highly oscillatory targets.
    It can represent XOR and low-order interactions but struggles
    with spatial/sequential structure that CNN/Attention exploit.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 128, **kwargs
    ):
        self.hidden_dim = hidden_dim
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hypothesis_class_name="mlp",
            **kwargs,
        )
        self.build_model()

    def build_model(self) -> nn.Module:
        self._model = MLPModel(
            self.input_dim, self.output_dim, self.hidden_dim, self.dropout_rate
        ).to(self.device)
        return self._model

    def predict(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device))
