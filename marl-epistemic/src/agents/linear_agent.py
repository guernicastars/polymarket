"""Linear agent -- smallest hypothesis class H, largest blind spot."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseAgent


class LinearModel(nn.Module):
    """y = Wx + b. Cannot represent any nonlinear function."""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        return self.linear(x)


class LinearAgent(BaseAgent):
    """
    Linear agent: y = Wx + b.

    Hypothesis class H_linear is the set of all affine functions R^d -> R^k.
    Blind spot B_linear contains ALL nonlinear functions -- XOR, any polynomial
    of degree > 1, any function with interactions between features.

    This is the weakest agent, serving as the baseline. Its blind spot is the
    largest, but it excels in truly linear regions where other agents may overfit.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hypothesis_class_name="linear",
            **kwargs,
        )
        self.build_model()

    def build_model(self) -> nn.Module:
        self._model = LinearModel(
            self.input_dim, self.output_dim, self.dropout_rate
        ).to(self.device)
        return self._model

    def predict(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device))
