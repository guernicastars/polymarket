"""CNN agent -- good at local patterns, blind to long-range dependencies."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseAgent


class CNN1DModel(nn.Module):
    """
    1D CNN with pooling. Captures local patterns via convolution kernels
    but has limited receptive field for long-range dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # Treat input as 1D sequence with 1 channel
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(n_channels, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.view(x.shape[0], -1).unsqueeze(1)
        h = self.conv_net(x)  # (batch, n_channels, 1)
        h = h.squeeze(-1)  # (batch, n_channels)
        return self.fc(h)


class CNNAgent(BaseAgent):
    """
    CNN agent: 1D convolutions with local receptive field.

    Hypothesis class H_CNN captures local patterns efficiently via
    weight sharing and translation equivariance. However, without
    sufficient depth, it cannot capture global/long-range dependencies.

    Blind spot B_CNN contains functions that depend on interactions
    between distant positions in the input (e.g., x[0] * x[d-1]).
    This is provably different from H_MLP's function space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        **kwargs,
    ):
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hypothesis_class_name="cnn",
            **kwargs,
        )
        self.build_model()

    def build_model(self) -> nn.Module:
        self._model = CNN1DModel(
            self.input_dim,
            self.output_dim,
            self.n_channels,
            self.kernel_size,
            self.dropout_rate,
        ).to(self.device)
        return self._model

    def predict(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device))
