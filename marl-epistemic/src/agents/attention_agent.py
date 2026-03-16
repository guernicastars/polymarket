"""Attention agent -- good at long-range dependencies, less efficient on local patterns."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseAgent


class AttentionModel(nn.Module):
    """
    Single-head self-attention + FFN.
    Treats each input feature as a token, enabling global interactions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 32,
        ff_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Project each feature to embed_dim (each feature is a "token")
        self.token_embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)

        # Single-head self-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Output head: pool over tokens then project
        self.output_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        x = x.view(batch, -1)  # (batch, input_dim)
        seq_len = x.shape[1]

        # Token embedding: (batch, seq_len, 1) -> (batch, seq_len, embed_dim)
        tokens = self.token_embed(x.unsqueeze(-1))
        tokens = tokens + self.pos_embed[:, :seq_len, :]

        # Self-attention
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        scale = math.sqrt(self.embed_dim)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        attended = torch.bmm(attn, v)

        # Residual + norm
        tokens = self.norm1(tokens + attended)

        # FFN + residual + norm
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Mean pool over token dimension
        pooled = tokens.mean(dim=1)  # (batch, embed_dim)
        return self.output_head(pooled)


class AttentionAgent(BaseAgent):
    """
    Attention agent: single-head self-attention over input features.

    Hypothesis class H_Attention treats each input feature as a token
    and computes pairwise interactions via attention. This enables
    capturing long-range dependencies (x[0] interacting with x[d-1])
    that CNNs miss, but the O(d^2) attention is less parameter-efficient
    than CNN weight sharing for local patterns.

    Blind spot B_Attention: struggles with purely local patterns that
    CNNs capture efficiently (local smoothness, translation equivariance),
    and can overfit small datasets due to higher flexibility.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 32,
        ff_dim: int = 64,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hypothesis_class_name="attention",
            **kwargs,
        )
        self.build_model()

    def build_model(self) -> nn.Module:
        self._model = AttentionModel(
            self.input_dim,
            self.output_dim,
            self.embed_dim,
            self.ff_dim,
            self.dropout_rate,
        ).to(self.device)
        return self._model

    def predict(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device))
