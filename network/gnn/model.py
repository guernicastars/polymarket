"""GNN-TCN model: Graph Attention + Temporal Convolution for market prediction.

Architecture (per forward pass):
  1. Input: (batch, N_nodes, window_size, 12_features)
  2. Per-timestep GAT: encodes spatial graph structure → (batch, N, W, gat_out)
  3. Per-node TCN: captures temporal patterns → (batch, N, tcn_out)
  4. Prediction head: FC → logit per Polymarket target node → (batch, n_targets)
  5. Platt scaling (post-hoc): calibrates logits to probabilities

Dilated TCN layers: receptive field grows exponentially without pooling,
exactly as Anastasiia specified — layer 1 sees every step, top layer
sees every 4th step via dilation [1, 2, 4, 8].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ============================================================
# TCN building blocks
# ============================================================

class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))  # left pad only
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Single TCN block: 2x causal conv + residual + dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))
        return F.gelu(out + self.residual(x))


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially growing dilation.

    Dilation pattern: [1, 2, 4, 8, ...] — top layer sees every 2^(L-1)-th step.
    With 4 layers and kernel=3, receptive field = 2 * (3-1) * (1+2+4+8) + 1 = 61 ≈ 64 steps.
    """

    def __init__(self, in_ch: int, channels: list[int], kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch if i == 0 else channels[i - 1], out_ch, kernel, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, seq_len) → (batch, out_channels, seq_len)"""
        return self.network(x)


# ============================================================
# Graph Attention layer (lightweight, no PyG dependency)
# ============================================================

class GraphAttention(nn.Module):
    """Multi-head graph attention (GAT) without PyTorch Geometric dependency.

    Uses dense attention over the adjacency matrix — feasible for 40-node graph.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_features % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, in_features)
            adj: (N, N) binary adjacency matrix (add self-loops before calling)
        Returns:
            (batch, N, out_features)
        """
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.head_dim)  # (B, N, H, D)

        # Attention scores
        e_src = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        e_dst = (h * self.a_dst).sum(dim=-1)

        # (B, N, H, 1) + (B, 1, H, N) → (B, N, H, N) via broadcasting
        attn = self.leaky_relu(
            e_src.unsqueeze(-1) + e_dst.unsqueeze(1).transpose(2, 3)
        )  # (B, N, H, N)

        # Mask non-edges
        mask = adj.unsqueeze(0).unsqueeze(2)  # (1, N, 1, N)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        # attn: (B, N, H, N) × h.permute: (B, N_src, H, D) → need (B, N, H, D)
        h_perm = h.permute(0, 2, 1, 3)  # (B, H, N, D)
        attn_perm = attn.permute(0, 2, 1, 3)  # (B, H, N, N)
        out = torch.matmul(attn_perm, h_perm)  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, out_features)

        return out


# ============================================================
# Full GNN-TCN Model
# ============================================================

class GNNTCN(nn.Module):
    """Graph Attention + Temporal Convolution Network for prediction markets.

    Forward pass:
      1. For each timestep t in [0, W): GAT encodes spatial features
      2. Per-node temporal sequence fed through TCN
      3. Only Polymarket target nodes (7) get prediction heads
      4. Output: raw logits (apply sigmoid + Platt scaling externally)
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, n_nodes: int = 40, target_indices: Optional[list[int]] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.n_nodes = n_nodes
        self.target_indices = target_indices or list(range(7))

        # Graph Attention
        self.gat = GraphAttention(
            in_features=self.cfg.gat_in_features,
            out_features=self.cfg.gat_out,
            n_heads=self.cfg.gat_heads,
            dropout=self.cfg.gat_dropout,
        )
        self.gat_norm = nn.LayerNorm(self.cfg.gat_out)

        # TCN (per node, shared weights)
        self.tcn = TCN(
            in_ch=self.cfg.gat_out,
            channels=self.cfg.tcn_channels,
            kernel=self.cfg.tcn_kernel_size,
            dropout=self.cfg.tcn_dropout,
        )

        # Prediction head — per target node
        tcn_out = self.cfg.tcn_channels[-1]
        self.pred_head = nn.Sequential(
            nn.Linear(tcn_out, self.cfg.fc_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.fc_dropout),
            nn.Linear(self.cfg.fc_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, N_nodes, window_size, n_features) — temporal graph snapshots
            adj: (N_nodes, N_nodes) — adjacency matrix with self-loops

        Returns:
            logits: (batch, n_targets) — raw logits for target settlements
        """
        B, N, W, F = x.shape

        # 1. Per-timestep GAT: encode spatial structure
        gat_out = []
        for t in range(W):
            h_t = self.gat(x[:, :, t, :], adj)  # (B, N, gat_out)
            h_t = self.gat_norm(h_t)
            gat_out.append(h_t)

        # Stack: (B, N, W, gat_out)
        h = torch.stack(gat_out, dim=2)

        # 2. Per-node TCN: flatten batch & nodes, reshape for Conv1d
        # (B*N, gat_out, W) — TCN expects (batch, channels, seq_len)
        h_flat = h.reshape(B * N, W, self.cfg.gat_out).permute(0, 2, 1)
        tcn_out = self.tcn(h_flat)  # (B*N, tcn_channels[-1], W)

        # Take last timestep (no future leakage)
        h_last = tcn_out[:, :, -1]  # (B*N, tcn_channels[-1])
        h_last = h_last.reshape(B, N, -1)  # (B, N, tcn_out)

        # 3. Predict only for target nodes
        target_h = h_last[:, self.target_indices, :]  # (B, n_targets, tcn_out)
        logits = self.pred_head(target_h).squeeze(-1)  # (B, n_targets)

        return logits


# ============================================================
# Platt Scaling (Post-hoc calibration — Anastasiia's Point 3)
# ============================================================

class PlattScaling(nn.Module):
    """Platt scaling: logit → calibrated probability via learned affine + sigmoid.

    Trained on held-out validation set AFTER main model is frozen.
    For Polymarket: price IS a probability, so we compare calibrated output
    to market price to find mispriced markets.
    """

    def __init__(self, n_targets: int = 7):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_targets))
        self.b = nn.Parameter(torch.zeros(n_targets))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (batch, n_targets) → calibrated probabilities (batch, n_targets)"""
        return torch.sigmoid(self.a * logits + self.b)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 200,
    ) -> list[float]:
        """Fit Platt scaling on validation data.

        Args:
            logits: (N_val, n_targets) — frozen model outputs
            targets: (N_val, n_targets) — binary labels or probabilities
            lr: learning rate
            epochs: optimization steps

        Returns:
            list of losses per epoch
        """
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=20)
        losses = []

        for _ in range(epochs):
            def closure():
                optimizer.zero_grad()
                probs = self.forward(logits)
                loss = F.binary_cross_entropy(probs, targets)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(loss.item())

        return losses
