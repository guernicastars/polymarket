"""Transformer-based Orthogonal Supervised VAE (T-OSVAE).

Drop-in replacement for MarketAutoencoder (Orth-SVAE) with a Transformer
encoder backbone instead of an MLP encoder. Designed for structured feature
groups where each group is tokenized independently and processed with
multi-head self-attention.

Architecture:
    1. GroupFeatureTokenizer: Each feature group -> Linear(group_size -> d_token).
       Learnable [CLS] token. Learnable position embeddings.
    2. TransformerEncoder: 2 layers, 4 heads, pre-norm, GELU, dropout=0.1.
    3. CLS pooling -> Linear(d_token -> hidden) -> BN -> ReLU -> Dropout.
    4. VAE heads: fc_mu, fc_logvar -> d_emb. Reparameterization trick.
    5. Decoder: MLP d_emb -> 128 -> 256 -> D_input (reconstructs flat features).
    6. Prediction head: Linear(d_emb -> 1) on mu.

Loss (4 components, same as Orth-SVAE):
    L = L_recon + beta * L_KL + alpha * L_pred + gamma * L_orth

    L_recon: MSE reconstruction
    L_KL:    KL divergence
    L_pred:  BCE (classification) or MSE (regression)
    L_orth:  batch Pearson correlation penalty on mu
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TransformerAutoencoderConfig:
    """Configuration for T-OSVAE architecture and training."""

    input_dim: int
    embedding_dim: int = 32
    d_token: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    bottleneck_hidden: int = 128

    beta: float = 1.0    # KL weight
    alpha: float = 1.0   # Prediction loss weight
    gamma: float = 0.1   # Orthogonality regularization weight

    task: Literal["classification", "regression"] = "classification"

    feature_groups: dict[str, list[int]] = field(default_factory=dict)


class GroupFeatureTokenizer(nn.Module):
    """Tokenizes structured feature groups into a sequence for the Transformer.

    Each feature group (e.g. "artist" with 6 features, "physical" with 7)
    gets its own Linear projection to d_token dimensions. A learnable [CLS]
    token is prepended. Learnable position embeddings are added.

    Input:  (batch, input_dim) flat feature vector
    Output: (batch, 1 + n_groups, d_token) token sequence
    """

    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        d_token: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_groups = feature_groups
        self.group_names = sorted(feature_groups.keys())
        n_groups = len(self.group_names)

        # Per-group linear projections
        self.projections = nn.ModuleDict()
        for name in self.group_names:
            group_size = len(feature_groups[name])
            self.projections[name] = nn.Linear(group_size, d_token)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Learnable position embeddings: [CLS] + n_groups
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + n_groups, d_token) * 0.02
        )

        self.dropout = nn.Dropout(dropout)
        self.seq_len = 1 + n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize flat features into a sequence.

        Args:
            x: (batch, input_dim) flat feature vector.

        Returns:
            (batch, 1 + n_groups, d_token) token sequence with [CLS] at position 0.
        """
        batch_size = x.shape[0]
        tokens = []

        for name in self.group_names:
            indices = self.feature_groups[name]
            group_feats = x[:, indices]  # (batch, group_size)
            token = self.projections[name](group_feats)  # (batch, d_token)
            tokens.append(token.unsqueeze(1))  # (batch, 1, d_token)

        # Concatenate group tokens: (batch, n_groups, d_token)
        group_tokens = torch.cat(tokens, dim=1)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_token)
        sequence = torch.cat([cls, group_tokens], dim=1)  # (batch, 1+n_groups, d_token)

        # Add position embeddings
        sequence = sequence + self.pos_embedding
        sequence = self.dropout(sequence)

        return sequence


class TransformerAutoencoder(nn.Module):
    """Transformer-based Orthogonal Supervised VAE (T-OSVAE).

    Drop-in replacement for MarketAutoencoder. Same forward/loss/encode interface.

    Usage:
        config = TransformerAutoencoderConfig(
            input_dim=65,
            embedding_dim=32,
            feature_groups={"artist": [0,1,2,3,4,5], "physical": [6,7,...], ...},
        )
        model = TransformerAutoencoder(config)

        # Training (same interface as MarketAutoencoder)
        x_hat, z, loss_dict = model(x_batch, y_batch)

        # Inference
        embeddings = model.encode(x_batch)  # (batch, embedding_dim)

        # Interpretability
        attn_weights = model.get_attention_weights(x_batch)
    """

    def __init__(self, config: TransformerAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Group feature tokenizer
        self.tokenizer = GroupFeatureTokenizer(
            feature_groups=config.feature_groups,
            d_token=config.d_token,
            dropout=config.dropout,
        )

        # 2. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_token,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # 3. CLS -> Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(config.d_token, config.bottleneck_hidden),
            nn.BatchNorm1d(config.bottleneck_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # 4. VAE heads
        self.fc_mu = nn.Linear(config.bottleneck_hidden, config.embedding_dim)
        self.fc_logvar = nn.Linear(config.bottleneck_hidden, config.embedding_dim)

        # 5. Decoder (MLP: d_emb -> 128 -> 256 -> input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.input_dim),
        )

        # 6. Prediction head
        self.prediction_head = nn.Linear(config.embedding_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _correlation_penalty(z: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality penalty: ||corr(Z) - I||_F^2.

        Same implementation as MarketAutoencoder._correlation_penalty.
        """
        z_centered = z - z.mean(dim=0, keepdim=True)
        std = z_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        z_norm = z_centered / std
        corr = (z_norm.T @ z_norm) / max(z.shape[0] - 1, 1)
        eye = torch.eye(corr.shape[0], device=corr.device)
        off_diag = corr - eye
        return (off_diag ** 2).sum()

    def _encode_transformer(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encoder pass: tokenize -> transformer -> bottleneck -> VAE heads.

        Returns:
            (z, mu, logvar) where z is the reparameterized sample.
        """
        tokens = self.tokenizer(x)           # (batch, seq_len, d_token)
        encoded = self.transformer(tokens)    # (batch, seq_len, d_token)
        cls_out = encoded[:, 0, :]            # (batch, d_token) — [CLS] token
        h = self.bottleneck(cls_out)          # (batch, bottleneck_hidden)
        mu = self.fc_mu(h)                    # (batch, embedding_dim)
        logvar = self.fc_logvar(h)            # (batch, embedding_dim)
        z = self._reparameterize(mu, logvar)  # (batch, embedding_dim)
        return z, mu, logvar

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass: encode, decode, compute losses.

        API-compatible with MarketAutoencoder.forward().

        Args:
            x: Input tensor of shape (batch, input_dim).
            y: Target labels of shape (batch,) for supervised mode.

        Returns:
            x_hat: Reconstructed input, shape (batch, input_dim).
            z: Embedding tensor (reparameterized sample), shape (batch, embedding_dim).
            losses: Dict with 'recon_loss', 'kl_loss', 'pred_loss', 'orth_loss',
                    'total_loss'.
        """
        z, mu, logvar = self._encode_transformer(x)
        x_hat = self.decoder(z)

        # L_recon
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")

        # L_KL
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.config.beta * kl_loss

        losses: dict[str, torch.Tensor] = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

        # L_pred
        if y is not None:
            pred_logits = self.prediction_head(mu).squeeze(-1)
            if self.config.task == "classification":
                pred_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, y.float(), reduction="mean",
                )
            else:
                pred_loss = nn.functional.mse_loss(
                    pred_logits, y.float(), reduction="mean",
                )
            losses["pred_loss"] = pred_loss
            losses["total_loss"] = losses["total_loss"] + self.config.alpha * pred_loss

        # L_orth
        if self.config.gamma > 0 and mu.shape[0] > 1:
            orth_loss = self._correlation_penalty(mu)
            losses["orth_loss"] = orth_loss
            losses["total_loss"] = losses["total_loss"] + self.config.gamma * orth_loss

        return x_hat, z, losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space (deterministic mu).

        API-compatible with MarketAutoencoder.encode().
        """
        _z, mu, _logvar = self._encode_transformer(x)
        return mu

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict outcome from input.

        Returns:
            Classification: probability tensor of shape (batch,).
            Regression: raw prediction tensor of shape (batch,).
        """
        mu = self.encode(x)
        logits = self.prediction_head(mu).squeeze(-1)
        if self.config.task == "classification":
            return torch.sigmoid(logits)
        return logits

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embedding back to input space."""
        return self.decoder(z)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> np.ndarray:
        """Extract embeddings as numpy array (inference mode).

        API-compatible with MarketAutoencoder.get_embedding().
        """
        self.eval()
        z = self.encode(x)
        return z.cpu().numpy()

    @torch.no_grad()
    def get_attention_weights(
        self, x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass returning attention weight matrices for interpretability.

        Registers forward hooks on each layer's MultiheadAttention module to
        capture attention weights during the forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (batch, n_heads, seq_len, seq_len).
        """
        self.eval()
        attn_weights: list[torch.Tensor] = []
        hooks = []

        def make_hook(storage: list[torch.Tensor]):
            def hook_fn(module, args, output):
                # MultiheadAttention returns (attn_output, attn_weights)
                # when need_weights=True. We capture the weights.
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    storage.append(output[1].detach())
            return hook_fn

        # Temporarily set need_weights on each self_attn module
        originals = []
        for layer in self.transformer.layers:
            layer_storage: list[torch.Tensor] = []
            attn_weights.append(layer_storage)  # type: ignore[arg-type]

            # Register hook on the self_attn (MultiheadAttention) module
            hook = layer.self_attn.register_forward_hook(make_hook(layer_storage))
            hooks.append(hook)

        # Monkey-patch self_attn forward to request weights
        patched_originals = []
        for layer in self.transformer.layers:
            orig = layer.self_attn.forward
            patched_originals.append(orig)

            def make_patched(original):
                def patched_forward(*a, **kw):
                    kw["need_weights"] = True
                    kw["average_attn_weights"] = False
                    return original(*a, **kw)
                return patched_forward

            layer.self_attn.forward = make_patched(orig)

        try:
            tokens = self.tokenizer(x)
            self.transformer(tokens)
        finally:
            # Remove hooks and restore original forwards
            for hook in hooks:
                hook.remove()
            for layer, orig in zip(self.transformer.layers, patched_originals):
                layer.self_attn.forward = orig

        # Flatten: attn_weights is currently list of lists, extract first element
        result = []
        for layer_storage in attn_weights:
            if isinstance(layer_storage, list) and layer_storage:
                result.append(layer_storage[0])
            elif isinstance(layer_storage, torch.Tensor):
                result.append(layer_storage)
        return result

    @property
    def group_names(self) -> list[str]:
        """Sorted feature group names (matches token ordering)."""
        return self.tokenizer.group_names
