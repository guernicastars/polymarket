"""Patch-based encoder-only transformer for temporal market embeddings.

Architecture:
  1. Input: (batch, seq_len, 12_features) — variable-length hourly bars
  2. Patch embedding: group `patch_size` consecutive bars → linear projection → d_model
  3. Prepend [CLS] token, add relative positional encoding
  4. Transformer encoder: N layers of pre-norm multi-head self-attention + FFN
  5. CLS token output → d_model-dim embedding

Pre-training objective: Masked Patch Prediction (MPP)
  - Randomly mask `mask_ratio` of patches, replace with learnable [MASK] token
  - Predict masked patches from context via linear reconstruction head
  - Works on ALL markets (no resolution labels needed) → 76K+ samples

Design choices:
  - Patch-based (not per-timestep) reduces sequence length 24x, critical for efficiency
  - Pre-norm LayerNorm for training stability with small models
  - Relative positional encoding (fraction of market lifetime) handles variable durations
  - CLS token for fixed-size embedding extraction (standard BERT pattern)
  - Small architecture (3 layers, 4 heads, dim 64) appropriate for ~5K supervised samples
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    """Convert raw hourly features into patch tokens via linear projection.

    Input:  (batch, n_bars, n_features) — e.g. (B, 720, 12) for a 30-day market
    Output: (batch, n_patches, d_model)  — e.g. (B, 30, 64) with patch_size=24
    """

    def __init__(self, patch_size: int, n_features: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, n_patches, d_model)"""
        B, T, F = x.shape
        # Truncate to multiple of patch_size
        n_patches = T // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]
        # Reshape: (B, n_patches, patch_size * F)
        x = x.reshape(B, n_patches, self.patch_size * F)
        # Project
        return self.norm(self.proj(x))


# ============================================================
# Positional Encoding
# ============================================================

class RelativePositionalEncoding(nn.Module):
    """Positional encoding based on relative position in market lifetime.

    Instead of absolute position indices, each patch gets a position value
    in [0, 1] representing its fraction through the market's lifecycle.
    This makes the encoding invariant to absolute market duration.

    Uses sinusoidal encoding of the relative position, plus a learnable
    linear projection — best of both worlds.
    """

    def __init__(self, d_model: int, max_patches: int = 128):
        super().__init__()
        self.d_model = d_model
        # Learnable position embedding as fallback / complement
        self.pos_embed = nn.Embedding(max_patches, d_model)
        # Continuous position projection: relative_pos → d_model
        self.rel_proj = nn.Linear(1, d_model, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        n_patches: int,
        relative_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            n_patches: number of patches in sequence
            relative_positions: (B, n_patches) values in [0, 1], or None for default

        Returns:
            (1, n_patches, d_model) or (B, n_patches, d_model) positional encoding
        """
        if relative_positions is not None:
            # Continuous: project relative position to d_model
            # relative_positions: (B, n_patches) → (B, n_patches, 1)
            rp = relative_positions.unsqueeze(-1)
            return self.rel_proj(rp) * self.scale
        else:
            # Fallback: standard learnable position embedding
            positions = torch.arange(n_patches, device=self.pos_embed.weight.device)
            return self.pos_embed(positions).unsqueeze(0) * self.scale


# ============================================================
# Transformer Encoder Layer (Pre-Norm)
# ============================================================

class PreNormEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer.

    Pre-norm (LayerNorm before attention/FFN) is more stable for small models
    and doesn't require careful learning rate warmup.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, S, d_model), key_padding_mask: (B, S) True=ignore"""
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Full Transformer Encoder
# ============================================================

class MarketTransformer(nn.Module):
    """Patch-based encoder-only transformer for market time series.

    Forward pass:
      1. Patch embed: (B, T, 12) → (B, n_patches, d_model)
      2. Prepend CLS token → (B, 1 + n_patches, d_model)
      3. Add positional encoding
      4. Transformer encoder stack
      5. Extract CLS token → (B, d_model) embedding

    Returns:
      - embedding: (B, d_model) — CLS token output
      - patch_embeddings: (B, n_patches, d_model) — all patch representations
    """

    def __init__(self, cfg: Optional[TransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            self.cfg.patch_size, self.cfg.n_input_features, self.cfg.d_model,
        )

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)

        # Positional encoding
        self.pos_encoder = RelativePositionalEncoding(
            self.cfg.d_model, self.cfg.max_patches + 1,  # +1 for CLS
        )

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(
                self.cfg.d_model, self.cfg.n_heads, self.cfg.d_ff,
                self.cfg.dropout, self.cfg.attn_dropout,
            )
            for _ in range(self.cfg.n_layers)
        ])

        # Final layer norm (needed for pre-norm architecture)
        self.final_norm = nn.LayerNorm(self.cfg.d_model)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_features) — raw hourly feature bars
            padding_mask: (B, T) — True where bars are padding (not real data)
            relative_positions: (B, n_patches) — position as fraction of lifetime [0, 1]

        Returns:
            embedding: (B, d_model) — CLS token embedding
            patch_embeddings: (B, n_patches, d_model) — per-patch representations
        """
        B = x.shape[0]

        # 1. Patch embedding
        patches = self.patch_embed(x)  # (B, n_patches, d_model)
        n_patches = patches.shape[1]

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls_tokens, patches], dim=1)  # (B, 1+n_patches, d_model)

        # 3. Positional encoding
        if relative_positions is not None:
            # Add CLS position (0.0) at the front
            cls_pos = torch.zeros(B, 1, device=relative_positions.device)
            full_pos = torch.cat([cls_pos, relative_positions[:, :n_patches]], dim=1)
            pos_enc = self.pos_encoder(n_patches + 1, full_pos)
        else:
            pos_enc = self.pos_encoder(n_patches + 1)
        tokens = tokens + pos_enc

        # 4. Build attention mask if needed (for padded patches)
        key_padding_mask = None
        if padding_mask is not None:
            # Convert bar-level mask to patch-level mask
            # A patch is padding if ALL its bars are padding
            n_usable = n_patches * self.cfg.patch_size
            bar_mask = padding_mask[:, :n_usable]  # (B, n_usable)
            patch_mask = bar_mask.reshape(B, n_patches, self.cfg.patch_size).all(dim=2)
            # Prepend False for CLS (never masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, patch_mask], dim=1)

        # 5. Transformer encoder
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)
        tokens = self.final_norm(tokens)

        # 6. Extract CLS embedding and patch representations
        embedding = tokens[:, 0, :]          # (B, d_model)
        patch_embeddings = tokens[:, 1:, :]  # (B, n_patches, d_model)

        return embedding, patch_embeddings

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convenience: extract only the CLS embedding."""
        embedding, _ = self.forward(x, **kwargs)
        return embedding


# ============================================================
# Pre-training Head: Masked Patch Prediction
# ============================================================

class MaskedPatchPredictionHead(nn.Module):
    """Reconstruction head for masked patch prediction pre-training.

    Predicts the raw feature values of masked patches from their
    transformer representations. Lightweight linear head — the
    transformer does the heavy lifting.
    """

    def __init__(self, d_model: int, patch_size: int, n_features: int):
        super().__init__()
        output_dim = patch_size * n_features
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, output_dim),
        )

    def forward(self, patch_representations: torch.Tensor) -> torch.Tensor:
        """patch_representations: (B, n_masked, d_model) → (B, n_masked, patch_size * n_features)"""
        return self.head(patch_representations)


class MarketTransformerForPretraining(nn.Module):
    """Transformer + MPP head for self-supervised pre-training.

    Wraps MarketTransformer with masking logic and reconstruction head.

    Pre-training loop:
      1. Patch-embed the input
      2. Randomly mask `mask_ratio` of patches (replace with [MASK] token)
      3. Run through transformer encoder
      4. Reconstruct masked patches from their output representations
      5. Loss = MSE between predicted and original patch features
    """

    def __init__(self, cfg: Optional[TransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        self.transformer = MarketTransformer(self.cfg)
        self.mpp_head = MaskedPatchPredictionHead(
            self.cfg.d_model, self.cfg.patch_size, self.cfg.n_input_features,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_features) — raw hourly bars
            padding_mask: (B, T) — True for padding positions
            relative_positions: (B, n_patches) — lifetime fraction
            mask_ratio: override default mask ratio (for curriculum)

        Returns:
            loss: scalar MSE reconstruction loss (only on masked patches)
            embedding: (B, d_model) — CLS embedding (for monitoring)
            predictions: (B, n_masked, patch_size * n_features)
        """
        B, T, n_feat = x.shape
        ratio = mask_ratio if mask_ratio is not None else self.cfg.mask_ratio

        # 1. Create patch targets BEFORE masking
        n_patches = T // self.cfg.patch_size
        T_usable = n_patches * self.cfg.patch_size
        patch_dim = self.cfg.patch_size * n_feat
        patch_targets = x[:, :T_usable, :].reshape(
            B, n_patches, patch_dim,
        )  # (B, n_patches, patch_size * n_feat)

        # 2. Determine which patches to mask
        n_mask = max(1, int(n_patches * ratio))
        # Per-sample random mask indices
        noise = torch.rand(B, n_patches, device=x.device)
        # If we have a padding mask, don't mask padding patches
        if padding_mask is not None:
            bar_mask = padding_mask[:, :T_usable]
            patch_pad = bar_mask.reshape(B, n_patches, self.cfg.patch_size).all(dim=2)
            noise[patch_pad] = 2.0  # ensure padding patches aren't selected for masking

        mask_indices = noise.argsort(dim=1)[:, :n_mask]  # (B, n_mask)
        bool_mask = torch.zeros(B, n_patches, dtype=torch.bool, device=x.device)
        bool_mask.scatter_(1, mask_indices, True)

        # 3. Patch-embed, then replace masked patches with [MASK] token
        patches = self.transformer.patch_embed(x)  # (B, n_patches, d_model)
        mask_token = self.transformer.mask_token.expand(B, n_patches, -1)
        patches = torch.where(bool_mask.unsqueeze(-1), mask_token, patches)

        # 4. Prepend CLS + positional encoding + run encoder
        cls_tokens = self.transformer.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)

        if relative_positions is not None:
            cls_pos = torch.zeros(B, 1, device=relative_positions.device)
            full_pos = torch.cat([cls_pos, relative_positions[:, :n_patches]], dim=1)
            pos_enc = self.transformer.pos_encoder(n_patches + 1, full_pos)
        else:
            pos_enc = self.transformer.pos_encoder(n_patches + 1)
        tokens = tokens + pos_enc

        # Key padding mask
        key_padding_mask = None
        if padding_mask is not None:
            bar_mask = padding_mask[:, :T_usable]
            patch_pad = bar_mask.reshape(B, n_patches, self.cfg.patch_size).all(dim=2)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, patch_pad], dim=1)

        for layer in self.transformer.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)
        tokens = self.transformer.final_norm(tokens)

        embedding = tokens[:, 0, :]
        patch_out = tokens[:, 1:, :]  # (B, n_patches, d_model)

        # 5. Reconstruct only masked patches
        # Gather masked patch representations
        mask_idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
        masked_reps = patch_out.gather(1, mask_idx_expanded)  # (B, n_mask, d_model)
        predictions = self.mpp_head(masked_reps)  # (B, n_mask, patch_dim)

        # Gather corresponding targets
        target_idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, patch_dim)
        masked_targets = patch_targets.gather(1, target_idx_expanded)

        # 6. MSE loss on masked patches only
        loss = F.mse_loss(predictions, masked_targets)

        return loss, embedding, predictions
