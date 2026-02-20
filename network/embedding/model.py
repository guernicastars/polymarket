"""Autoencoder models for market embedding.

Two variants:
  - MarketAutoencoder: standard AE (deterministic bottleneck)
  - VariationalAutoencoder: VAE with reparameterization trick (beta-VAE for disentanglement)

Architecture per layer: Linear -> BatchNorm -> GELU -> Dropout
Final decoder layer has no activation (reconstruction in z-score space).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AutoencoderConfig


def _build_layers(
    dims: list[int],
    dropout: float,
    use_bn: bool,
    final_activation: bool = True,
) -> nn.Sequential:
    """Build a stack of Linear -> BN -> GELU -> Dropout layers."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MarketAutoencoder(nn.Module):
    """Standard autoencoder for market summary features.

    Forward pass:
      x: (batch, input_dim)
      -> encoder -> z: (batch, latent_dim)
      -> decoder -> x_hat: (batch, input_dim)

    Returns (x_hat, z) for reconstruction loss + embedding extraction.
    """

    def __init__(self, input_dim: int, cfg: Optional[AutoencoderConfig] = None):
        super().__init__()
        self.cfg = cfg or AutoencoderConfig()
        self.input_dim = input_dim
        self.latent_dim = self.cfg.latent_dim

        # Encoder: input_dim -> hidden layers -> latent_dim
        enc_dims = [input_dim] + list(self.cfg.encoder_hidden) + [self.cfg.latent_dim]
        self.encoder = _build_layers(
            enc_dims, self.cfg.dropout, self.cfg.use_batch_norm, final_activation=False
        )

        # Decoder: latent_dim -> hidden layers -> input_dim
        dec_dims = [self.cfg.latent_dim] + list(self.cfg.decoder_hidden) + [input_dim]
        self.decoder = _build_layers(
            dec_dims, self.cfg.dropout, self.cfg.use_batch_norm, final_activation=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, latent_dim)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> (batch, input_dim)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, embedding)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for disentangled representations.

    Encoder outputs mu and log_var; reparameterization trick samples z.
    Loss = reconstruction_loss + beta * KL_divergence (beta-VAE).

    Forward pass:
      x: (batch, input_dim)
      -> encoder -> (mu, log_var): each (batch, latent_dim)
      -> reparameterize -> z: (batch, latent_dim)
      -> decoder -> x_hat: (batch, input_dim)

    Returns (x_hat, mu, log_var, z).
    """

    def __init__(self, input_dim: int, cfg: Optional[AutoencoderConfig] = None):
        super().__init__()
        self.cfg = cfg or AutoencoderConfig()
        self.input_dim = input_dim
        self.latent_dim = self.cfg.latent_dim

        # Shared encoder backbone
        enc_dims = [input_dim] + list(self.cfg.encoder_hidden)
        self.encoder_backbone = _build_layers(
            enc_dims, self.cfg.dropout, self.cfg.use_batch_norm, final_activation=True
        )

        # Separate heads for mu and log_var
        backbone_out = self.cfg.encoder_hidden[-1]
        self.fc_mu = nn.Linear(backbone_out, self.cfg.latent_dim)
        self.fc_log_var = nn.Linear(backbone_out, self.cfg.latent_dim)

        # Decoder: latent_dim -> hidden layers -> input_dim
        dec_dims = [self.cfg.latent_dim] + list(self.cfg.decoder_hidden) + [input_dim]
        self.decoder = _build_layers(
            dec_dims, self.cfg.dropout, self.cfg.use_batch_norm, final_activation=False
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, input_dim) -> (mu, log_var) each (batch, latent_dim)."""
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * epsilon (differentiable)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at eval time

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> (batch, input_dim)"""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_var, z)."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 0.001,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combined reconstruction + KL divergence loss.

        Returns (total_loss, recon_loss, kl_loss).
        """
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total = recon + kl_weight * kl
        return total, recon, kl


def create_autoencoder(input_dim: int, cfg: Optional[AutoencoderConfig] = None) -> nn.Module:
    """Factory: create AE or VAE based on config."""
    cfg = cfg or AutoencoderConfig()
    if cfg.variational:
        return VariationalAutoencoder(input_dim, cfg)
    return MarketAutoencoder(input_dim, cfg)
