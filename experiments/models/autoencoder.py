"""Autoencoder and Variational Autoencoder for learning market embeddings.

Architecture (default):
    Encoder: input_dim -> 256 -> 128 -> embedding_dim (64)
    Decoder: embedding_dim -> 128 -> 256 -> input_dim

The encoder learns a compressed representation z of the input features x.
For the standard autoencoder, we minimize reconstruction loss:

    L_recon = ||x - x_hat||^2

For the VAE, the encoder outputs parameters (mu, logvar) of a Gaussian
posterior q(z|x), and we minimize:

    L_VAE = L_recon + beta * D_KL(q(z|x) || p(z))

where D_KL is the Kullback-Leibler divergence:

    D_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

and beta controls the information bottleneck (beta=1 is standard VAE).

For the Supervised VAE, we add a prediction head that forces outcome-relevant
information into specific embedding dimensions:

    L_SVAE = L_recon + beta * D_KL + alpha * BCE(sigma(W^T z + b), y)

where alpha controls the prediction loss weight and BCE is binary cross-entropy.

Without orthogonality regularization, the prediction gradient flows equally
through all embedding dimensions, causing them all to encode the same signal
(multicollinearity in the embedding space). To break this symmetry, we add a
correlation penalty:

    L_orth = ||corr(Z) - I||_F^2

where corr(Z) is the batch Pearson correlation matrix of the embedding
activations and I is the identity matrix. This penalizes off-diagonal
correlations, forcing each dimension to encode distinct information.

Full loss:
    L = L_recon + beta * D_KL + alpha * BCE + gamma * L_orth

gamma controls the orthogonality penalty strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder architecture and training."""

    input_dim: int
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    model_type: Literal["ae", "vae", "supervised_vae"] = "ae"
    beta: float = 1.0  # KL weight for VAE
    alpha: float = 1.0  # Prediction loss weight for supervised VAE
    gamma: float = 0.0  # Orthogonality regularization weight


class Encoder(nn.Module):
    """Encoder network: maps input features to embedding space.

    For standard AE, produces a single embedding vector.
    For VAE, produces mu and logvar vectors parameterizing q(z|x).
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        prev_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)

        if config.model_type in ("vae", "supervised_vae"):
            self.fc_mu = nn.Linear(prev_dim, config.embedding_dim)
            self.fc_logvar = nn.Linear(prev_dim, config.embedding_dim)
        else:
            self.fc_out = nn.Linear(prev_dim, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Returns:
            AE mode: embedding tensor of shape (batch, embedding_dim)
            VAE/supervised_vae mode: tuple of (z, mu, logvar) where z is the reparameterized sample
        """
        h = self.shared(x)
        if self.config.model_type in ("vae", "supervised_vae"):
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar
        return self.fc_out(h)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma.

        Enables gradient flow through the stochastic sampling step.
        eps ~ N(0, I), sigma = exp(0.5 * logvar).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """Decoder network: maps embedding back to input space."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        reversed_dims = list(reversed(config.hidden_dims))
        prev_dim = config.embedding_dim
        for h_dim in reversed_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, config.input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from embedding."""
        return self.network(z)


class MarketAutoencoder(nn.Module):
    """Autoencoder / VAE for learning disentangled market embeddings.

    Usage:
        config = AutoencoderConfig(input_dim=50, embedding_dim=64, model_type='vae')
        model = MarketAutoencoder(config)

        # Training
        x_hat, z, loss_dict = model(x_batch)

        # Inference
        embeddings = model.get_embedding(x_batch)  # numpy array
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        if config.model_type == "supervised_vae":
            self.prediction_head = nn.Linear(config.embedding_dim, 1)

    @staticmethod
    def _correlation_penalty(z: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality penalty: ||corr(Z) - I||_F^2.

        Penalizes off-diagonal entries of the batch Pearson correlation matrix,
        forcing embedding dimensions to encode decorrelated (distinct) information.

        For batch embedding matrix Z of shape (B, D):

            corr(Z)_{ij} = cov(z_i, z_j) / (std(z_i) * std(z_j))

            L_orth = sum_{i != j} corr(Z)_{ij}^2

        This equals ||corr(Z) - I||_F^2 since diagonal entries are 1.

        Args:
            z: Embedding tensor of shape (batch, embedding_dim).

        Returns:
            Scalar penalty tensor.
        """
        # Center columns
        z_centered = z - z.mean(dim=0, keepdim=True)
        # Standard deviation per dim (with stability epsilon)
        std = z_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        z_norm = z_centered / std
        # Correlation matrix: (D, D)
        corr = (z_norm.T @ z_norm) / max(z.shape[0] - 1, 1)
        # Penalty: squared Frobenius norm of off-diagonal
        eye = torch.eye(corr.shape[0], device=corr.device)
        off_diag = corr - eye
        return (off_diag ** 2).sum()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass: encode, decode, compute losses.

        Args:
            x: Input tensor of shape (batch, input_dim).
            y: Target labels of shape (batch,) for supervised mode. Ignored for ae/vae.

        Returns:
            x_hat: Reconstructed input, shape (batch, input_dim).
            z: Embedding tensor, shape (batch, embedding_dim).
            losses: Dict with 'recon_loss' and optionally 'kl_loss', 'pred_loss',
                    'orth_loss', 'total_loss'.
        """
        if self.config.model_type in ("vae", "supervised_vae"):
            z, mu, logvar = self.encoder(x)
            x_hat = self.decoder(z)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + self.config.beta * kl_loss
            losses = {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "total_loss": total_loss,
            }
            if self.config.model_type == "supervised_vae" and y is not None:
                pred_logits = self.prediction_head(mu).squeeze(-1)
                pred_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, y.float(), reduction="mean",
                )
                losses["pred_loss"] = pred_loss
                losses["total_loss"] = losses["total_loss"] + self.config.alpha * pred_loss
            # Orthogonality regularization on mu (deterministic embedding)
            if self.config.gamma > 0 and mu.shape[0] > 1:
                orth_loss = self._correlation_penalty(mu)
                losses["orth_loss"] = orth_loss
                losses["total_loss"] = losses["total_loss"] + self.config.gamma * orth_loss
        else:
            z = self.encoder(x)
            x_hat = self.decoder(z)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
            losses = {
                "recon_loss": recon_loss,
                "total_loss": recon_loss,
            }
        return x_hat, z, losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space.

        For VAE/supervised_vae, returns the mean (mu) for deterministic embedding.
        """
        if self.config.model_type in ("vae", "supervised_vae"):
            _z, mu, _logvar = self.encoder(x)
            return mu
        return self.encoder(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict outcome probability from input (supervised_vae only).

        Returns:
            Probability tensor of shape (batch,).
        """
        mu = self.encode(x)
        logits = self.prediction_head(mu).squeeze(-1)
        return torch.sigmoid(logits)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embedding back to input space."""
        return self.decoder(z)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> np.ndarray:
        """Extract embeddings as a numpy array (inference mode).

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Numpy array of shape (N, embedding_dim).
        """
        self.eval()
        z = self.encode(x)
        return z.cpu().numpy()
