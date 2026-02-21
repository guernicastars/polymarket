"""Training pipeline for the market autoencoder.

Supports both standard autoencoder and VAE training with:
- Adam optimizer with cosine annealing LR schedule
- Early stopping with configurable patience
- Embedding quality monitoring (mean, std, max inter-dim correlation)
- Best model checkpointing
- Post-training embedding extraction

Usage:
    python -m models.train --config config.yaml
    python -m models.train --data-dir data/ --model-type vae --embedding-dim 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import AutoencoderConfig, MarketAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    model_type: str = "ae"
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    beta: float = 1.0
    alpha: float = 1.0  # Prediction loss weight for supervised_vae
    gamma: float = 0.0  # Orthogonality regularization weight

    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20

    data_dir: str = "data"
    output_dir: str = "results"
    seed: int = 42


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Load preprocessed feature matrix, optional labels, and metadata.

    Expects:
        data_dir/features.npy — (N, D) float32 feature matrix (already scaled)
        data_dir/metadata.json — dict with feature_names, label info, etc.
        data_dir/labels.npy — (N,) float32 outcome labels (optional, for supervised mode)

    Returns:
        X: Feature matrix as numpy array.
        y: Label array (or None if not available).
        metadata: Dict with dataset information.
    """
    data_path = Path(data_dir)
    X = np.load(data_path / "features.npy")
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    y = None
    labels_path = data_path / "labels.npy"
    if labels_path.exists():
        y = np.load(labels_path)
        logger.info("Loaded data: %d samples, %d features, with labels", X.shape[0], X.shape[1])
    else:
        logger.info("Loaded data: %d samples, %d features, no labels", X.shape[0], X.shape[1])
    return X, y, metadata


def compute_embedding_stats(embeddings: np.ndarray) -> dict[str, float]:
    """Monitor embedding quality during training.

    Computes:
        - Mean and std of embedding activations
        - Max absolute pairwise Pearson correlation between embedding dimensions
          (lower = more disentangled)
        - Fraction of "dead" dimensions (std < 1e-6)

    Args:
        embeddings: Array of shape (N, embedding_dim).

    Returns:
        Dict of summary statistics.
    """
    mean_act = float(np.mean(np.abs(embeddings)))
    std_act = float(np.std(embeddings))

    if embeddings.shape[0] > 1:
        corr = np.corrcoef(embeddings.T)
        np.fill_diagonal(corr, 0)
        max_corr = float(np.max(np.abs(corr)))
        mean_corr = float(np.mean(np.abs(corr)))
    else:
        max_corr = 0.0
        mean_corr = 0.0

    dead_dims = int(np.sum(np.std(embeddings, axis=0) < 1e-6))

    return {
        "mean_activation": mean_act,
        "std_activation": std_act,
        "max_inter_dim_correlation": max_corr,
        "mean_inter_dim_correlation": mean_corr,
        "dead_dimensions": dead_dims,
    }


class EarlyStopping:
    """Early stopping based on validation loss with model checkpointing."""

    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: MarketAutoencoder, epoch: int) -> bool:
        """Check if training should stop.

        Returns True if patience is exhausted.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(config: TrainConfig) -> dict:
    """Main training loop.

    Args:
        config: Training configuration.

    Returns:
        Dict with training history and final metrics.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # Load data
    X, y, metadata = load_data(config.data_dir)
    input_dim = X.shape[1]
    supervised = config.model_type == "supervised_vae" and y is not None

    if supervised:
        logger.info("Supervised mode: using labels for prediction loss (alpha=%.2f)", config.alpha)

    # Train/val split (80/20, shuffled)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    X_train = torch.tensor(X[indices[:split]], dtype=torch.float32)
    X_val = torch.tensor(X[indices[split:]], dtype=torch.float32)

    if supervised:
        y_train = torch.tensor(y[indices[:split]], dtype=torch.float32)
        y_val = torch.tensor(y[indices[split:]], dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val), batch_size=config.batch_size, shuffle=False
        )

    # Model
    ae_config = AutoencoderConfig(
        input_dim=input_dim,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        model_type=config.model_type,
        beta=config.beta,
        alpha=config.alpha,
        gamma=config.gamma,
    )
    model = MarketAutoencoder(ae_config).to(device)
    logger.info("Model: %s, params: %d", config.model_type.upper(),
                sum(p.numel() for p in model.parameters()))

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.learning_rate,
                     weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Checkpointing
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"best_model_{config.model_type}.pt"
    early_stop = EarlyStopping(config.patience, checkpoint_path)

    # Training loop
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "recon_loss": [],
        "kl_loss": [], "pred_loss": [], "orth_loss": [], "pred_acc": [],
        "lr": [], "embedding_stats": [],
    }

    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        # -- Train --
        model.train()
        epoch_losses: dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0, "pred": 0.0, "orth": 0.0}
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for batch in train_loader:
            if supervised:
                batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                _x_hat, _z, losses = model(batch_x, batch_y)
            else:
                batch_x = batch[0].to(device)
                _x_hat, _z, losses = model(batch_x)

            loss = losses["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["recon"] += losses["recon_loss"].item()
            if "kl_loss" in losses:
                epoch_losses["kl"] += losses["kl_loss"].item()
            if "pred_loss" in losses:
                epoch_losses["pred"] += losses["pred_loss"].item()
                # Track prediction accuracy
                with torch.no_grad():
                    pred_probs = model.predict(batch_x)
                    pred_labels = (pred_probs >= 0.5).float()
                    epoch_correct += (pred_labels == batch_y).sum().item()
                    epoch_total += batch_y.shape[0]
            if "orth_loss" in losses:
                epoch_losses["orth"] += losses["orth_loss"].item()
            n_batches += 1

        avg_train = {k: v / n_batches for k, v in epoch_losses.items()}
        train_pred_acc = epoch_correct / max(epoch_total, 1) if supervised else 0.0

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_embeddings = []
        with torch.no_grad():
            for batch in val_loader:
                if supervised:
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    _x_hat, z, losses = model(batch_x, batch_y)
                    pred_probs = model.predict(batch_x)
                    pred_labels = (pred_probs >= 0.5).float()
                    val_correct += (pred_labels == batch_y).sum().item()
                    val_total += batch_y.shape[0]
                else:
                    batch_x = batch[0].to(device)
                    _x_hat, z, losses = model(batch_x)
                val_loss += losses["total_loss"].item()
                all_embeddings.append(z.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_pred_acc = val_correct / max(val_total, 1) if supervised else 0.0
        embeddings = np.concatenate(all_embeddings, axis=0)
        emb_stats = compute_embedding_stats(embeddings)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        history["train_loss"].append(avg_train["total"])
        history["val_loss"].append(avg_val_loss)
        history["recon_loss"].append(avg_train["recon"])
        history["kl_loss"].append(avg_train["kl"])
        history["pred_loss"].append(avg_train["pred"])
        history["orth_loss"].append(avg_train["orth"])
        history["pred_acc"].append(val_pred_acc)
        history["lr"].append(current_lr)
        history["embedding_stats"].append(emb_stats)

        if epoch % 10 == 0 or epoch == 1:
            msg = (f"Epoch {epoch:4d} | train={avg_train['total']:.6f} "
                   f"val={avg_val_loss:.6f} recon={avg_train['recon']:.6f}")
            if config.model_type in ("vae", "supervised_vae"):
                msg += f" kl={avg_train['kl']:.6f}"
            if supervised:
                msg += f" pred={avg_train['pred']:.6f} acc={val_pred_acc:.3f}"
            if avg_train["orth"] > 0:
                msg += f" orth={avg_train['orth']:.4f}"
            msg += (f" | max_corr={emb_stats['max_inter_dim_correlation']:.3f} "
                    f"dead={emb_stats['dead_dimensions']}")
            logger.info(msg)

        if early_stop.step(avg_val_loss, model, epoch):
            logger.info("Early stopping at epoch %d (best: %d, loss: %.6f)",
                        epoch, early_stop.best_epoch, early_stop.best_loss)
            break

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    # Load best model and extract embeddings for full dataset
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    X_full = torch.tensor(X, dtype=torch.float32).to(device)
    all_embeddings = model.get_embedding(X_full)

    np.save(output_path / "embeddings.npy", all_embeddings)
    logger.info("Saved embeddings: shape %s", all_embeddings.shape)

    # Save training history
    with open(output_path / "train_history.json", "w") as f:
        json.dump({
            "config": asdict(config),
            "ae_config": asdict(ae_config),
            "best_epoch": early_stop.best_epoch,
            "best_val_loss": early_stop.best_loss,
            "elapsed_seconds": elapsed,
            "final_embedding_stats": compute_embedding_stats(all_embeddings),
            "history": {k: v for k, v in history.items() if k != "embedding_stats"},
        }, f, indent=2, default=str)

    return {
        "embeddings": all_embeddings,
        "history": history,
        "best_epoch": early_stop.best_epoch,
        "best_val_loss": early_stop.best_loss,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train market autoencoder")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="results", help="Path to output directory")
    parser.add_argument("--model-type", choices=["ae", "vae", "supervised_vae"], default="ae")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight for VAE")
    parser.add_argument("--alpha", type=float, default=1.0, help="Prediction loss weight for supervised VAE")
    parser.add_argument("--gamma", type=float, default=0.0, help="Orthogonality regularization weight")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = TrainConfig(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    train(config)


if __name__ == "__main__":
    main()
