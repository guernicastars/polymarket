"""Training loop, embedding extraction, and CLI for the embedding module.

Autoencoder modes (summary features):
    python -m network.embedding.train --mode full --latent-dim 64 --epochs 200
    python -m network.embedding.train --mode train --variational --kl-weight 0.005
    python -m network.embedding.train --mode analyze --checkpoint checkpoints/best.pt

Transformer modes (temporal sequences):
    python -m network.embedding.train --mode pretrain --arch transformer --epochs 100
    python -m network.embedding.train --mode finetune --arch transformer --checkpoint checkpoints/transformer_pretrained.pt
    python -m network.embedding.train --mode fuse --ae-checkpoint checkpoints/best.pt --tf-checkpoint checkpoints/transformer_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import EmbeddingConfig
from .data import ResolvedMarketDataset, collate_embedding_batch
from .model import MarketAutoencoder, VariationalAutoencoder, create_autoencoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def get_clickhouse_client(cfg: EmbeddingConfig):
    """Create ClickHouse client from config (same pattern as GNN module)."""
    import clickhouse_connect

    host = os.environ.get("CLICKHOUSE_HOST", cfg.clickhouse_host)
    port = int(os.environ.get("CLICKHOUSE_PORT", cfg.clickhouse_port))
    user = os.environ.get("CLICKHOUSE_USER", cfg.clickhouse_user)
    password = os.environ.get("CLICKHOUSE_PASSWORD", cfg.clickhouse_password)
    database = os.environ.get("CLICKHOUSE_DATABASE", cfg.clickhouse_database)

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
        secure=True,
    )


class EmbeddingTrainer:
    """Trains the autoencoder and orchestrates the full probe pipeline."""

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        self.cfg = cfg or EmbeddingConfig()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

    def train_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
    ) -> tuple[nn.Module, list[float], list[float]]:
        """Train autoencoder with early stopping on reconstruction loss.

        Returns (trained_model, train_losses, val_losses).
        """
        ac = self.cfg.autoencoder
        model = model.to(self.device)
        is_vae = isinstance(model, VariationalAutoencoder)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=ac.learning_rate, weight_decay=ac.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ac.epochs, eta_min=1e-6
        )

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []

        checkpoint_dir = PROJECT_ROOT / self.cfg.model_save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(ac.epochs):
            # --- Training ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x = batch["features"].to(self.device)
                optimizer.zero_grad()

                if is_vae:
                    x_hat, mu, log_var, z = model(x)
                    loss, recon, kl = VariationalAutoencoder.loss(
                        x, x_hat, mu, log_var, ac.kl_weight
                    )
                else:
                    x_hat, z = model(x)
                    loss = nn.functional.mse_loss(x_hat, x)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ac.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # --- Validation ---
            val_loss = self._evaluate(model, val_loader, is_vae)
            val_losses.append(val_loss)

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
                    epoch + 1, ac.epochs, train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= ac.patience:
                    logger.info("Early stopping at epoch %d (best val=%.6f)", epoch + 1, best_val_loss)
                    break

        # Load best model
        model.load_state_dict(
            torch.load(checkpoint_dir / "best.pt", weights_only=True, map_location=self.device)
        )
        logger.info("Training complete. Best val loss: %.6f", best_val_loss)
        return model, train_losses, val_losses

    def extract_embeddings(
        self,
        model: nn.Module,
        dataset: ResolvedMarketDataset,
    ) -> np.ndarray:
        """Extract embeddings for all markets using frozen encoder.

        Returns (N_markets, latent_dim) numpy array.
        """
        model.eval()
        model.to(self.device)

        loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_embedding_batch)
        all_embeddings = []

        with torch.no_grad():
            for batch in loader:
                x = batch["features"].to(self.device)
                if isinstance(model, VariationalAutoencoder):
                    mu, _ = model.encode(x)
                    all_embeddings.append(mu.cpu().numpy())
                else:
                    z = model.encode(x)
                    all_embeddings.append(z.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def run_probes(
        self,
        embeddings: np.ndarray,
        dataset: ResolvedMarketDataset,
    ) -> list:
        """Run all linear probes on extracted embeddings."""
        from .probes import LinearProbe

        probe = LinearProbe(self.cfg.probe)
        return probe.run_all_probes(embeddings, dataset.labels, dataset.feature_names)

    def save_results(
        self,
        embeddings: np.ndarray,
        probe_results: list,
        dataset: ResolvedMarketDataset,
        model: nn.Module,
        train_losses: list[float],
        val_losses: list[float],
    ) -> None:
        """Save all artifacts: embeddings, probes, model, normalization params."""
        results_dir = PROJECT_ROOT / self.cfg.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # Embeddings
        np.save(results_dir / "embeddings.npy", embeddings)

        # Raw features + normalization
        np.save(results_dir / "raw_features.npy", dataset.raw_features)
        np.save(results_dir / "feature_mean.npy", dataset.mean)
        np.save(results_dir / "feature_std.npy", dataset.std)

        # Probe results
        probe_dicts = []
        for pr in probe_results:
            d = {
                "concept_name": pr.concept_name,
                "task_type": pr.task_type,
                "accuracy": pr.accuracy,
                "accuracy_std": pr.accuracy_std,
                "baseline_accuracy": pr.baseline_accuracy,
                "p_value": pr.p_value,
                "is_significant": pr.is_significant,
                "cv_scores": pr.cv_scores,
            }
            probe_dicts.append(d)
        with open(results_dir / "probe_results.json", "w") as f:
            json.dump(probe_dicts, f, indent=2)

        # Training curves
        with open(results_dir / "training_curves.json", "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f)

        # Market metadata
        market_meta = []
        for i, m in enumerate(dataset.markets):
            market_meta.append({
                "condition_id": m["condition_id"],
                "question": m.get("question", ""),
                "category": m.get("category", ""),
                "winning_outcome": m.get("winning_outcome", ""),
                "volume_total": m.get("volume_total", 0),
            })
        with open(results_dir / "market_metadata.json", "w") as f:
            json.dump(market_meta, f, indent=2)

        # Feature names
        with open(results_dir / "feature_names.json", "w") as f:
            json.dump(dataset.feature_names, f)

        logger.info("Results saved to %s", results_dir)

    def _evaluate(self, model: nn.Module, loader: DataLoader, is_vae: bool) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["features"].to(self.device)
                if is_vae:
                    x_hat, mu, log_var, z = model(x)
                    loss, _, _ = VariationalAutoencoder.loss(
                        x, x_hat, mu, log_var, self.cfg.autoencoder.kl_weight
                    )
                else:
                    x_hat, z = model(x)
                    loss = nn.functional.mse_loss(x_hat, x)
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)


# ============================================================
# Transformer Trainer
# ============================================================

class TransformerTrainer:
    """Trains the transformer encoder via masked patch prediction and fine-tuning.

    Two-phase training:
      1. Pre-train: MPP on all markets (active + resolved), self-supervised
      2. Fine-tune: optional supervised objective on resolved markets
      3. Fuse: concatenate with autoencoder embeddings, run probes
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        self.cfg = cfg or EmbeddingConfig()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Transformer trainer using device: %s", self.device)

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[nn.Module, list[float], list[float]]:
        """Pre-train transformer via masked patch prediction.

        Returns (pretrained_model, train_losses, val_losses).
        The returned model is the MarketTransformerForPretraining wrapper;
        use .transformer to get the encoder for embedding extraction.
        """
        from .transformer_model import MarketTransformerForPretraining

        tc = self.cfg.transformer
        model = MarketTransformerForPretraining(tc).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Transformer pre-training: %d params, d_model=%d, %d layers, %d heads",
                     n_params, tc.d_model, tc.n_layers, tc.n_heads)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc.pretrain_lr, weight_decay=tc.pretrain_weight_decay,
        )

        # Linear warmup + cosine decay
        total_steps = tc.pretrain_epochs * len(train_loader)
        warmup_steps = min(tc.pretrain_warmup_steps, total_steps // 5)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        checkpoint_dir = PROJECT_ROOT / self.cfg.model_save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []
        global_step = 0

        for epoch in range(tc.pretrain_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                rel_pos = batch["relative_positions"].to(self.device)

                optimizer.zero_grad()
                loss, embedding, predictions = model(
                    features, padding_mask=padding_mask,
                    relative_positions=rel_pos,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # Validation
            val_loss = self._evaluate_pretrain(model, val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "[Pretrain] Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
                    epoch + 1, tc.pretrain_epochs, train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "transformer_pretrained.pt")
            else:
                patience_counter += 1
                if patience_counter >= tc.pretrain_patience:
                    logger.info("Pre-training early stop at epoch %d (best val=%.6f)",
                                epoch + 1, best_val_loss)
                    break

        # Load best
        model.load_state_dict(
            torch.load(checkpoint_dir / "transformer_pretrained.pt",
                        weights_only=True, map_location=self.device)
        )
        logger.info("Pre-training complete. Best val loss: %.6f", best_val_loss)
        return model, train_losses, val_losses

    def extract_embeddings(
        self,
        model: nn.Module,
        dataset,
    ) -> np.ndarray:
        """Extract CLS embeddings from transformer encoder.

        Accepts either MarketTransformerForPretraining or MarketTransformer.
        """
        from .transformer_model import MarketTransformerForPretraining, MarketTransformer
        from .temporal_dataset import collate_temporal_batch

        # Get the base transformer encoder
        if isinstance(model, MarketTransformerForPretraining):
            encoder = model.transformer
        elif isinstance(model, MarketTransformer):
            encoder = model
        else:
            raise TypeError(f"Expected MarketTransformer*, got {type(model)}")

        encoder.eval()
        encoder.to(self.device)

        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_temporal_batch)
        all_embeddings = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                rel_pos = batch["relative_positions"].to(self.device)

                embedding = encoder.encode(
                    features, padding_mask=padding_mask,
                    relative_positions=rel_pos,
                )
                all_embeddings.append(embedding.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def run_probes(
        self,
        embeddings: np.ndarray,
        labels: list[dict],
        feature_names: list[str],
    ) -> list:
        """Run linear probes on transformer embeddings."""
        from .probes import LinearProbe
        probe = LinearProbe(self.cfg.probe)
        return probe.run_all_probes(embeddings, labels, feature_names)

    def save_results(
        self,
        embeddings: np.ndarray,
        probe_results: list,
        dataset,
        train_losses: list[float],
        val_losses: list[float],
        prefix: str = "transformer",
    ) -> None:
        """Save transformer embedding results."""
        results_dir = PROJECT_ROOT / self.cfg.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        np.save(results_dir / f"{prefix}_embeddings.npy", embeddings)

        probe_dicts = []
        for pr in probe_results:
            probe_dicts.append({
                "concept_name": pr.concept_name,
                "task_type": pr.task_type,
                "accuracy": pr.accuracy,
                "accuracy_std": pr.accuracy_std,
                "baseline_accuracy": pr.baseline_accuracy,
                "p_value": pr.p_value,
                "is_significant": pr.is_significant,
                "cv_scores": pr.cv_scores,
            })
        with open(results_dir / f"{prefix}_probe_results.json", "w") as f:
            json.dump(probe_dicts, f, indent=2)

        with open(results_dir / f"{prefix}_training_curves.json", "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f)

        # Config snapshot
        with open(results_dir / f"{prefix}_config.json", "w") as f:
            import dataclasses
            json.dump(dataclasses.asdict(self.cfg.transformer), f, indent=2)

        logger.info("Transformer results saved to %s", results_dir)

    def _evaluate_pretrain(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                rel_pos = batch["relative_positions"].to(self.device)
                loss, _, _ = model(
                    features, padding_mask=padding_mask,
                    relative_positions=rel_pos,
                )
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)


def _print_probe_results(probe_results: list, alpha: float) -> None:
    """Print probe results table."""
    print("\n" + "=" * 70)
    print("LINEAR PROBE RESULTS")
    print("=" * 70)
    for pr in probe_results:
        sig = "*" if pr.is_significant else " "
        print(
            f"  {sig} {pr.concept_name:<25} {pr.task_type:<15} "
            f"acc={pr.accuracy:.3f}\u00b1{pr.accuracy_std:.3f}  "
            f"baseline={pr.baseline_accuracy:.3f}  p={pr.p_value:.4f}"
        )
    print("=" * 70)
    print(f"  Significant probes (p<{alpha}): "
          f"{sum(1 for pr in probe_results if pr.is_significant)}/{len(probe_results)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Embedding module: autoencoder + transformer + probes")
    parser.add_argument("--mode", choices=[
        "train", "analyze", "full",           # autoencoder modes
        "pretrain", "finetune", "fuse",       # transformer modes
    ], default="full")
    parser.add_argument("--arch", choices=["autoencoder", "transformer"], default="autoencoder")

    # Autoencoder args
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--variational", action="store_true")
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--cutoff", type=float, default=0.8)
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)

    # Transformer args
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--pretrain-epochs", type=int, default=100)

    # Fusion args
    parser.add_argument("--ae-checkpoint", type=str, default=None)
    parser.add_argument("--tf-checkpoint", type=str, default=None)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = EmbeddingConfig()

    # Apply autoencoder overrides
    cfg.autoencoder.latent_dim = args.latent_dim
    cfg.autoencoder.variational = args.variational
    cfg.autoencoder.kl_weight = args.kl_weight
    cfg.autoencoder.epochs = args.epochs
    cfg.autoencoder.batch_size = args.batch_size
    cfg.autoencoder.learning_rate = args.lr
    cfg.features.min_volume_total = args.min_volume
    cfg.features.lifetime_cutoff_ratio = args.cutoff
    cfg.features.max_markets = args.max_markets

    # Apply transformer overrides
    cfg.transformer.d_model = args.d_model
    cfg.transformer.n_layers = args.n_layers
    cfg.transformer.n_heads = args.n_heads
    cfg.transformer.patch_size = args.patch_size
    cfg.transformer.mask_ratio = args.mask_ratio
    cfg.transformer.pretrain_epochs = args.pretrain_epochs

    logger.info("Connecting to ClickHouse...")
    client = get_clickhouse_client(cfg)

    # ---- Transformer modes ----
    if args.mode in ("pretrain", "finetune", "fuse"):
        from .temporal_dataset import TemporalMarketDataset, collate_temporal_batch
        from .transformer_model import MarketTransformerForPretraining, MarketTransformer

        tf_trainer = TransformerTrainer(cfg)

        if args.mode == "pretrain":
            logger.info("Building temporal dataset for pre-training (all markets)...")
            dataset = TemporalMarketDataset(client, cfg.transformer, mode="pretrain")
            if len(dataset) < 20:
                logger.error("Not enough markets: %d found.", len(dataset))
                return

            logger.info("Pre-training dataset: %d markets", len(dataset))

            # Split 85/15 for pre-training
            n = len(dataset)
            n_val = max(int(n * 0.15), 1)
            n_train = n - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(
                train_ds, batch_size=cfg.transformer.pretrain_batch_size,
                shuffle=True, collate_fn=collate_temporal_batch,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.transformer.pretrain_batch_size,
                shuffle=False, collate_fn=collate_temporal_batch,
            )

            model, train_losses, val_losses = tf_trainer.pretrain(train_loader, val_loader)

            # Extract embeddings for monitoring
            embeddings = tf_trainer.extract_embeddings(model, dataset)
            logger.info("Pre-trained embeddings: shape %s", embeddings.shape)
            tf_trainer.save_results(embeddings, [], dataset, train_losses, val_losses, prefix="transformer_pretrain")

        elif args.mode == "finetune":
            logger.info("Building temporal dataset for fine-tuning (resolved markets)...")
            dataset = TemporalMarketDataset(client, cfg.transformer, mode="finetune")
            if len(dataset) < 20:
                logger.error("Not enough resolved markets: %d found.", len(dataset))
                return

            logger.info("Fine-tune dataset: %d markets", len(dataset))

            # Load pre-trained model
            model_wrapper = MarketTransformerForPretraining(cfg.transformer)
            if args.checkpoint:
                model_wrapper.load_state_dict(
                    torch.load(args.checkpoint, weights_only=True, map_location=tf_trainer.device)
                )
                logger.info("Loaded pre-trained checkpoint: %s", args.checkpoint)

            # Extract embeddings + run probes
            embeddings = tf_trainer.extract_embeddings(model_wrapper, dataset)
            probe_results = tf_trainer.run_probes(embeddings, dataset.labels, dataset.feature_names)
            _print_probe_results(probe_results, cfg.probe.alpha)
            tf_trainer.save_results(embeddings, probe_results, dataset, [], [], prefix="transformer_finetune")

        elif args.mode == "fuse":
            if not args.ae_checkpoint or not args.tf_checkpoint:
                logger.error("--ae-checkpoint and --tf-checkpoint required for fuse mode")
                return

            # Load autoencoder embeddings
            logger.info("Loading autoencoder for fusion...")
            ae_dataset = ResolvedMarketDataset(client, cfg.features)
            input_dim = ae_dataset.features.shape[1]
            ae_model = create_autoencoder(input_dim, cfg.autoencoder)
            ae_model.load_state_dict(
                torch.load(args.ae_checkpoint, weights_only=True, map_location=tf_trainer.device)
            )
            ae_trainer = EmbeddingTrainer(cfg)
            ae_embeddings = ae_trainer.extract_embeddings(ae_model, ae_dataset)

            # Load transformer embeddings
            logger.info("Loading transformer for fusion...")
            tf_dataset = TemporalMarketDataset(client, cfg.transformer, mode="finetune")
            tf_model = MarketTransformerForPretraining(cfg.transformer)
            tf_model.load_state_dict(
                torch.load(args.tf_checkpoint, weights_only=True, map_location=tf_trainer.device)
            )
            tf_embeddings = tf_trainer.extract_embeddings(tf_model, tf_dataset)

            # Match markets by condition_id
            ae_cids = {m["condition_id"]: i for i, m in enumerate(ae_dataset.markets)}
            tf_cids = {m["condition_id"]: i for i, m in enumerate(tf_dataset.markets)}
            common_cids = sorted(set(ae_cids.keys()) & set(tf_cids.keys()))

            logger.info("Fusion: %d AE markets, %d TF markets, %d in common",
                        len(ae_cids), len(tf_cids), len(common_cids))

            if len(common_cids) < 20:
                logger.error("Not enough common markets for fusion: %d", len(common_cids))
                return

            # Build fused embeddings
            ae_matched = np.array([ae_embeddings[ae_cids[c]] for c in common_cids])
            tf_matched = np.array([tf_embeddings[tf_cids[c]] for c in common_cids])
            fused = np.concatenate([ae_matched, tf_matched], axis=1)

            # Build matching labels
            fused_labels = [ae_dataset.labels[ae_cids[c]] for c in common_cids]

            logger.info("Fused embedding: shape %s (AE=%d + TF=%d)",
                        fused.shape, ae_matched.shape[1], tf_matched.shape[1])

            # Run probes on all three: AE-only, TF-only, fused
            print("\n--- Autoencoder Only ---")
            ae_probes = tf_trainer.run_probes(ae_matched, fused_labels, ae_dataset.feature_names)
            _print_probe_results(ae_probes, cfg.probe.alpha)

            print("\n--- Transformer Only ---")
            tf_probes = tf_trainer.run_probes(tf_matched, fused_labels, tf_dataset.feature_names)
            _print_probe_results(tf_probes, cfg.probe.alpha)

            print("\n--- Fused (AE + Transformer) ---")
            fused_probes = tf_trainer.run_probes(
                fused, fused_labels,
                ae_dataset.feature_names + [f"tf_{n}" for n in tf_dataset.feature_names],
            )
            _print_probe_results(fused_probes, cfg.probe.alpha)

            # Save fused results
            results_dir = PROJECT_ROOT / cfg.results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            np.save(results_dir / "fused_embeddings.npy", fused)

            probe_dicts = []
            for pr in fused_probes:
                probe_dicts.append({
                    "concept_name": pr.concept_name,
                    "accuracy": pr.accuracy,
                    "baseline_accuracy": pr.baseline_accuracy,
                    "p_value": pr.p_value,
                    "is_significant": pr.is_significant,
                })
            with open(results_dir / "fused_probe_results.json", "w") as f:
                json.dump(probe_dicts, f, indent=2)

            logger.info("Fusion results saved to %s", results_dir)

        return

    # ---- Autoencoder modes (original) ----
    logger.info("Building resolved market dataset (cutoff=%.1f)...", args.cutoff)
    dataset = ResolvedMarketDataset(client, cfg.features)

    if len(dataset) < 20:
        logger.error("Not enough resolved markets: %d found. Need at least 20.", len(dataset))
        return

    logger.info(
        "Dataset: %d markets, %d features, %d binary outcomes",
        len(dataset),
        dataset.features.shape[1],
        sum(1 for lb in dataset.labels if lb["outcome_binary"] >= 0),
    )

    trainer = EmbeddingTrainer(cfg)

    if args.mode in ("train", "full"):
        n = len(dataset)
        ac = cfg.autoencoder
        n_train = int(n * ac.train_ratio)
        n_val = int(n * ac.val_ratio)
        n_test = n - n_train - n_val

        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

        train_loader = DataLoader(
            train_ds, batch_size=ac.batch_size, shuffle=True, collate_fn=collate_embedding_batch
        )
        val_loader = DataLoader(
            val_ds, batch_size=ac.batch_size, shuffle=False, collate_fn=collate_embedding_batch
        )

        input_dim = dataset.features.shape[1]
        model = create_autoencoder(input_dim, cfg.autoencoder)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model: %s, %d parameters, latent_dim=%d", type(model).__name__, n_params, cfg.autoencoder.latent_dim)

        model, train_losses, val_losses = trainer.train_autoencoder(train_loader, val_loader, model)
        embeddings = trainer.extract_embeddings(model, dataset)
        logger.info("Extracted embeddings: shape %s", embeddings.shape)

        if args.mode == "full":
            logger.info("Running linear probes...")
            probe_results = trainer.run_probes(embeddings, dataset)
            _print_probe_results(probe_results, cfg.probe.alpha)

            logger.info("Running disentanglement analysis...")
            from .analysis import DisentanglementAnalyzer
            analyzer = DisentanglementAnalyzer(cfg.probe)
            pca_result = analyzer.pca_analysis(embeddings, dataset.feature_names, dataset.raw_features)

            print(f"\n  PCA: {pca_result['cumulative_variance'][-1]:.1%} variance in "
                  f"{cfg.probe.n_pca_components} components")

            novel = analyzer.search_novel_directions(embeddings, dataset.feature_names, dataset.raw_features)
            if novel:
                print(f"  Novel directions found: {len(novel)}")
                for nd in novel[:3]:
                    print(f"    PC{nd.direction_idx}: var={nd.variance_explained:.3f} "
                          f"- {nd.description}")

            trainer.save_results(embeddings, probe_results, dataset, model, train_losses, val_losses)
        else:
            trainer.save_results(embeddings, [], dataset, model, train_losses, val_losses)

    elif args.mode == "analyze":
        if not args.checkpoint:
            logger.error("--checkpoint required for analyze mode")
            return

        input_dim = dataset.features.shape[1]
        model = create_autoencoder(input_dim, cfg.autoencoder)
        model.load_state_dict(
            torch.load(args.checkpoint, weights_only=True, map_location=trainer.device)
        )

        embeddings = trainer.extract_embeddings(model, dataset)
        probe_results = trainer.run_probes(embeddings, dataset)

        for pr in probe_results:
            sig = "*" if pr.is_significant else " "
            print(
                f"  {sig} {pr.concept_name:<25} {pr.task_type:<15} "
                f"acc={pr.accuracy:.3f}\u00b1{pr.accuracy_std:.3f}  "
                f"baseline={pr.baseline_accuracy:.3f}  p={pr.p_value:.4f}"
            )

        trainer.save_results(embeddings, probe_results, dataset, model, [], [])


if __name__ == "__main__":
    main()
