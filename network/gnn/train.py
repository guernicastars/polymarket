"""Training loop, evaluation, and CLI entrypoint for the GNN-TCN model.

Usage:
    python -m network.gnn.train --epochs 100 --lr 1e-3
    python -m network.gnn.train --mode backtest --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import GNNConfig
from .model import GNNTCN, PlattScaling
from .dataset import DonbasTemporalDataset, collate_graph_batch
from .backtest import BacktestEngine, kelly_criterion

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def get_clickhouse_client(cfg: GNNConfig):
    """Create ClickHouse client from config."""
    import clickhouse_connect

    # Try environment variables first, then config defaults
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


class Trainer:
    """Handles training, validation, Platt calibration, and backtesting."""

    def __init__(self, cfg: Optional[GNNConfig] = None):
        self.cfg = cfg or GNNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: GNNTCN,
    ) -> tuple[GNNTCN, list[float], list[float]]:
        """Main training loop with early stopping.

        Returns:
            (trained_model, train_losses, val_losses)
        """
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.model.epochs, eta_min=1e-6
        )
        criterion = nn.MSELoss()  # predict probability (price)

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []

        checkpoint_dir = PROJECT_ROOT / self.cfg.model_save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.model.epochs):
            # --- Training ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x = batch["x"].to(self.device)         # (B, N, W, F)
                adj = batch["adj"].to(self.device)      # (N, N)
                y = batch["y"].to(self.device)           # (B, n_targets)

                optimizer.zero_grad()
                logits = model(x, adj)                   # (B, n_targets)
                preds = torch.sigmoid(logits)
                loss = criterion(preds, y)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.model.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # --- Validation ---
            val_loss = self._evaluate(model, val_loader, criterion)
            val_losses.append(val_loss)

            scheduler.step()

            logger.info(
                "Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
                epoch + 1, self.cfg.model.epochs,
                train_loss, val_loss,
                optimizer.param_groups[0]["lr"],
            )

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
                logger.info("  → New best model saved (val=%.6f)", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.model.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Load best model
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt", weights_only=True))
        return model, train_losses, val_losses

    def calibrate(
        self,
        model: GNNTCN,
        val_loader: DataLoader,
    ) -> PlattScaling:
        """Fit Platt scaling on validation set (Point 3).

        The model is frozen; only a/b parameters are learned.
        """
        model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                adj = batch["adj"].to(self.device)
                y = batch["y"]

                logits = model(x, adj).cpu()
                all_logits.append(logits)
                all_targets.append(y)

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        n_targets = logits_cat.shape[1]
        platt = PlattScaling(n_targets=n_targets)
        losses = platt.fit(
            logits_cat, targets_cat,
            lr=self.cfg.model.platt_lr,
            epochs=self.cfg.model.platt_epochs,
        )
        logger.info("Platt scaling fitted. Final loss: %.6f", losses[-1] if losses else float("nan"))

        return platt

    def backtest(
        self,
        model: GNNTCN,
        platt: PlattScaling,
        test_loader: DataLoader,
        target_names: Optional[list[str]] = None,
    ) -> None:
        """Run full backtest on test set (Point 5)."""
        model.eval()
        platt.eval()

        all_preds, all_prices, all_ts = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                adj = batch["adj"].to(self.device)
                y = batch["y"]
                ts = batch["timestamps"]

                logits = model(x, adj).cpu()
                calibrated = platt(logits)

                all_preds.append(calibrated.numpy())
                all_prices.append(y.numpy())
                all_ts.extend(ts)

        predictions = np.concatenate(all_preds, axis=0)
        prices = np.concatenate(all_prices, axis=0)

        engine = BacktestEngine(
            cfg=self.cfg.backtest,
            target_names=target_names or [f"target_{i}" for i in range(predictions.shape[1])],
        )
        result = engine.run(predictions, prices, all_ts)
        report = engine.print_report(result)

        # Save report
        report_path = PROJECT_ROOT / self.cfg.log_dir / "backtest_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        logger.info("Backtest report saved to %s", report_path)

    def _evaluate(self, model, loader, criterion) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                adj = batch["adj"].to(self.device)
                y = batch["y"].to(self.device)
                logits = model(x, adj)
                preds = torch.sigmoid(logits)
                total_loss += criterion(preds, y).item()
                n += 1
        return total_loss / max(n, 1)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GNN-TCN Polymarket prediction model")
    parser.add_argument("--mode", choices=["train", "backtest", "predict"], default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--days-back", type=int, default=14, help="Days of history to use")
    parser.add_argument("--stride", type=int, default=30, help="Window stride in minutes")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Build config
    cfg = GNNConfig()
    cfg.model.epochs = args.epochs
    cfg.model.learning_rate = args.lr
    cfg.model.batch_size = args.batch_size
    cfg.model.window_size = args.window
    cfg.features.window_size = args.window
    cfg.features.step_minutes = args.step_minutes

    # Connect to ClickHouse
    logger.info("Connecting to ClickHouse...")
    client = get_clickhouse_client(cfg)

    # Build dataset
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days_back)

    logger.info("Building dataset from %s to %s...", start_date, end_date)
    dataset = DonbasTemporalDataset(
        client=client,
        config=cfg,
        start_date=start_date,
        end_date=end_date,
        stride_minutes=args.stride,
    )

    if len(dataset) < 10:
        logger.error("Not enough data: only %d samples. Need at least 10.", len(dataset))
        sys.exit(1)

    # Split: 70/15/15
    n = len(dataset)
    n_train = int(n * cfg.backtest.train_ratio)
    n_val = int(n * cfg.backtest.val_ratio)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.model.batch_size,
        shuffle=True, collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.model.batch_size,
        shuffle=False, collate_fn=collate_graph_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.model.batch_size,
        shuffle=False, collate_fn=collate_graph_batch,
    )

    # Build model
    model = GNNTCN(
        cfg=cfg.model,
        n_nodes=dataset.n_nodes,
        target_indices=dataset.target_indices,
    )

    logger.info(
        "Model: %d parameters, %d nodes, %d targets",
        sum(p.numel() for p in model.parameters()),
        dataset.n_nodes,
        len(dataset.target_indices),
    )

    trainer = Trainer(cfg)

    if args.mode == "train":
        model, train_losses, val_losses = trainer.train(train_loader, val_loader, model)
        platt = trainer.calibrate(model, val_loader)

        # Save Platt parameters
        checkpoint_dir = PROJECT_ROOT / cfg.model_save_dir
        torch.save({"a": platt.a.data, "b": platt.b.data}, checkpoint_dir / "platt.pt")

        # Auto-backtest on test set
        trainer.backtest(model, platt, test_loader, target_names=dataset.target_ids)

    elif args.mode == "backtest":
        if not args.checkpoint:
            logger.error("--checkpoint required for backtest mode")
            sys.exit(1)
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

        checkpoint_dir = pathlib.Path(args.checkpoint).parent
        platt = PlattScaling(n_targets=len(dataset.target_indices))
        platt_data = torch.load(checkpoint_dir / "platt.pt", weights_only=True)
        platt.a.data = platt_data["a"]
        platt.b.data = platt_data["b"]

        trainer.backtest(model, platt, test_loader, target_names=dataset.target_ids)

    elif args.mode == "predict":
        if not args.checkpoint:
            logger.error("--checkpoint required for predict mode")
            sys.exit(1)
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        model.eval()

        # Get latest single sample
        latest = dataset[len(dataset) - 1]
        x = latest["x"].unsqueeze(0).to(trainer.device)
        adj = latest["adj"].to(trainer.device)

        with torch.no_grad():
            logits = model(x, adj)

        checkpoint_dir = pathlib.Path(args.checkpoint).parent
        platt = PlattScaling(n_targets=len(dataset.target_indices))
        platt_data = torch.load(checkpoint_dir / "platt.pt", weights_only=True)
        platt.a.data = platt_data["a"]
        platt.b.data = platt_data["b"]

        probs = platt(logits.cpu()).squeeze(0).numpy()

        print("\n" + "=" * 60)
        print("LIVE PREDICTIONS")
        print("=" * 60)
        for i, sid in enumerate(dataset.target_ids):
            p = probs[i]
            market_p = latest["y"][i].item()
            edge = p - market_p
            direction = "BUY" if edge > 0.02 else ("SELL" if edge < -0.02 else "HOLD")
            kf = kelly_criterion(p, market_p) if direction != "HOLD" else 0.0
            print(
                f"  {sid:<20}  P(model)={p:.3f}  P(market)={market_p:.3f}  "
                f"edge={edge:+.3f}  → {direction}  kelly={kf:.3f}"
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
