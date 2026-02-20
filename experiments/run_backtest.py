#!/usr/bin/env python3
"""Out-of-sample backtest: prove the model generalizes to unseen data.

Evaluates Raw, PCA, and Orth-Supervised-VAE on the held-out temporal test
split (last 15% of markets by resolution date). Also runs a walk-forward
temporal backtest and a simple betting simulation.

Usage:
    python run_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from models.autoencoder import AutoencoderConfig, MarketAutoencoder
from models.statistics import (
    compute_condition_number,
    compute_vif,
    test_orthogonality,
    test_predictive_power,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits() -> dict:
    """Load train/val/test splits from features.npz with metadata."""
    data_dir = EXPERIMENT_DIR / "results" / "data"
    data = np.load(data_dir / "features.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [])

    splits = {}
    for name in ("train", "val", "test"):
        X = data[f"X_{name}"]
        y = data[f"y_{name}"]

        # Impute NaNs with column median (computed per-split for safety)
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                med = np.nanmedian(X[:, col])
                X[mask, col] = med if np.isfinite(med) else 0.0

        splits[name] = {"X": X, "y": y}

    # Drop zero-variance features (same as training pipeline)
    X_all = np.concatenate([splits[s]["X"] for s in ("train", "val", "test")])
    variances = np.var(X_all, axis=0)
    keep = variances >= 1e-10
    if not keep.all():
        dropped = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
        print(f"  Dropped {len(dropped)} zero-variance features: {dropped}")
        for name in splits:
            splits[name]["X"] = splits[name]["X"][:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    return splits, feature_names, metadata


def load_orth_model(input_dim: int) -> MarketAutoencoder:
    """Load the best orth-supervised-VAE model."""
    ckpt_dir = EXPERIMENT_DIR / "results" / "orth_b1_a1_g1_ckpt"
    ckpt_path = ckpt_dir / "best_model_supervised_vae.pt"

    config = AutoencoderConfig(
        input_dim=input_dim,
        embedding_dim=8,
        hidden_dims=(256, 128),
        dropout=0.1,
        model_type="supervised_vae",
        beta=1.0,
        alpha=1.0,
        gamma=1.0,
    )
    model = MarketAutoencoder(config)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute full classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_yes": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_no": float(f1_score(y_true, y_pred, pos_label=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def eval_representation(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict:
    """Evaluate a representation: train classifier on train, predict test."""
    # VIF and condition number on test data
    dim_names = [f"dim_{i}" for i in range(Z_test.shape[1])]
    vif = compute_vif(Z_test, dim_names)
    cond = compute_condition_number(Z_test)
    orth = test_orthogonality(Z_test)
    wald = test_predictive_power(Z_test, y_test, alpha=0.05)

    # Train logistic regression on train split, predict test
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    y_prob = clf.predict_proba(Z_test)[:, 1]

    metrics = eval_classification(y_test, y_pred, y_prob)
    metrics.update({
        "label": label,
        "dims": Z_test.shape[1],
        "vif_mean": vif.mean_vif,
        "vif_max": vif.max_vif,
        "vif_severe": vif.n_severe,
        "condition_number": cond,
        "orth_mean_cos": orth.mean_off_diagonal,
        "orth_max_cos": orth.max_off_diagonal,
        "wald_significant": wald.n_significant,
        "wald_total": len(wald.dimension_names),
        "wald_fraction": wald.n_significant / max(len(wald.dimension_names), 1),
    })
    return metrics


# ---------------------------------------------------------------------------
# Walk-forward temporal backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    X: np.ndarray,
    y: np.ndarray,
    model: MarketAutoencoder,
    scaler: StandardScaler,
    pca: PCA,
    min_train_size: int = 200,
    step_size: int = 50,
) -> dict:
    """Walk-forward backtest: train on first N, predict next K, slide forward.

    Uses the full dataset (already sorted by resolution date from the
    temporal split in extract.py).
    """
    n = X.shape[0]
    y_binary = (y >= 0.5).astype(int)

    raw_cumulative, pca_cumulative, svae_cumulative = [], [], []
    window_starts = []

    for start in range(min_train_size, n - step_size, step_size):
        train_end = start
        test_end = min(start + step_size, n)

        X_tr = X[:train_end]
        y_tr = y_binary[:train_end]
        X_te = X[train_end:test_end]
        y_te = y_binary[train_end:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        # Fit scaler on training window
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # Raw
        clf_raw = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf_raw.fit(X_tr_sc, y_tr)
        raw_acc = accuracy_score(y_te, clf_raw.predict(X_te_sc))
        raw_cumulative.append(raw_acc)

        # PCA
        pc = PCA(n_components=8, random_state=42)
        Z_tr_pca = pc.fit_transform(X_tr_sc)
        Z_te_pca = pc.transform(X_te_sc)
        clf_pca = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf_pca.fit(Z_tr_pca, y_tr)
        pca_acc = accuracy_score(y_te, clf_pca.predict(Z_te_pca))
        pca_cumulative.append(pca_acc)

        # Orth-SVAE (encode with pre-trained model, train new classifier)
        X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_sc, dtype=torch.float32)
        with torch.no_grad():
            Z_tr_svae = model.get_embedding(X_tr_t)
            Z_te_svae = model.get_embedding(X_te_t)
        clf_svae = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf_svae.fit(Z_tr_svae, y_tr)
        svae_acc = accuracy_score(y_te, clf_svae.predict(Z_te_svae))
        svae_cumulative.append(svae_acc)

        window_starts.append(train_end)

    return {
        "window_starts": window_starts,
        "raw_accuracy": raw_cumulative,
        "pca_accuracy": pca_cumulative,
        "svae_accuracy": svae_cumulative,
        "raw_mean": float(np.mean(raw_cumulative)) if raw_cumulative else 0.0,
        "pca_mean": float(np.mean(pca_cumulative)) if pca_cumulative else 0.0,
        "svae_mean": float(np.mean(svae_cumulative)) if svae_cumulative else 0.0,
        "n_windows": len(window_starts),
    }


# ---------------------------------------------------------------------------
# Betting simulation
# ---------------------------------------------------------------------------

def betting_simulation(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.70,
    bet_size: float = 100.0,
) -> dict:
    """Simulate a simple betting strategy on test data.

    Strategy: if model predicts P(Yes) > threshold, bet Yes.
              if model predicts P(Yes) < (1 - threshold), bet No.
              Otherwise, no bet.

    P&L: bet_size * (1/price - 1) if correct, -bet_size if wrong.
    (Simplified: assume price = predicted probability for the bet side.)
    """
    y_binary = (y_true >= 0.5).astype(int)
    bets = []
    pnl = []

    for i in range(len(y_prob)):
        p = y_prob[i]
        if p > threshold:
            # Bet Yes
            correct = y_binary[i] == 1
            payoff = bet_size * (1.0 / p - 1.0) if correct else -bet_size
            bets.append({"index": i, "side": "Yes", "prob": float(p), "correct": bool(correct), "pnl": float(payoff)})
            pnl.append(payoff)
        elif p < (1.0 - threshold):
            # Bet No
            correct = y_binary[i] == 0
            payoff = bet_size * (1.0 / (1.0 - p) - 1.0) if correct else -bet_size
            bets.append({"index": i, "side": "No", "prob": float(1.0 - p), "correct": bool(correct), "pnl": float(payoff)})
            pnl.append(payoff)

    total_bets = len(bets)
    wins = sum(1 for b in bets if b["correct"])
    total_pnl = sum(pnl)
    cumulative_pnl = np.cumsum(pnl).tolist() if pnl else []

    return {
        "threshold": threshold,
        "bet_size": bet_size,
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "win_rate": wins / max(total_bets, 1),
        "total_pnl": float(total_pnl),
        "avg_pnl_per_bet": float(total_pnl / max(total_bets, 1)),
        "max_drawdown": float(min(cumulative_pnl)) if cumulative_pnl else 0.0,
        "cumulative_pnl": cumulative_pnl,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(
    walk_forward: dict,
    betting_raw: dict,
    betting_svae: dict,
    output_dir: Path,
) -> None:
    """Generate backtest plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Walk-forward accuracy over time
    if walk_forward["n_windows"] > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(walk_forward["raw_accuracy"]))
        ax.plot(x, walk_forward["raw_accuracy"], label=f"Raw (mean={walk_forward['raw_mean']:.3f})", alpha=0.8)
        ax.plot(x, walk_forward["pca_accuracy"], label=f"PCA (mean={walk_forward['pca_mean']:.3f})", alpha=0.8)
        ax.plot(x, walk_forward["svae_accuracy"], label=f"Orth-SVAE (mean={walk_forward['svae_mean']:.3f})", alpha=0.8)
        ax.set_xlabel("Window")
        ax.set_ylabel("Accuracy")
        ax.set_title("Walk-Forward Temporal Backtest")
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "walk_forward_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_dir / 'walk_forward_accuracy.png'}")

    # 2. Cumulative P&L
    fig, ax = plt.subplots(figsize=(10, 5))
    if betting_raw["cumulative_pnl"]:
        ax.plot(betting_raw["cumulative_pnl"], label=f"Raw (PnL=${betting_raw['total_pnl']:.0f})", alpha=0.8)
    if betting_svae["cumulative_pnl"]:
        ax.plot(betting_svae["cumulative_pnl"], label=f"Orth-SVAE (PnL=${betting_svae['total_pnl']:.0f})", alpha=0.8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Betting Simulation: Cumulative P&L (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "betting_pnl.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'betting_pnl.png'}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    """Print side-by-side test set comparison."""
    print("\n" + "=" * 95)
    print("  OUT-OF-SAMPLE TEST SET COMPARISON")
    print("=" * 95)

    labels = [r["label"] for r in results]
    header = f"{'Metric':<30s}"
    for lbl in labels:
        header += f" {lbl:>18s}"
    print(f"\n{header}")
    print("-" * (30 + 19 * len(labels)))

    rows = [
        ("Dimensions", "dims", "d"),
        ("Accuracy", "accuracy", ".4f"),
        ("Balanced Accuracy", "balanced_accuracy", ".4f"),
        ("F1 (Yes class)", "f1_yes", ".4f"),
        ("F1 (No class)", "f1_no", ".4f"),
        ("AUC-ROC", "auc_roc", ".4f"),
        ("Mean VIF", "vif_mean", ".4f"),
        ("Max VIF", "vif_max", ".4f"),
        ("Condition Number", "condition_number", ".1f"),
        ("Sig Dims (Wald)", "wald_significant", "d"),
        ("Sig Fraction", "wald_fraction", ".0%"),
    ]

    for name, key, fmt in rows:
        line = f"  {name:<28s}"
        for r in results:
            val = r.get(key, "N/A")
            if val == "N/A":
                line += f" {'N/A':>18s}"
            elif fmt == "d":
                line += f" {int(val):>18d}"
            elif fmt == ".0%":
                line += f" {val:>17.0%} "
            else:
                line += f" {val:>18{fmt}}"
        print(line)

    print("-" * (30 + 19 * len(labels)))

    # Confusion matrices
    for r in results:
        cm = r.get("confusion_matrix")
        if cm:
            print(f"\n  Confusion Matrix ({r['label']}):")
            print(f"                  Predicted No  Predicted Yes")
            print(f"    Actual No     {cm[0][0]:>11d}  {cm[0][1]:>13d}")
            print(f"    Actual Yes    {cm[1][0]:>11d}  {cm[1][1]:>13d}")


def print_train_val_test(
    train_res: list[dict],
    val_res: list[dict],
    test_res: list[dict],
) -> None:
    """Print train/val/test comparison for generalization analysis."""
    print("\n" + "=" * 95)
    print("  GENERALIZATION: TRAIN vs VAL vs TEST (Orth-SVAE)")
    print("=" * 95)

    # Find orth-svae in each list
    def find(lst, label_prefix):
        for r in lst:
            if "Orth" in r["label"]:
                return r
        return None

    tr = find(train_res, "Orth")
    va = find(val_res, "Orth")
    te = find(test_res, "Orth")

    if not all([tr, va, te]):
        print("  Cannot build comparison (missing splits)")
        return

    header = f"{'Metric':<30s} {'Train':>12s} {'Val':>12s} {'Test':>12s} {'Overfit?':>10s}"
    print(f"\n{header}")
    print("-" * 78)

    rows = [
        ("Accuracy", "accuracy", ".4f"),
        ("Balanced Accuracy", "balanced_accuracy", ".4f"),
        ("F1 (Yes)", "f1_yes", ".4f"),
        ("F1 (No)", "f1_no", ".4f"),
        ("AUC-ROC", "auc_roc", ".4f"),
    ]

    for name, key, fmt in rows:
        tr_v = tr[key]
        va_v = va[key]
        te_v = te[key]
        # Overfitting: train >> test by > 5pp
        overfit = "YES" if (tr_v - te_v) > 0.05 else "no"
        print(f"  {name:<28s} {tr_v:>12{fmt}} {va_v:>12{fmt}} {te_v:>12{fmt}} {overfit:>10s}")

    print("-" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Out-of-Sample Backtest")
    print("  Prove the model generalizes to unseen temporal data")
    print("=" * 70)

    total_start = time.perf_counter()

    # --- Load data ---
    print("\n  Loading temporal splits...")
    splits, feature_names, metadata = load_splits()
    X_train, y_train_raw = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val_raw = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test_raw = splits["test"]["X"], splits["test"]["y"]

    y_train = (y_train_raw >= 0.5).astype(int)
    y_val = (y_val_raw >= 0.5).astype(int)
    y_test = (y_test_raw >= 0.5).astype(int)

    print(f"  Train: {X_train.shape[0]} samples (Yes={y_train.sum()}, No={len(y_train)-y_train.sum()})")
    print(f"  Val:   {X_val.shape[0]} samples (Yes={y_val.sum()}, No={len(y_val)-y_val.sum()})")
    print(f"  Test:  {X_test.shape[0]} samples (Yes={y_test.sum()}, No={len(y_test)-y_test.sum()})")

    # --- Fit scaler on TRAIN only ---
    print("\n  Fitting StandardScaler on train split only...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # --- Load model ---
    print("  Loading Orth-Supervised-VAE model (b1_a1_g1)...")
    model = load_orth_model(input_dim=X_train_sc.shape[1])

    # --- Encode all splits ---
    print("  Encoding all splits through model...")
    with torch.no_grad():
        Z_train_svae = model.get_embedding(torch.tensor(X_train_sc, dtype=torch.float32))
        Z_val_svae = model.get_embedding(torch.tensor(X_val_sc, dtype=torch.float32))
        Z_test_svae = model.get_embedding(torch.tensor(X_test_sc, dtype=torch.float32))

    # --- PCA ---
    print("  Fitting PCA(n_components=8) on train split...")
    pca = PCA(n_components=8, random_state=42)
    Z_train_pca = pca.fit_transform(X_train_sc)
    Z_val_pca = pca.transform(X_val_sc)
    Z_test_pca = pca.transform(X_test_sc)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # =====================================================================
    # PART 1: Test set evaluation
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Test Set Evaluation (train on train, predict test)")
    print("=" * 70)

    print("\n  Evaluating Raw features on TEST...")
    test_raw = eval_representation(X_train_sc, X_test_sc, y_train, y_test, "Raw (25D)")

    print("  Evaluating PCA on TEST...")
    test_pca = eval_representation(Z_train_pca, Z_test_pca, y_train, y_test, "PCA (8D)")

    print("  Evaluating Orth-SVAE on TEST...")
    test_svae = eval_representation(Z_train_svae, Z_test_svae, y_train, y_test, "Orth-SVAE (8D)")

    print_comparison([test_raw, test_pca, test_svae])

    # =====================================================================
    # PART 2: Train/Val/Test generalization check
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Generalization Check (same model, all splits)")
    print("=" * 70)

    # Evaluate on train (in-sample)
    print("\n  Evaluating on TRAIN (in-sample)...")
    train_raw = eval_representation(X_train_sc, X_train_sc, y_train, y_train, "Raw (25D)")
    train_pca = eval_representation(Z_train_pca, Z_train_pca, y_train, y_train, "PCA (8D)")
    train_svae = eval_representation(Z_train_svae, Z_train_svae, y_train, y_train, "Orth-SVAE (8D)")

    # Evaluate on val
    print("  Evaluating on VAL...")
    val_raw = eval_representation(X_train_sc, X_val_sc, y_train, y_val, "Raw (25D)")
    val_pca = eval_representation(Z_train_pca, Z_val_pca, y_train, y_val, "PCA (8D)")
    val_svae = eval_representation(Z_train_svae, Z_val_svae, y_train, y_val, "Orth-SVAE (8D)")

    print_train_val_test(
        [train_raw, train_pca, train_svae],
        [val_raw, val_pca, val_svae],
        [test_raw, test_pca, test_svae],
    )

    # =====================================================================
    # PART 3: Walk-forward temporal backtest
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Walk-Forward Temporal Backtest")
    print("=" * 70)

    # Concatenate all data in temporal order for walk-forward
    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train_raw, y_val_raw, y_test_raw])

    print(f"\n  Full dataset: {X_all.shape[0]} markets (temporal order)")
    print("  Walk-forward: min_train=200, step=50")
    walk_forward = walk_forward_backtest(X_all, y_all, model, scaler, pca)
    print(f"  Windows: {walk_forward['n_windows']}")
    print(f"  Raw mean accuracy:       {walk_forward['raw_mean']:.4f}")
    print(f"  PCA mean accuracy:       {walk_forward['pca_mean']:.4f}")
    print(f"  Orth-SVAE mean accuracy: {walk_forward['svae_mean']:.4f}")

    # =====================================================================
    # PART 4: Betting simulation
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Betting Simulation (Test Set)")
    print("=" * 70)

    # Get predicted probabilities from test set
    clf_raw = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf_raw.fit(X_train_sc, y_train)
    prob_raw = clf_raw.predict_proba(X_test_sc)[:, 1]

    clf_svae = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf_svae.fit(Z_train_svae, y_train)
    prob_svae = clf_svae.predict_proba(Z_test_svae)[:, 1]

    for threshold in (0.70, 0.80):
        print(f"\n  --- Threshold: {threshold:.0%} ---")
        bet_raw = betting_simulation(prob_raw, y_test_raw, threshold=threshold)
        bet_svae = betting_simulation(prob_svae, y_test_raw, threshold=threshold)

        print(f"  Raw:       {bet_raw['total_bets']} bets, "
              f"win rate {bet_raw['win_rate']:.0%}, "
              f"P&L ${bet_raw['total_pnl']:+.0f} "
              f"(${bet_raw['avg_pnl_per_bet']:+.1f}/bet)")
        print(f"  Orth-SVAE: {bet_svae['total_bets']} bets, "
              f"win rate {bet_svae['win_rate']:.0%}, "
              f"P&L ${bet_svae['total_pnl']:+.0f} "
              f"(${bet_svae['avg_pnl_per_bet']:+.1f}/bet)")

    # Use 70% threshold for the plots
    bet_raw_70 = betting_simulation(prob_raw, y_test_raw, threshold=0.70)
    bet_svae_70 = betting_simulation(prob_svae, y_test_raw, threshold=0.70)

    # =====================================================================
    # Save results and plots
    # =====================================================================
    output_dir = EXPERIMENT_DIR / "results" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Generating plots...")
    plot_results(walk_forward, bet_raw_70, bet_svae_70, output_dir)

    report = {
        "experiment": "out_of_sample_backtest",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_sizes": {
            "train": X_train.shape[0],
            "val": X_val.shape[0],
            "test": X_test.shape[0],
        },
        "test_results": {
            "raw": test_raw,
            "pca": test_pca,
            "orth_svae": test_svae,
        },
        "train_results": {
            "raw": train_raw,
            "pca": train_pca,
            "orth_svae": train_svae,
        },
        "val_results": {
            "raw": val_raw,
            "pca": val_pca,
            "orth_svae": val_svae,
        },
        "walk_forward": {k: v for k, v in walk_forward.items() if k != "cumulative_pnl"},
        "betting_70": {
            "raw": {k: v for k, v in bet_raw_70.items() if k != "cumulative_pnl"},
            "orth_svae": {k: v for k, v in bet_svae_70.items() if k != "cumulative_pnl"},
        },
    }

    report_path = output_dir / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.perf_counter() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print("  BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Test set accuracy:  Raw={test_raw['accuracy']:.4f}  PCA={test_pca['accuracy']:.4f}  Orth-SVAE={test_svae['accuracy']:.4f}")
    print(f"  Test set bal. acc:  Raw={test_raw['balanced_accuracy']:.4f}  PCA={test_pca['balanced_accuracy']:.4f}  Orth-SVAE={test_svae['balanced_accuracy']:.4f}")
    print(f"  Test set AUC:       Raw={test_raw['auc_roc']:.4f}  PCA={test_pca['auc_roc']:.4f}  Orth-SVAE={test_svae['auc_roc']:.4f}")
    print(f"  Walk-forward mean:  Raw={walk_forward['raw_mean']:.4f}  PCA={walk_forward['pca_mean']:.4f}  Orth-SVAE={walk_forward['svae_mean']:.4f}")

    overfit_gap = train_svae["accuracy"] - test_svae["accuracy"]
    print(f"\n  Overfit gap (train-test acc): {overfit_gap:+.4f} ({'OK' if overfit_gap < 0.05 else 'OVERFITTING'})")

    print(f"\n  Report: {report_path}")
    print(f"  Figures: {output_dir}")
    print(f"  Completed in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
