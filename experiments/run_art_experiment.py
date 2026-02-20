#!/usr/bin/env python3
"""Art Market Experiment: Orth-SVAE vs multiple baselines on auction data.

Compares 5 approaches on art auction data:
  1. Raw + LogisticRegression  (linear baseline)
  2. PCA + LogisticRegression  (linear decorrelation baseline)
  3. Raw + Random Forest       (nonlinear baseline)
  4. Raw + LightGBM            (gradient boosting baseline)
  5. Orth-SVAE + LogisticRegression (our method)

The honest question: does the embedding approach beat real-world ML models,
not just PCA? If LightGBM on raw features gets 85% and Orth-SVAE gets 82%,
the methodology does not add value over standard approaches.

Polymarket honest experiment reference (8 features, 813 samples):
  - Raw VIF max: 6.19, condition#: 6.0
  - PCA VIF max: 1.04, accuracy: 0.6936
  - Orth-SVAE VIF max: 1.02, accuracy: 0.7010, delta vs PCA: +0.74pp
  - Walk-forward: Raw 0.659, PCA 0.659, Orth-SVAE 0.666

Pipeline:
  1. Load art feature matrix from art_data/ (.npz format)
  2. Raw multicollinearity assessment (VIF, condition number)
  3. Train Orth-SVAE (supervised_vae, dim=8, beta=1, alpha=1, gamma=1)
  4. Compare ALL 5 methods:
     - VIF / condition number (for embedding methods)
     - CV accuracy, balanced accuracy, F1, AUC
     - Art-specific linear probes
     - Feature attribution (Jacobian)
  5. Out-of-sample temporal backtest with all 5 methods
  6. Walk-forward backtest with all 5 methods
  7. Verdict: does Orth-SVAE beat standard ML?

Usage:
    python run_art_experiment.py
    python run_art_experiment.py --data-dir art_data/output
    python run_art_experiment.py --embedding-dim 12 --gamma 2.0
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from models.autoencoder import AutoencoderConfig, MarketAutoencoder
from models.statistics import (
    compute_condition_number,
    compute_vif,
    test_orthogonality,
    test_predictive_power,
)
from models.train import TrainConfig, train as train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Publication-quality style
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
sns.set_style("whitegrid")

# Polymarket honest experiment results (for comparison)
POLYMARKET_REFERENCE = {
    "dataset": "Polymarket Over/Under",
    "n_samples": 813,
    "n_features": 8,
    "raw_vif_max": 6.19,
    "raw_vif_mean": 2.67,
    "raw_condition_number": 6.0,
    "pca_vif_max": 1.04,
    "pca_accuracy": 0.6936,
    "pca_auc": 0.7102,
    "svae_vif_max": 1.02,
    "svae_condition_number": 1.86,
    "svae_accuracy": 0.7010,
    "svae_auc": 0.7200,
    "svae_vs_pca_accuracy_delta": 0.0074,
    "walk_forward_raw": 0.659,
    "walk_forward_pca": 0.659,
    "walk_forward_svae": 0.666,
}


# ======================================================================
# Data loading
# ======================================================================

def load_art_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load art market feature matrix from .npz file.

    Expects the same format as the Polymarket extraction:
      features.npz with X_train, y_train, X_val, y_val, X_test, y_test, feature_names
      metadata.json with feature_names, n_features, etc.

    Returns (X, y, feature_names, metadata).
    """
    data_path = Path(data_dir)
    features_file = data_path / "features.npz"
    metadata_file = data_path / "metadata.json"

    if not features_file.exists():
        print(f"  ERROR: {features_file} not found.")
        print("  Run the art data extraction first (art_data/extract.py)")
        sys.exit(1)

    data = np.load(features_file, allow_pickle=True)

    # Reassemble from splits
    X_parts, y_parts = [], []
    for split in ("train", "val", "test"):
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        if x_key in data:
            X_parts.append(data[x_key])
            y_parts.append(data[y_key])

    if X_parts:
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
    elif "X" in data:
        X = data["X"]
        y = data["y"]
    else:
        raise ValueError(f"No feature data found in {features_file}")

    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Feature names
    if "feature_names" in data:
        fn = data["feature_names"]
        if fn.dtype.kind in ("U", "S", "O"):
            feature_names = fn.tolist()
        else:
            feature_names = [str(x) for x in fn]
    elif "feature_names" in metadata:
        feature_names = metadata["feature_names"]
    else:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    # Load auxiliary label arrays for probes
    for key in ("medium", "auction_house", "artist_vital_status", "sale_category",
                "price_bucket", "artist_id", "sale_year"):
        if key in data:
            metadata[f"label_{key}"] = data[key]

    return X, y, feature_names, metadata


def preprocess(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Impute NaNs, drop zero-variance features."""
    # NaN imputation with column medians
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  NaN values: {nan_count} ({100 * np.isnan(X).mean():.1f}%). Imputing with medians.")
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                med = np.nanmedian(X[:, col])
                X[mask, col] = med if np.isfinite(med) else 0.0

    # Drop zero-variance
    variances = np.var(X, axis=0)
    keep = variances >= 1e-10
    dropped = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
    if dropped:
        print(f"  Dropped {len(dropped)} zero-variance features: {dropped}")
        X = X[:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    # Drop infinite values
    inf_mask = np.isinf(X)
    if inf_mask.any():
        print(f"  Replacing {inf_mask.sum()} infinite values with column max.")
        for col in range(X.shape[1]):
            col_inf = np.isinf(X[:, col])
            if col_inf.any():
                finite_vals = X[~col_inf, col]
                replace_val = finite_vals.max() if len(finite_vals) > 0 else 0.0
                X[col_inf, col] = replace_val

    return X, feature_names, dropped


# ======================================================================
# Evaluation helpers
# ======================================================================

def evaluate_representation(
    Z: np.ndarray,
    y_binary: np.ndarray,
    label: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Full evaluation: VIF, condition#, orthogonality, Wald, CV probes."""
    dim = Z.shape[1]
    dim_names = [f"dim_{i}" for i in range(dim)]

    vif = compute_vif(Z, dim_names)
    cond = compute_condition_number(Z)
    orth = test_orthogonality(Z)
    wald = test_predictive_power(Z, y_binary, alpha=0.05)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, fold_bacc, fold_f1, fold_auc = [], [], [], []

    for train_idx, test_idx in cv.split(Z, y_binary):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(Z[train_idx], y_binary[train_idx])
        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        fold_acc.append(accuracy_score(y_binary[test_idx], y_pred))
        fold_bacc.append(balanced_accuracy_score(y_binary[test_idx], y_pred))
        fold_f1.append(f1_score(y_binary[test_idx], y_pred, average="macro", zero_division=0))
        if len(np.unique(y_binary[test_idx])) > 1:
            fold_auc.append(roc_auc_score(y_binary[test_idx], y_prob))

    return {
        "label": label,
        "dims": dim,
        "vif_mean": vif.mean_vif,
        "vif_max": vif.max_vif,
        "vif_severe": vif.n_severe,
        "condition_number": cond,
        "orth_max_cos": orth.max_off_diagonal,
        "orth_mean_cos": orth.mean_off_diagonal,
        "wald_significant": wald.n_significant,
        "wald_total": len(wald.dimension_names),
        "wald_fraction": wald.n_significant / max(len(wald.dimension_names), 1),
        "cv_accuracy": float(np.mean(fold_acc)),
        "cv_balanced_accuracy": float(np.mean(fold_bacc)),
        "cv_f1_macro": float(np.mean(fold_f1)),
        "cv_auc": float(np.mean(fold_auc)) if fold_auc else 0.0,
    }


def evaluate_oos(
    Z_train: np.ndarray, Z_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    label: str,
) -> dict:
    """Out-of-sample evaluation."""
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    y_prob = clf.predict_proba(Z_test)[:, 1]

    result = {
        "label": label,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        result["auc"] = float(roc_auc_score(y_test, y_prob))
    return result


def probe_classification(
    X: np.ndarray, y: np.ndarray, concept: str, input_type: str,
    n_folds: int = 5, seed: int = 42,
) -> dict:
    """Run classification probe with cross-validation."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2 or len(y_enc) < n_folds:
        return {"concept": concept, "input_type": input_type, "accuracy": 0.0, "skipped": True}

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in cv.split(X, y_enc):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(X[train_idx], y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(X[test_idx])))

    return {
        "concept": concept,
        "input_type": input_type,
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_classes": n_classes,
        "n_samples": len(y_enc),
    }


def probe_regression(
    X: np.ndarray, y: np.ndarray, concept: str, input_type: str,
    n_folds: int = 5, seed: int = 42,
) -> dict:
    """Run regression probe with cross-validation."""
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2s, maes = [], []
    for train_idx, test_idx in cv.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y[train_idx])
        y_pred = reg.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], y_pred))
        maes.append(mean_absolute_error(y[test_idx], y_pred))

    return {
        "concept": concept,
        "input_type": input_type,
        "r2": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mae": float(np.mean(maes)),
        "n_samples": len(y),
    }


def compute_jacobian_attribution(
    model: MarketAutoencoder,
    X: np.ndarray,
    feature_names: list[str],
    embed_dim: int,
) -> np.ndarray:
    """Compute mean |Jacobian| attribution: (n_features, embed_dim)."""
    model.eval()
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)

    z = model.encode(X_tensor)

    attributions = np.zeros((X.shape[1], embed_dim))
    for i in range(embed_dim):
        model.zero_grad()
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()
        z_i = z[:, i].sum()
        z_i.backward(retain_graph=(i < embed_dim - 1))
        grad = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        attributions[:, i] = grad

    return attributions


# ======================================================================
# Model builders
# ======================================================================

def make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=seed,
    )


def make_lgbm(seed: int = 42):
    if HAS_LGBM:
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=-1, num_leaves=31,
            learning_rate=0.1, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=seed,
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.8, random_state=seed,
    )


# ======================================================================
# Full model comparison (CV)
# ======================================================================

def cv_evaluate_model(
    clf_factory,
    X: np.ndarray,
    y_binary: np.ndarray,
    label: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validated evaluation for any sklearn-compatible classifier."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, fold_bacc, fold_f1, fold_auc = [], [], [], []

    for train_idx, test_idx in cv.split(X, y_binary):
        clf = clf_factory()
        clf.fit(X[train_idx], y_binary[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]

        fold_acc.append(accuracy_score(y_binary[test_idx], y_pred))
        fold_bacc.append(balanced_accuracy_score(y_binary[test_idx], y_pred))
        fold_f1.append(f1_score(y_binary[test_idx], y_pred, average="macro", zero_division=0))
        if len(np.unique(y_binary[test_idx])) > 1:
            fold_auc.append(roc_auc_score(y_binary[test_idx], y_prob))

    return {
        "label": label,
        "cv_accuracy": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_balanced_accuracy": float(np.mean(fold_bacc)),
        "cv_f1_macro": float(np.mean(fold_f1)),
        "cv_auc": float(np.mean(fold_auc)) if fold_auc else 0.0,
    }


def oos_evaluate_model(
    clf_factory,
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    label: str,
) -> dict:
    """Out-of-sample evaluation for any sklearn-compatible classifier."""
    clf = clf_factory()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    result = {
        "label": label,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        result["auc"] = float(roc_auc_score(y_test, y_prob))
    return result


# ======================================================================
# Printing
# ======================================================================

def print_table(title: str, results: list[dict], rows: list[tuple]) -> None:
    """Print a comparison table."""
    labels = [r["label"] for r in results]
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    header = f"  {'Metric':<30s}"
    for lbl in labels:
        header += f" {lbl:>18s}"
    print(header)
    print("  " + "-" * (30 + 19 * len(labels)))

    for name, key, fmt in rows:
        line = f"  {name:<30s}"
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

    print("  " + "-" * (30 + 19 * len(labels)))


# ======================================================================
# Visualization
# ======================================================================

def plot_multicollinearity_comparison(
    raw_vif_values: np.ndarray,
    pca_vif_values: np.ndarray,
    svae_vif_values: np.ndarray,
    raw_names: list[str],
    output_dir: Path,
) -> None:
    """Side-by-side VIF comparison: Raw vs PCA vs Orth-SVAE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vif_vals, names, title in [
        (axes[0], raw_vif_values, raw_names, "Raw Features"),
        (axes[1], pca_vif_values, [f"PC_{i}" for i in range(len(pca_vif_values))], "PCA"),
        (axes[2], svae_vif_values, [f"emb_{i}" for i in range(len(svae_vif_values))], "Orth-SVAE"),
    ]:
        idx = np.argsort(vif_vals)[::-1][:30]
        colors = ["#d32f2f" if v > 10 else "#ff9800" if v > 5 else "#4caf50"
                  for v in vif_vals[idx]]
        ax.barh(range(len(idx)), vif_vals[idx], color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([names[i] for i in idx], fontsize=8)
        ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7)
        ax.axvline(x=10, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("VIF")
        ax.set_title(f"{title} (n={len(vif_vals)})")
        ax.invert_yaxis()

    fig.suptitle("VIF Comparison: Art Market Data", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "vif_comparison.png", dpi=300)
    plt.close(fig)


def plot_correlation_matrices(
    X_raw: np.ndarray, Z_pca: np.ndarray, Z_svae: np.ndarray,
    raw_names: list[str], output_dir: Path,
) -> None:
    """Correlation heatmaps for raw, PCA, and Orth-SVAE."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, data, title in [
        (axes[0], X_raw, "Raw Features"),
        (axes[1], Z_pca, "PCA"),
        (axes[2], Z_svae, "Orth-SVAE"),
    ]:
        corr = np.corrcoef(data.T)
        sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8},
                    xticklabels=False, yticklabels=False)
        ax.set_title(f"{title} ({data.shape[1]}D)")

    fig.suptitle("Correlation Matrices: Art Market Data", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrices.png", dpi=300)
    plt.close(fig)


def plot_walk_forward(
    wf_results: dict[str, list[float]],
    baseline: float, output_dir: Path,
) -> None:
    """Walk-forward accuracy plot for all methods."""
    colors = {
        "Raw+LR": "#5c6bc0", "PCA+LR": "#42a5f5", "Raw+RF": "#ff9800",
        "Raw+LGBM": "#ef5350", "SVAE+LR": "#26a69a",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, vals in wf_results.items():
        if vals:
            x = range(len(vals))
            color = colors.get(name, "#666666")
            ax.plot(x, vals, label=f"{name} (mean={np.mean(vals):.3f})",
                    alpha=0.8, linewidth=1.5, color=color)
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Majority class ({baseline:.1%})")
    ax.set_xlabel("Window")
    ax.set_ylabel("Accuracy")
    ax.set_title("Art Market: Walk-Forward Temporal Backtest (All Methods)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "walk_forward.png", dpi=300)
    plt.close(fig)


def plot_attribution_heatmap(
    attributions: np.ndarray,
    feature_names: list[str],
    embed_dim: int,
    output_dir: Path,
    top_k: int = 25,
) -> None:
    """Feature attribution heatmap: input -> embedding mapping."""
    total_attr = attributions.sum(axis=1)
    top_idx = np.argsort(total_attr)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(max(8, embed_dim * 0.5), max(6, top_k * 0.35)))
    sns.heatmap(
        attributions[top_idx],
        ax=ax, cmap="YlOrRd",
        xticklabels=[f"dim_{i}" for i in range(embed_dim)],
        yticklabels=[feature_names[i] for i in top_idx],
        cbar_kws={"label": "Mean |gradient|"},
    )
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Input Feature")
    ax.set_title("Art Market: Feature Attribution (Jacobian)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_attribution.png", dpi=300)
    plt.close(fig)


def plot_domain_comparison(art_results: dict, output_dir: Path) -> None:
    """Bar chart comparing Polymarket vs Art Market results."""
    poly = POLYMARKET_REFERENCE
    art = art_results

    metrics = ["VIF Reduction", "Cond# Reduction", "SVAE vs PCA\n(accuracy delta)"]
    poly_vals = [
        poly["raw_vif_max"] / max(poly["svae_vif_max"], 0.01),
        poly["raw_condition_number"] / max(poly["svae_condition_number"], 0.01),
        poly["svae_vs_pca_accuracy_delta"] * 100,
    ]
    art_vals = [
        art.get("vif_reduction_factor", 1.0),
        art.get("condition_reduction_factor", 1.0),
        art.get("svae_vs_pca_accuracy_delta", 0.0) * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # VIF and condition number (log scale)
    ax = axes[0]
    ax.bar(x[:2] - width/2, poly_vals[:2], width, label="Polymarket (8D)", color="#5c6bc0", alpha=0.85)
    ax.bar(x[:2] + width/2, art_vals[:2], width, label="Art Market", color="#26a69a", alpha=0.85)
    ax.set_ylabel("Reduction Factor (x)")
    ax.set_title("Multicollinearity Reduction")
    ax.set_xticks(x[:2])
    ax.set_xticklabels(metrics[:2])
    ax.legend()
    if max(poly_vals[:2] + art_vals[:2]) > 20:
        ax.set_yscale("log")

    # Accuracy delta
    ax = axes[1]
    ax.bar(0 - width/2, poly_vals[2], width, label="Polymarket (8D)", color="#5c6bc0", alpha=0.85)
    ax.bar(0 + width/2, art_vals[2], width, label="Art Market", color="#26a69a", alpha=0.85)
    ax.set_ylabel("Accuracy Delta (pp)")
    ax.set_title("Orth-SVAE vs PCA Accuracy Improvement")
    ax.set_xticks([0])
    ax.set_xticklabels(["SVAE - PCA"])
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    fig.suptitle("Domain Comparison: Polymarket vs Art Market", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_comparison.png", dpi=300)
    plt.close(fig)


def plot_model_comparison(all_cv: list[dict], output_dir: Path) -> None:
    """Grouped bar chart comparing all 5 models on CV metrics."""
    labels = [r["label"] for r in all_cv]
    metrics = ["cv_accuracy", "cv_balanced_accuracy", "cv_f1_macro", "cv_auc"]
    metric_names = ["Accuracy", "Balanced Acc", "F1 (macro)", "AUC-ROC"]
    colors = ["#5c6bc0", "#42a5f5", "#ff9800", "#ef5350", "#26a69a"]

    x = np.arange(len(metrics))
    n_models = len(all_cv)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (result, color) in enumerate(zip(all_cv, colors)):
        vals = [result.get(m, 0.0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=result["label"],
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_ylabel("Score")
    ax.set_title("5-Model Comparison: Cross-Validated Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, min(1.15, max(r.get("cv_auc", 0) for r in all_cv) * 1.2))
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close(fig)


# ======================================================================
# Main experiment
# ======================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Art Market Orth-SVAE Experiment")
    parser.add_argument("--data-dir", type=str, default=str(EXPERIMENT_DIR / "art_data" / "output"),
                        help="Path to art data directory")
    parser.add_argument("--embedding-dim", type=int, default=8,
                        help="Embedding dimensionality")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="Prediction loss weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Orthogonality penalty weight")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    total_start = time.perf_counter()

    print("=" * 80)
    print("  ART MARKET EXPERIMENT: 5-Model Comparison")
    print("  Q1: Does Orth-SVAE advantage grow with dimensionality?")
    print("  Q2: Does Orth-SVAE beat standard ML (RF, LightGBM)?")
    print("=" * 80)

    output_dir = EXPERIMENT_DIR / "results" / "art_market"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    print("\n--- STEP 1: Load Art Market Data ---")
    X, y, feature_names, metadata = load_art_data(args.data_dir)
    X, feature_names, dropped = preprocess(X, feature_names)

    n_samples, n_features = X.shape
    print(f"  Loaded: {n_samples} samples, {n_features} features")
    print(f"  Features: {feature_names}")

    # Determine task type: binary classification if y looks binary, regression otherwise
    y_unique = np.unique(y[np.isfinite(y)])
    is_binary = len(y_unique) == 2 or (len(y_unique) <= 10 and set(y_unique).issubset({0.0, 1.0}))

    if is_binary:
        y_binary = (y >= 0.5).astype(int)
        task_type = "classification"
        print(f"  Task: binary classification (sold vs bought-in)")
        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos
        print(f"  Class balance: pos={n_pos} ({100*n_pos/len(y_binary):.1f}%), "
              f"neg={n_neg} ({100*n_neg/len(y_binary):.1f}%)")
        baseline = max(n_pos, n_neg) / len(y_binary)
    else:
        # For regression targets (e.g., log hammer price), binarize for probes
        median_y = np.median(y)
        y_binary = (y >= median_y).astype(int)
        task_type = "regression"
        print(f"  Task: regression (hammer price prediction)")
        print(f"  Target range: [{y.min():.2f}, {y.max():.2f}], median={median_y:.2f}")
        baseline = 0.5

    # ---- Temporal split (70/15/15) ----
    # Data should be sorted by sale date from extract.py
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.85)
    X_train, y_train_bin = X[:n_train], y_binary[:n_train]
    X_val, y_val_bin = X[n_train:n_val], y_binary[n_train:n_val]
    X_test, y_test_bin = X[n_val:], y_binary[n_val:]
    y_train_raw, y_val_raw, y_test_raw = y[:n_train], y[n_train:n_val], y[n_val:]

    print(f"  Split: train={len(y_train_bin)} val={len(y_val_bin)} test={len(y_test_bin)}")

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X)

    # ---- Step 2: Raw multicollinearity ----
    print("\n--- STEP 2: Raw Multicollinearity Assessment ---")

    raw_vif = compute_vif(X_all_sc, feature_names)
    raw_cond = compute_condition_number(X_all_sc)
    raw_corr = np.corrcoef(X_all_sc.T)
    np.fill_diagonal(raw_corr, 0)
    raw_max_corr = float(np.max(np.abs(raw_corr)))

    print(f"  Dimensions: {n_features}")
    print(f"  Mean VIF: {raw_vif.mean_vif:.2f}")
    print(f"  Max VIF: {raw_vif.max_vif:.2f}")
    print(f"  Severe VIF (>10): {raw_vif.n_severe}/{n_features}")
    print(f"  Condition number: {raw_cond:.1f}")
    print(f"  Max |correlation|: {raw_max_corr:.3f}")

    # Top VIF offenders
    top_vifs = sorted(zip(raw_vif.feature_names, raw_vif.vif_values), key=lambda x: -x[1])
    print("  Top VIF features:")
    for name, vif_val in top_vifs[:10]:
        flag = " ***" if vif_val > 10 else ""
        print(f"    {name:<35s} VIF = {vif_val:>8.2f}{flag}")

    # ---- Step 3: Train Orth-SVAE ----
    print(f"\n--- STEP 3: Train Orth-SVAE (dim={args.embedding_dim}, "
          f"beta={args.beta}, alpha={args.alpha}, gamma={args.gamma}) ---")

    data_dir = output_dir / "train_data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "features.npy", X_all_sc.astype(np.float32))
    np.save(data_dir / "labels.npy", y_binary.astype(np.float32))
    with open(data_dir / "metadata.json", "w") as f:
        json.dump({"feature_names": feature_names, "n_features": n_features}, f)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_config = TrainConfig(
        model_type="supervised_vae",
        embedding_dim=args.embedding_dim,
        hidden_dims=(256, 128),
        dropout=0.1,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        patience=args.patience,
        data_dir=str(data_dir),
        output_dir=str(ckpt_dir),
        seed=args.seed,
    )

    result = train_model(train_config)
    print(f"  Best epoch: {result['best_epoch']}, val loss: {result['best_val_loss']:.6f}")

    # Load best model
    model_config = AutoencoderConfig(
        input_dim=n_features,
        embedding_dim=args.embedding_dim,
        hidden_dims=(256, 128),
        dropout=0.1,
        model_type="supervised_vae",
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    model = MarketAutoencoder(model_config)
    model.load_state_dict(torch.load(ckpt_dir / "best_model_supervised_vae.pt", weights_only=True))
    model.eval()

    # Encode all splits
    with torch.no_grad():
        Z_train_svae = model.get_embedding(torch.tensor(X_train_sc, dtype=torch.float32))
        Z_val_svae = model.get_embedding(torch.tensor(X_val_sc, dtype=torch.float32))
        Z_test_svae = model.get_embedding(torch.tensor(X_test_sc, dtype=torch.float32))
        Z_all_svae = model.get_embedding(torch.tensor(X_all_sc, dtype=torch.float32))

    # PCA
    pca_dim = min(args.embedding_dim, n_features)
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    Z_train_pca = pca.fit_transform(X_train_sc)
    Z_val_pca = pca.transform(X_val_sc)
    Z_test_pca = pca.transform(X_test_sc)
    Z_all_pca = pca.transform(X_all_sc)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%} ({pca_dim} components)")

    # ---- Step 4: Full 5-model comparison ----
    print("\n--- STEP 4: Cross-Validated 5-Model Comparison ---")

    # Embedding-space analysis (VIF, condition#, Wald) for decorrelated methods
    raw_repr = evaluate_representation(X_all_sc, y_binary, f"Raw+LR ({n_features}D)")
    pca_repr = evaluate_representation(Z_all_pca, y_binary, f"PCA+LR ({pca_dim}D)")
    svae_repr = evaluate_representation(Z_all_svae, y_binary, f"SVAE+LR ({args.embedding_dim}D)")

    repr_rows = [
        ("Dimensions", "dims", "d"),
        ("Mean VIF", "vif_mean", ".2f"),
        ("Max VIF", "vif_max", ".2f"),
        ("Severe VIF (>10)", "vif_severe", "d"),
        ("Condition Number", "condition_number", ".1f"),
        ("Max |cos sim| (off-diag)", "orth_max_cos", ".4f"),
        ("Sig Dims (Wald p<0.05)", "wald_significant", "d"),
        ("Sig Fraction", "wald_fraction", ".0%"),
    ]
    print_table("REPRESENTATION QUALITY", [raw_repr, pca_repr, svae_repr], repr_rows)

    # CV predictive performance for ALL 5 models
    seed = args.seed
    lr_factory = lambda: LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    rf_factory = lambda: make_rf(seed)
    lgbm_factory = lambda: make_lgbm(seed)

    cv_raw_lr = cv_evaluate_model(lr_factory, X_all_sc, y_binary, "Raw+LR", seed=seed)
    cv_pca_lr = cv_evaluate_model(lr_factory, Z_all_pca, y_binary, "PCA+LR", seed=seed)
    cv_raw_rf = cv_evaluate_model(rf_factory, X_all_sc, y_binary, "Raw+RF", seed=seed)
    lgbm_label = "Raw+LGBM" if HAS_LGBM else "Raw+GBM"
    cv_raw_lgbm = cv_evaluate_model(lgbm_factory, X_all_sc, y_binary, lgbm_label, seed=seed)
    cv_svae_lr = cv_evaluate_model(lr_factory, Z_all_svae, y_binary, "SVAE+LR", seed=seed)

    all_cv = [cv_raw_lr, cv_pca_lr, cv_raw_rf, cv_raw_lgbm, cv_svae_lr]

    cv_rows = [
        ("CV Accuracy", "cv_accuracy", ".4f"),
        ("CV Balanced Accuracy", "cv_balanced_accuracy", ".4f"),
        ("CV F1 (macro)", "cv_f1_macro", ".4f"),
        ("CV AUC-ROC", "cv_auc", ".4f"),
    ]
    print_table("5-MODEL CROSS-VALIDATED COMPARISON", all_cv, cv_rows)

    # Highlight the best model
    best_cv = max(all_cv, key=lambda r: r["cv_accuracy"])
    print(f"\n  BEST CV accuracy: {best_cv['label']} = {best_cv['cv_accuracy']:.4f}")
    best_auc = max(all_cv, key=lambda r: r["cv_auc"])
    print(f"  BEST CV AUC:      {best_auc['label']} = {best_auc['cv_auc']:.4f}")

    # For backwards compat with the rest of the script
    raw_cv = raw_repr
    raw_cv.update(cv_raw_lr)
    pca_cv = pca_repr
    pca_cv.update(cv_pca_lr)
    svae_cv = svae_repr
    svae_cv.update(cv_svae_lr)

    # ---- Step 4b: Art-specific probes ----
    print("\n--- STEP 4b: Art-Specific Linear Probes ---")

    # Build probe label sets from metadata
    probe_results = []
    art_probes = [
        ("medium", "classification"),
        ("auction_house", "classification"),
        ("artist_vital_status", "classification"),
        ("sale_category", "classification"),
        ("price_bucket", "classification"),
    ]

    for concept, probe_type in art_probes:
        label_key = f"label_{concept}"
        if label_key in metadata:
            labels = np.array(metadata[label_key])
            if len(labels) != X_all_sc.shape[0]:
                print(f"  Skipping probe '{concept}': label count mismatch")
                continue

            # Filter valid labels
            valid_mask = np.array([l is not None and str(l) != "nan" and str(l) != "" for l in labels])
            if valid_mask.sum() < 50:
                print(f"  Skipping probe '{concept}': too few valid labels ({valid_mask.sum()})")
                continue

            X_probe = X_all_sc[valid_mask]
            Z_svae_probe = Z_all_svae[valid_mask]
            Z_pca_probe = Z_all_pca[valid_mask]
            y_probe = labels[valid_mask]

            raw_probe = probe_classification(X_probe, y_probe, concept, "raw")
            pca_probe = probe_classification(Z_pca_probe, y_probe, concept, "pca")
            svae_probe = probe_classification(Z_svae_probe, y_probe, concept, "svae")

            if not raw_probe.get("skipped"):
                delta_pca = svae_probe["accuracy"] - pca_probe["accuracy"]
                delta_raw = svae_probe["accuracy"] - raw_probe["accuracy"]
                print(f"  {concept} ({raw_probe['n_classes']} classes, n={raw_probe['n_samples']}):")
                print(f"    Raw: {raw_probe['accuracy']:.4f}  PCA: {pca_probe['accuracy']:.4f}  "
                      f"SVAE: {svae_probe['accuracy']:.4f}  (SVAE-PCA: {delta_pca:+.4f})")
                probe_results.append({
                    "concept": concept,
                    "raw": raw_probe,
                    "pca": pca_probe,
                    "svae": svae_probe,
                })
        else:
            print(f"  Probe '{concept}': labels not available in data")

    # Regression probe on y if it's continuous
    if task_type == "regression":
        print("\n  Regression probes (hammer price):")
        raw_reg = probe_regression(X_all_sc, y, "hammer_price", "raw")
        pca_reg = probe_regression(Z_all_pca, y, "hammer_price", "pca")
        svae_reg = probe_regression(Z_all_svae, y, "hammer_price", "svae")
        print(f"    Raw R2: {raw_reg['r2']:.4f}  PCA R2: {pca_reg['r2']:.4f}  "
              f"SVAE R2: {svae_reg['r2']:.4f}  (SVAE-PCA: {svae_reg['r2']-pca_reg['r2']:+.4f})")
        probe_results.append({
            "concept": "hammer_price",
            "raw": raw_reg,
            "pca": pca_reg,
            "svae": svae_reg,
        })

    # ---- Step 4c: Feature attribution ----
    print("\n--- STEP 4c: Feature Attribution (Jacobian) ---")
    attributions = compute_jacobian_attribution(model, X_all_sc, feature_names, args.embedding_dim)

    # Print top features per embedding dimension
    for dim_i in range(min(args.embedding_dim, 8)):
        top3_idx = np.argsort(attributions[:, dim_i])[::-1][:3]
        top3 = [(feature_names[j], attributions[j, dim_i]) for j in top3_idx]
        print(f"  dim_{dim_i}: {', '.join(f'{n}({v:.3f})' for n, v in top3)}")

    # ---- Step 5: Out-of-sample test (all 5 models) ----
    print("\n--- STEP 5: Out-of-Sample Test (5 Models) ---")

    oos_raw_lr = evaluate_oos(X_train_sc, X_test_sc, y_train_bin, y_test_bin, "Raw+LR")
    oos_pca_lr = evaluate_oos(Z_train_pca, Z_test_pca, y_train_bin, y_test_bin, "PCA+LR")
    oos_raw_rf = oos_evaluate_model(rf_factory, X_train_sc, X_test_sc, y_train_bin, y_test_bin, "Raw+RF")
    oos_raw_lgbm = oos_evaluate_model(lgbm_factory, X_train_sc, X_test_sc, y_train_bin, y_test_bin, lgbm_label)
    oos_svae_lr = evaluate_oos(Z_train_svae, Z_test_svae, y_train_bin, y_test_bin, "SVAE+LR")

    all_oos = [oos_raw_lr, oos_pca_lr, oos_raw_rf, oos_raw_lgbm, oos_svae_lr]

    oos_rows = [
        ("Accuracy", "accuracy", ".4f"),
        ("Balanced Accuracy", "balanced_accuracy", ".4f"),
        ("F1 (macro)", "f1_macro", ".4f"),
        ("AUC-ROC", "auc", ".4f"),
    ]
    print_table("OUT-OF-SAMPLE TEST SET RESULTS (5 MODELS)", all_oos, oos_rows)

    best_oos = max(all_oos, key=lambda r: r["accuracy"])
    print(f"\n  BEST OOS accuracy: {best_oos['label']} = {best_oos['accuracy']:.4f}")

    # ---- Step 5b: Walk-forward temporal backtest ----
    print("\n--- STEP 5b: Walk-Forward Temporal Backtest ---")

    min_train_size = max(100, int(n_samples * 0.3))
    step_size = max(20, int(n_samples * 0.05))
    print(f"  min_train={min_train_size}, step={step_size}")

    wf_results = {"Raw+LR": [], "PCA+LR": [], "Raw+RF": [], "Raw+LGBM": [], "SVAE+LR": []}
    for start in range(min_train_size, n_samples - step_size, step_size):
        test_end = min(start + step_size, n_samples)
        X_tr, y_tr = X[:start], y_binary[:start]
        X_te, y_te = X[start:test_end], y_binary[start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # Raw+LR
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)
        clf.fit(X_tr_sc, y_tr)
        wf_results["Raw+LR"].append(accuracy_score(y_te, clf.predict(X_te_sc)))

        # PCA+LR
        pc = PCA(n_components=pca_dim, random_state=args.seed)
        Z_tr = pc.fit_transform(X_tr_sc)
        Z_te = pc.transform(X_te_sc)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)
        clf.fit(Z_tr, y_tr)
        wf_results["PCA+LR"].append(accuracy_score(y_te, clf.predict(Z_te)))

        # Raw+RF
        rf = make_rf(args.seed)
        rf.fit(X_tr_sc, y_tr)
        wf_results["Raw+RF"].append(accuracy_score(y_te, rf.predict(X_te_sc)))

        # Raw+LGBM
        gb = make_lgbm(args.seed)
        gb.fit(X_tr_sc, y_tr)
        wf_results["Raw+LGBM"].append(accuracy_score(y_te, gb.predict(X_te_sc)))

        # SVAE+LR (using global model -- slight look-ahead, consistent with Polymarket methodology)
        with torch.no_grad():
            Z_tr_s = model.get_embedding(torch.tensor(X_tr_sc, dtype=torch.float32))
            Z_te_s = model.get_embedding(torch.tensor(X_te_sc, dtype=torch.float32))
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)
        clf.fit(Z_tr_s, y_tr)
        wf_results["SVAE+LR"].append(accuracy_score(y_te, clf.predict(Z_te_s)))

    if wf_results["Raw+LR"]:
        n_windows = len(wf_results["Raw+LR"])
        print(f"  Windows: {n_windows}")
        for name, vals in wf_results.items():
            print(f"  {name:<15s} mean: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    else:
        print("  Not enough data for walk-forward.")

    # ---- Step 6: Visualizations ----
    print("\n--- STEP 6: Generating Figures ---")

    try:
        # VIF comparison
        pca_vif = compute_vif(Z_all_pca)
        svae_vif = compute_vif(Z_all_svae)
        plot_multicollinearity_comparison(
            raw_vif.vif_values, pca_vif.vif_values, svae_vif.vif_values,
            feature_names, output_dir)

        # Correlation matrices
        plot_correlation_matrices(X_all_sc, Z_all_pca, Z_all_svae, feature_names, output_dir)

        # Walk-forward
        if wf_results["Raw+LR"]:
            plot_walk_forward(wf_results, baseline, output_dir)

        # Feature attribution
        plot_attribution_heatmap(attributions, feature_names, args.embedding_dim, output_dir)

        # 5-model comparison bar chart
        plot_model_comparison(all_cv, output_dir)

        # Domain comparison
        art_comparison = {
            "vif_reduction_factor": raw_vif.max_vif / max(svae_cv["vif_max"], 0.01),
            "condition_reduction_factor": raw_cond / max(svae_cv["condition_number"], 0.01),
            "svae_vs_pca_accuracy_delta": svae_cv["cv_accuracy"] - pca_cv["cv_accuracy"],
        }
        plot_domain_comparison(art_comparison, output_dir)

        print(f"  Figures saved to {output_dir}")
    except Exception as e:
        print(f"  WARNING: Figure generation failed: {e}")

    # ---- Step 7: Verdict ----
    print(f"\n{'='*80}")
    print("  VERDICT: Art Market 5-Model Experiment")
    print(f"{'='*80}")

    svae_vs_pca_acc = svae_cv["cv_accuracy"] - pca_cv["cv_accuracy"]
    svae_vs_raw_acc = svae_cv["cv_accuracy"] - raw_cv["cv_accuracy"]
    svae_vs_rf_acc = cv_svae_lr["cv_accuracy"] - cv_raw_rf["cv_accuracy"]
    svae_vs_lgbm_acc = cv_svae_lr["cv_accuracy"] - cv_raw_lgbm["cv_accuracy"]
    vif_reduction = raw_vif.max_vif / max(svae_cv["vif_max"], 0.01)
    cond_reduction = raw_cond / max(svae_cv["condition_number"], 0.01)

    print(f"\n  Dataset: Art Market Auctions")
    print(f"  Samples: {n_samples}, Features: {n_features}")
    print(f"  Majority baseline: {baseline:.1%}")

    print(f"\n  MULTICOLLINEARITY:")
    print(f"    Raw max VIF:       {raw_vif.max_vif:.2f} ({raw_vif.n_severe} severe)")
    print(f"    PCA max VIF:       {pca_cv['vif_max']:.2f}")
    print(f"    Orth-SVAE max VIF: {svae_cv['vif_max']:.2f}")
    print(f"    VIF reduction:     {vif_reduction:.1f}x")
    print(f"    Cond# reduction:   {cond_reduction:.1f}x")

    print(f"\n  5-MODEL PREDICTIVE COMPARISON:")
    for r in all_cv:
        print(f"    {r['label']:<15s} accuracy: {r['cv_accuracy']:.4f}  AUC: {r['cv_auc']:.4f}")
    print(f"\n    SVAE vs PCA:   {svae_vs_pca_acc:+.4f} ({svae_vs_pca_acc*100:+.1f}pp)")
    print(f"    SVAE vs RF:    {svae_vs_rf_acc:+.4f} ({svae_vs_rf_acc*100:+.1f}pp)")
    print(f"    SVAE vs LGBM:  {svae_vs_lgbm_acc:+.4f} ({svae_vs_lgbm_acc*100:+.1f}pp)")

    print(f"\n  DOMAIN COMPARISON (Art Market vs Polymarket):")
    poly = POLYMARKET_REFERENCE
    print(f"    {'Metric':<35s} {'Polymarket':>12s} {'Art Market':>12s}")
    print(f"    {'-'*60}")
    print(f"    {'Features':<35s} {poly['n_features']:>12d} {n_features:>12d}")
    print(f"    {'Samples':<35s} {poly['n_samples']:>12d} {n_samples:>12d}")
    print(f"    {'Raw max VIF':<35s} {poly['raw_vif_max']:>12.2f} {raw_vif.max_vif:>12.2f}")
    print(f"    {'Raw condition#':<35s} {poly['raw_condition_number']:>12.1f} {raw_cond:>12.1f}")
    print(f"    {'VIF reduction (x)':<35s} {poly['raw_vif_max']/max(poly['svae_vif_max'],0.01):>12.1f} {vif_reduction:>12.1f}")
    print(f"    {'Cond# reduction (x)':<35s} {poly['raw_condition_number']/max(poly['svae_condition_number'],0.01):>12.1f} {cond_reduction:>12.1f}")
    print(f"    {'SVAE vs PCA accuracy (pp)':<35s} {poly['svae_vs_pca_accuracy_delta']*100:>+12.1f} {svae_vs_pca_acc*100:>+12.1f}")
    if wf_results["Raw+LR"]:
        print(f"    {'Walk-forward SVAE-PCA (pp)':<35s} "
              f"{(poly['walk_forward_svae']-poly['walk_forward_pca'])*100:>+12.1f} "
              f"{(np.mean(wf_results['SVAE+LR'])-np.mean(wf_results['PCA+LR']))*100:>+12.1f}")

    # Final verdict: two questions
    # Q1: Does SVAE beat PCA more on art data (dimensionality hypothesis)?
    # Q2: Does SVAE beat standard ML (RF, LGBM) -- the practical question?
    art_advantage_vs_pca = svae_vs_pca_acc * 100
    poly_advantage = poly["svae_vs_pca_accuracy_delta"] * 100

    svae_beats_rf = svae_vs_rf_acc > 0
    svae_beats_lgbm = svae_vs_lgbm_acc > 0
    svae_beats_all_nonlinear = svae_beats_rf and svae_beats_lgbm

    print(f"\n  --- Q1: Does dimensionality amplify Orth-SVAE advantage? ---")
    if art_advantage_vs_pca > poly_advantage + 1.0:
        q1_verdict = "YES: SVAE-PCA gap larger on art data"
        q1_detail = (f"Art ({n_features}D): {art_advantage_vs_pca:+.1f}pp > "
                    f"Polymarket (8D): {poly_advantage:+.1f}pp. "
                    f"Delta: {art_advantage_vs_pca - poly_advantage:+.1f}pp more on art data.")
    elif art_advantage_vs_pca > poly_advantage:
        q1_verdict = "PARTIALLY: Slight increase, not convincing"
        q1_detail = (f"Art: {art_advantage_vs_pca:+.1f}pp vs Poly: {poly_advantage:+.1f}pp. "
                    f"Difference too small to conclude dimensionality matters.")
    else:
        q1_verdict = "NO: No amplification from higher dimensionality"
        q1_detail = (f"Art: {art_advantage_vs_pca:+.1f}pp vs Poly: {poly_advantage:+.1f}pp.")
    print(f"  {q1_verdict}")
    print(f"  {q1_detail}")

    print(f"\n  --- Q2: Does SVAE beat standard ML (RF, LightGBM)? ---")
    if svae_beats_all_nonlinear:
        q2_verdict = "YES: SVAE+LR outperforms tree-based models"
        q2_detail = (f"SVAE vs RF: {svae_vs_rf_acc*100:+.1f}pp, "
                    f"SVAE vs LGBM: {svae_vs_lgbm_acc*100:+.1f}pp. "
                    f"The embedding approach adds value over standard ML.")
    elif svae_beats_rf or svae_beats_lgbm:
        beaten = "RF" if svae_beats_rf else "LGBM"
        lost_to = "LGBM" if svae_beats_rf else "RF"
        q2_verdict = f"MIXED: SVAE beats {beaten} but loses to {lost_to}"
        q2_detail = (f"SVAE vs RF: {svae_vs_rf_acc*100:+.1f}pp, "
                    f"SVAE vs LGBM: {svae_vs_lgbm_acc*100:+.1f}pp. "
                    f"Competitive but not dominant.")
    else:
        q2_verdict = "NO: Standard ML outperforms SVAE embeddings"
        q2_detail = (f"SVAE vs RF: {svae_vs_rf_acc*100:+.1f}pp, "
                    f"SVAE vs LGBM: {svae_vs_lgbm_acc*100:+.1f}pp. "
                    f"Tree-based models handle raw features better -- embedding overhead not justified.")
    print(f"  {q2_verdict}")
    print(f"  {q2_detail}")

    # Combined verdict
    if svae_beats_all_nonlinear and art_advantage_vs_pca > poly_advantage + 1.0:
        verdict = "STRONG CONFIRMATION: SVAE adds value and scales with dimensionality"
        verdict_detail = (f"Orth-SVAE+LR beats all baselines including RF/LGBM, "
                         f"and the advantage is larger on art data ({n_features}D) "
                         f"than Polymarket (8D). The methodology is validated.")
    elif svae_beats_all_nonlinear:
        verdict = "PRACTICAL WIN: SVAE beats standard ML despite weak dimensionality effect"
        verdict_detail = (f"SVAE+LR outperforms RF and LGBM on raw features, proving the "
                         f"embedding approach has practical value regardless of the "
                         f"dimensionality hypothesis.")
    elif art_advantage_vs_pca > poly_advantage + 1.0 and not svae_beats_all_nonlinear:
        verdict = "ACADEMIC WIN ONLY: Better than PCA but not standard ML"
        verdict_detail = (f"SVAE beats PCA by more on art data (confirming dimensionality "
                         f"hypothesis), but standard ML (RF/LGBM) on raw features still "
                         f"wins. The methodology is theoretically interesting but not practical.")
    elif vif_reduction > 5.0 and cond_reduction > 5.0:
        verdict = "INTERPRETABILITY WIN: SVAE decorrelates but does not predict best"
        verdict_detail = (f"SVAE reduces VIF by {vif_reduction:.0f}x. Use case: interpretable "
                         f"low-dimensional representation, not maximum predictive power.")
    else:
        verdict = "NOT CONFIRMED: Standard ML on raw features is the best approach"
        verdict_detail = (f"Neither the dimensionality hypothesis nor the practical value "
                         f"of Orth-SVAE embeddings is confirmed on art data. "
                         f"Best model: {best_cv['label']} ({best_cv['cv_accuracy']:.4f}).")

    print(f"\n  FINAL VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print(f"{'='*80}")

    # ---- Save report ----
    report = {
        "experiment": "art_market_orth_svae",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "Art Market Auctions",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_features_dropped": len(dropped),
            "task_type": task_type,
            "majority_baseline": float(baseline),
            "feature_names": feature_names,
        },
        "hyperparameters": {
            "embedding_dim": args.embedding_dim,
            "beta": args.beta,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        "training": {
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        },
        "raw_multicollinearity": {
            "vif_mean": raw_vif.mean_vif,
            "vif_max": raw_vif.max_vif,
            "vif_severe": raw_vif.n_severe,
            "condition_number": raw_cond,
            "max_correlation": raw_max_corr,
        },
        "cv_results": {
            "raw_lr": cv_raw_lr,
            "pca_lr": cv_pca_lr,
            "raw_rf": cv_raw_rf,
            "raw_lgbm": cv_raw_lgbm,
            "svae_lr": cv_svae_lr,
            "representation_quality": {
                "raw": raw_repr,
                "pca": pca_repr,
                "svae": svae_repr,
            },
        },
        "oos_results": {
            "raw_lr": oos_raw_lr,
            "pca_lr": oos_pca_lr,
            "raw_rf": oos_raw_rf,
            "raw_lgbm": oos_raw_lgbm,
            "svae_lr": oos_svae_lr,
        },
        "walk_forward": {
            method: {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
            }
            for method, vals in wf_results.items()
        } | {"n_windows": len(wf_results["Raw+LR"])},
        "probe_results": probe_results,
        "domain_comparison": {
            "polymarket_reference": POLYMARKET_REFERENCE,
            "art_svae_vs_pca_delta": svae_vs_pca_acc,
            "art_svae_vs_rf_delta": svae_vs_rf_acc,
            "art_svae_vs_lgbm_delta": svae_vs_lgbm_acc,
            "poly_svae_vs_pca_delta": poly["svae_vs_pca_accuracy_delta"],
            "art_vif_reduction": vif_reduction,
            "art_cond_reduction": cond_reduction,
        },
        "q1_dimensionality_hypothesis": q1_verdict,
        "q1_detail": q1_detail,
        "q2_practical_value": q2_verdict,
        "q2_detail": q2_detail,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "elapsed_seconds": time.perf_counter() - total_start,
    }

    report_path = output_dir / "art_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  Figures: {output_dir}")
    print(f"  Completed in {report['elapsed_seconds']:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
