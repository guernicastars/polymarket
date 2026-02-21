#!/usr/bin/env python3
"""Art Market Regression Experiment: Predicting log hammer price.

Compares 6 approaches on art auction data (EXCLUDING estimate features):
  1. Raw + Ridge              (linear baseline)
  2. PCA(d) + Ridge           (decorrelation baseline)
  3. Raw + Random Forest      (nonlinear baseline)
  4. Raw + LightGBM           (gradient boosting baseline)
  5. Orth-SVAE(d) + Ridge     (existing model in regression mode)
  6. T-OSVAE(d) + Ridge       (transformer model -- conditional import)

Key design decisions:
  - Target: log_hammer_price (continuous regression)
  - Estimate features EXCLUDED: categories D and P plus any other
    estimate-derived features (hammer_start_ratio, final_hammer_ratio,
    num_bids, reserve_met) to prevent target leakage
  - Temporal split: 70/15/15 by sale date
  - StandardScaler fit on train only
  - Walk-forward validation with expanding window (10+ windows)

Metrics: MAE, RMSE, R2, MAPE, Median AE, within-10% accuracy, within-25% accuracy

Usage:
    python run_art_regression.py
    python run_art_regression.py --data-dir art_data/output
    python run_art_regression.py --embedding-dim 12 --gamma 2.0
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
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from models.autoencoder import AutoencoderConfig, MarketAutoencoder
from models.statistics import (
    compute_condition_number,
    compute_vif,
)
from models.train import TrainConfig, train as train_model

# Conditional import for T-OSVAE (Task #9 may not be complete)
HAS_TOSVAE = False
try:
    from models.transformer_autoencoder import (
        TransformerAutoencoderConfig,
        TransformerAutoencoder,
    )
    HAS_TOSVAE = True
except ImportError:
    pass

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


# ======================================================================
# Estimate features to exclude
# ======================================================================

# Category D: Estimate / Valuation
ESTIMATE_FEATURES_D = {
    "log_estimate_low",
    "log_estimate_high",
    "log_estimate_mid",
    "estimate_spread",
}

# Category P: Estimate Accuracy
ESTIMATE_FEATURES_P = {
    "estimate_mid_usd",
    "log_estimate_range_usd",
    "estimate_relative_level",
}

# Other estimate-derived or target-leaking features:
#   hammer_start_ratio = hammer_price / starting_bid (uses target directly)
#   final_hammer_ratio = final_price / hammer_price (uses target directly)
#   num_bids = number of bids (strongly correlated with hammer price outcome)
#   reserve_met = whether reserve was met (post-auction outcome)
OTHER_ESTIMATE_DERIVED = {
    "hammer_start_ratio",
    "final_hammer_ratio",
    "num_bids",
    "reserve_met",
}

ALL_EXCLUDED_FEATURES = ESTIMATE_FEATURES_D | ESTIMATE_FEATURES_P | OTHER_ESTIMATE_DERIVED

# Feature group definitions (categories from art_data/features.py)
# Used by T-OSVAE for group tokenization. Excludes D/P and target-leaking groups.
FEATURE_GROUP_DEFINITIONS: dict[str, list[str]] = {
    "A_artist": [
        "is_living", "artist_birth_year", "artist_career_length",
        "is_rare_artist", "artist_market_depth", "nationality_known",
    ],
    "B_physical": [
        "height_cm", "width_cm", "log_surface_area",
        "aspect_ratio", "has_depth", "creation_year",
        "creation_is_approximate",
    ],
    "C_medium": [
        "is_painting", "is_sculpture", "is_work_on_paper",
        "is_decorative", "medium_known",
    ],
    # D_estimate: EXCLUDED
    "E_sale_context": [
        "sale_month", "sale_year_numeric", "sale_day_of_week",
        "sale_size", "lot_position_pct",
    ],
    "F_historical": [
        "artist_avg_log_price", "artist_median_log_price",
        "artist_price_std", "artist_prior_lots", "artist_price_trend",
    ],
    "G_derived": [
        "log_depth_cm", "age_at_creation", "years_since_creation",
    ],
    "H_confidence": [
        "artist_name_confidence", "dimensions_confidence",
        "medium_confidence", "creation_confidence",
    ],
    "I_sale_mechanics": [
        # num_bids, reserve_met, hammer_start_ratio excluded (target leakage)
        "log_starting_bid", "is_online_sale",
    ],
    "J_attribution": [
        "is_attributed_artist", "is_maker_not_artist",
    ],
    "K_provenance": [
        "provenance_count", "literature_count", "exhibition_count",
    ],
    "L_text": [
        "title_length", "has_description", "has_signed_inscribed",
        "has_condition_report",
    ],
    "M_style": [
        "has_style_period", "has_origin",
    ],
    "N_lot_category": [
        "is_wine", "is_jewelry", "is_book", "is_asian_art",
    ],
    "O_sale_flags": [
        # final_hammer_ratio excluded (target leakage)
        "has_crypto", "is_guaranteed",
    ],
    # P_estimate_accuracy: EXCLUDED
}


def build_feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Build feature group index mapping for T-OSVAE tokenizer.

    Maps group names to lists of integer indices into the feature_names array.
    Only includes features that exist in the actual feature set (after filtering
    and zero-variance removal).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    groups: dict[str, list[int]] = {}
    assigned = set()

    for group_name, group_features in FEATURE_GROUP_DEFINITIONS.items():
        indices = []
        for feat in group_features:
            if feat in name_to_idx:
                indices.append(name_to_idx[feat])
                assigned.add(feat)
        if indices:
            groups[group_name] = indices

    # Catch-all: features not in any defined group
    unassigned = [name_to_idx[f] for f in feature_names if f not in assigned]
    if unassigned:
        groups["Z_other"] = unassigned

    return groups


# ======================================================================
# Data loading
# ======================================================================

def load_art_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load art market feature matrix from .npz file.

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


def filter_estimate_features(
    X: np.ndarray, feature_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove estimate-related features from the feature matrix.

    Returns (X_filtered, kept_names, excluded_names).
    """
    excluded = []
    keep_idx = []
    for i, name in enumerate(feature_names):
        if name in ALL_EXCLUDED_FEATURES:
            excluded.append(name)
        else:
            keep_idx.append(i)

    X_filtered = X[:, keep_idx]
    kept_names = [feature_names[i] for i in keep_idx]
    return X_filtered, kept_names, excluded


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
# Regression metrics
# ======================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive regression metrics.

    All computed in log-price space (the native target).
    MAPE and within-X% accuracy computed in original price space.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))

    # Convert to original price space for MAPE and within-X% metrics
    price_true = np.exp(y_true)
    price_pred = np.exp(y_pred)

    # MAPE: mean absolute percentage error (in original price space)
    nonzero = price_true > 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(price_true[nonzero] - price_pred[nonzero]) / price_true[nonzero]))
    else:
        mape = float("inf")

    # Within-X% accuracy: fraction of predictions within X% of true price
    if nonzero.sum() > 0:
        pct_error = np.abs(price_true[nonzero] - price_pred[nonzero]) / price_true[nonzero]
        within_10 = float(np.mean(pct_error <= 0.10))
        within_25 = float(np.mean(pct_error <= 0.25))
    else:
        within_10 = 0.0
        within_25 = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ae": median_ae,
        "within_10pct": within_10,
        "within_25pct": within_25,
    }


# ======================================================================
# Model builders
# ======================================================================

def make_ridge(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def make_rf_regressor(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=seed,
    )


def make_lgbm_regressor(seed: int = 42):
    if HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=-1, num_leaves=31,
            learning_rate=0.1, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=seed,
        )
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.8, random_state=seed,
    )


# ======================================================================
# Cross-validated evaluation
# ======================================================================

def cv_evaluate_regression(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validated regression evaluation."""
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for train_idx, test_idx in cv.split(X):
        reg = model_factory()
        reg.fit(X[train_idx], y[train_idx])
        y_pred = reg.predict(X[test_idx])
        fold_metrics.append(compute_regression_metrics(y[test_idx], y_pred))

    # Average across folds
    avg = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))

    avg["label"] = label
    return avg


# ======================================================================
# VIF analysis for embedding models
# ======================================================================

def vif_analysis(Z: np.ndarray, label: str) -> dict:
    """Compute VIF and condition number for an embedding space."""
    dim_names = [f"dim_{i}" for i in range(Z.shape[1])]
    vif = compute_vif(Z, dim_names)
    cond = compute_condition_number(Z)
    return {
        "label": label,
        "dims": Z.shape[1],
        "vif_mean": vif.mean_vif,
        "vif_max": vif.max_vif,
        "vif_severe": vif.n_severe,
        "vif_moderate": vif.n_moderate,
        "condition_number": cond,
        "vif_values": vif.vif_values.tolist(),
    }


# ======================================================================
# Printing
# ======================================================================

def print_comparison_table(title: str, results: list[dict], metric_rows: list[tuple]) -> None:
    """Print a formatted comparison table."""
    labels = [r["label"] for r in results]
    col_width = max(18, max(len(l) for l in labels) + 2)

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    header = f"  {'Metric':<25s}"
    for lbl in labels:
        header += f" {lbl:>{col_width}s}"
    print(header)
    print("  " + "-" * (25 + (col_width + 1) * len(labels)))

    for name, key, fmt in metric_rows:
        line = f"  {name:<25s}"
        for r in results:
            val = r.get(key, "N/A")
            if val == "N/A":
                line += f" {'N/A':>{col_width}s}"
            elif fmt == "d":
                line += f" {int(val):>{col_width}d}"
            elif fmt == ".0%":
                line += f" {val:>{col_width - 1}.0%} "
            else:
                line += f" {val:>{col_width}{fmt}}"
        print(line)

    print("  " + "-" * (25 + (col_width + 1) * len(labels)))


# ======================================================================
# Visualization
# ======================================================================

def plot_model_comparison_bars(all_cv: list[dict], output_dir: Path) -> None:
    """Grouped bar chart comparing all models on regression metrics."""
    labels = [r["label"] for r in all_cv]
    metrics = ["r2", "mae", "rmse", "within_25pct"]
    metric_names = ["R-squared", "MAE (log)", "RMSE (log)", "Within 25%"]
    colors = ["#5c6bc0", "#42a5f5", "#ff9800", "#ef5350", "#26a69a", "#ab47bc"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        vals = [r.get(metric, 0.0) for r in all_cv]
        bars = ax.barh(range(len(labels)), vals,
                       color=colors[:len(labels)], alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(metric_name)
        ax.set_title(metric_name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2.,
                    f"{val:.3f}", ha="left", va="center", fontsize=8)
        ax.invert_yaxis()

    fig.suptitle("Art Market Regression: Model Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_model_comparison.png", dpi=300)
    plt.close(fig)


def plot_vif_comparison(vif_results: list[dict], output_dir: Path) -> None:
    """Bar chart comparing VIF across representation methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["label"] for r in vif_results]
    max_vifs = [r["vif_max"] for r in vif_results]
    mean_vifs = [r["vif_mean"] for r in vif_results]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, max_vifs, width, label="Max VIF", color="#ef5350", alpha=0.85)
    ax.bar(x + width / 2, mean_vifs, width, label="Mean VIF", color="#42a5f5", alpha=0.85)
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Severe (VIF>10)")
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="Moderate (VIF>5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_ylabel("VIF")
    ax.set_title("VIF Comparison: Regression Representations")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_vif_comparison.png", dpi=300)
    plt.close(fig)


def plot_walk_forward(wf_results: dict[str, list[float]], output_dir: Path) -> None:
    """Walk-forward R-squared plot for all methods."""
    colors = {
        "Raw+Ridge": "#5c6bc0", "PCA+Ridge": "#42a5f5", "Raw+RF": "#ff9800",
        "Raw+LGBM": "#ef5350", "SVAE+Ridge": "#26a69a", "T-OSVAE+Ridge": "#ab47bc",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, vals in wf_results.items():
        if vals:
            x = range(len(vals))
            color = colors.get(name, "#666666")
            ax.plot(x, vals, label=f"{name} (mean={np.mean(vals):.3f})",
                    alpha=0.8, linewidth=1.5, color=color)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Zero R2")
    ax.set_xlabel("Window")
    ax.set_ylabel("R-squared")
    ax.set_title("Art Market Regression: Walk-Forward Temporal Backtest")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_walk_forward.png", dpi=300)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, predictions: dict[str, np.ndarray], output_dir: Path) -> None:
    """Residual plots for each model."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (label, y_pred) in zip(axes, predictions.items()):
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.1, s=5, color="#5c6bc0")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Predicted (log price)")
        ax.set_ylabel("Residual")
        ax.set_title(f"{label}")

    fig.suptitle("Residual Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_residuals.png", dpi=300)
    plt.close(fig)


# ======================================================================
# T-OSVAE training helper
# ======================================================================

def train_tosvae(
    X_all_sc: np.ndarray,
    y: np.ndarray,
    n_features: int,
    embedding_dim: int,
    feature_groups: dict[str, list[int]],
    output_dir: Path,
    epochs: int = 300,
    batch_size: int = 256,
    patience: int = 30,
    seed: int = 42,
    gamma: float = 0.1,
    beta: float = 1.0,
    alpha: float = 1.0,
    max_train_samples: int = 50000,
) -> tuple[object, np.ndarray] | None:
    """Train T-OSVAE model if available. Returns (model, embeddings) or None.

    Uses task="regression" with MSE prediction loss on the continuous target.
    Feature groups are passed to the GroupFeatureTokenizer for structured attention.

    For large datasets, subsamples to max_train_samples for training (transformer
    attention is O(seq_len^2) per batch, making full-dataset training on CPU
    infeasible for 100K+ samples). The trained encoder is then applied to the
    full dataset for embedding extraction.
    """
    if not HAS_TOSVAE:
        print("  T-OSVAE not available (models/transformer_autoencoder.py not found)")
        return None

    if not feature_groups:
        print("  T-OSVAE skipped: no feature groups defined")
        return None

    try:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = TransformerAutoencoderConfig(
            input_dim=n_features,
            embedding_dim=embedding_dim,
            feature_groups=feature_groups,
            task="regression",
            beta=beta,
            alpha=alpha,
            gamma=gamma,
        )
        model = TransformerAutoencoder(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  T-OSVAE params: {n_params:,}, groups: {len(feature_groups)}, "
              f"tokens: {1 + len(feature_groups)} (CLS + groups)")
        for gname, gidx in sorted(feature_groups.items()):
            print(f"    {gname}: {len(gidx)} features")

        # Subsample for training if dataset is large (transformer is slow on CPU)
        n_total = X_all_sc.shape[0]
        if n_total > max_train_samples:
            rng = np.random.RandomState(seed)
            subsample_idx = rng.choice(n_total, max_train_samples, replace=False)
            X_sub = X_all_sc[subsample_idx]
            y_sub = y[subsample_idx]
            print(f"  Subsampled {max_train_samples:,} / {n_total:,} for T-OSVAE training")
        else:
            X_sub = X_all_sc
            y_sub = y

        X_tensor = torch.tensor(X_sub, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_sub, dtype=torch.float32).to(device)

        # Train/val split (80/20)
        n = X_tensor.shape[0]
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        from torch.utils.data import DataLoader, TensorDataset
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        best_epoch = 0
        wait = 0
        ckpt_path = output_dir / "best_model_tosvae.pt"

        import sys as _sys
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                _x_hat, _z, losses = model(batch_x, batch_y)
                loss = losses["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    _x_hat, _z, losses = model(batch_x, batch_y)
                    val_loss += losses["total_loss"].item()
            val_loss /= len(val_loader)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    break

            if epoch % 10 == 0 or epoch == 1:
                print(f"    T-OSVAE epoch {epoch}: train={train_loss/n_batches:.6f} val={val_loss:.6f}")
                _sys.stdout.flush()

        print(f"  T-OSVAE best epoch: {best_epoch}, val loss: {best_val_loss:.6f}")

        # Encode the FULL dataset (not just the subsample)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        X_full = torch.tensor(X_all_sc, dtype=torch.float32).to(device)
        # Encode in batches to avoid OOM on large datasets
        Z_parts = []
        encode_bs = 1024
        with torch.no_grad():
            for i in range(0, X_full.shape[0], encode_bs):
                Z_parts.append(model.encode(X_full[i:i+encode_bs]).cpu().numpy())
        Z = np.concatenate(Z_parts, axis=0)

        return model, Z

    except Exception as e:
        import traceback
        print(f"  T-OSVAE training failed: {e}")
        traceback.print_exc()
        return None


# ======================================================================
# Main experiment
# ======================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Art Market Regression Experiment")
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
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization")
    parser.add_argument("--walk-forward-windows", type=int, default=10,
                        help="Minimum walk-forward windows")
    parser.add_argument("--tosvae-max-samples", type=int, default=50000,
                        help="Max training samples for T-OSVAE (subsample for CPU speed)")
    args = parser.parse_args()

    total_start = time.perf_counter()

    print("=" * 90)
    print("  ART MARKET REGRESSION EXPERIMENT: 6-Model Comparison")
    print("  Target: log_hammer_price (continuous)")
    print("  Estimate features EXCLUDED to prevent leakage")
    print("=" * 90)

    output_dir = EXPERIMENT_DIR / "results" / "art_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load and filter data ----
    print("\n--- STEP 1: Load Art Market Data ---")
    X, y, feature_names, metadata = load_art_data(args.data_dir)

    # The target from extract.py with target="log_price" is already log(hammer_price)
    # Verify it looks like log prices (should be roughly in range 3-20)
    y_range = y.max() - y.min()
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target (log_hammer_price): mean={y.mean():.2f}, std={y.std():.2f}, "
          f"range=[{y.min():.2f}, {y.max():.2f}]")

    # Filter out estimate features
    print("\n  Excluding estimate-related features:")
    X, feature_names, excluded = filter_estimate_features(X, feature_names)
    print(f"    Category D (Estimate/Valuation): {sorted(ESTIMATE_FEATURES_D & set(excluded))}")
    print(f"    Category P (Estimate Accuracy):  {sorted(ESTIMATE_FEATURES_P & set(excluded))}")
    print(f"    Other target-leaking:            {sorted(OTHER_ESTIMATE_DERIVED & set(excluded))}")
    print(f"    Total excluded: {len(excluded)} features")
    print(f"    Remaining: {X.shape[1]} features")

    # Preprocess (NaN imputation, zero-variance removal)
    X, feature_names, dropped = preprocess(X, feature_names)
    n_samples, n_features = X.shape
    print(f"  After preprocessing: {n_samples} samples, {n_features} features")

    # ---- Temporal split (70/15/15) ----
    # Data is sorted by sale date from extract.py
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.85)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    print(f"  Temporal split: train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X)

    # ---- Step 2: Raw multicollinearity ----
    print("\n--- STEP 2: Raw Multicollinearity Assessment ---")
    raw_vif_result = compute_vif(X_all_sc, feature_names)
    raw_cond = compute_condition_number(X_all_sc)
    raw_corr = np.corrcoef(X_all_sc.T)
    np.fill_diagonal(raw_corr, 0)
    raw_max_corr = float(np.max(np.abs(raw_corr)))

    print(f"  Dimensions: {n_features}")
    print(f"  Mean VIF: {raw_vif_result.mean_vif:.2f}")
    print(f"  Max VIF: {raw_vif_result.max_vif:.2f}")
    print(f"  Severe VIF (>10): {raw_vif_result.n_severe}/{n_features}")
    print(f"  Condition number: {raw_cond:.1f}")
    print(f"  Max |correlation|: {raw_max_corr:.3f}")

    top_vifs = sorted(zip(raw_vif_result.feature_names, raw_vif_result.vif_values), key=lambda x: -x[1])
    print("  Top VIF features:")
    for name, vif_val in top_vifs[:10]:
        flag = " ***" if vif_val > 10 else ""
        print(f"    {name:<35s} VIF = {vif_val:>8.2f}{flag}")

    # ---- Step 3: Train Orth-SVAE ----
    print(f"\n--- STEP 3: Train Orth-SVAE (dim={args.embedding_dim}, "
          f"beta={args.beta}, alpha={args.alpha}, gamma={args.gamma}) ---")

    # For supervised_vae training, binarize target (above/below median)
    median_y = np.median(y)
    y_binary = (y >= median_y).astype(np.float32)

    data_dir = output_dir / "train_data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "features.npy", X_all_sc.astype(np.float32))
    np.save(data_dir / "labels.npy", y_binary)
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

    # Encode all data
    with torch.no_grad():
        Z_all_svae = model.get_embedding(torch.tensor(X_all_sc, dtype=torch.float32))
        Z_train_svae = model.get_embedding(torch.tensor(X_train_sc, dtype=torch.float32))
        Z_test_svae = model.get_embedding(torch.tensor(X_test_sc, dtype=torch.float32))

    # PCA
    pca_dim = min(args.embedding_dim, n_features)
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    Z_all_pca = pca.fit_transform(X_all_sc)
    Z_train_pca = pca.transform(X_train_sc)
    Z_test_pca = pca.transform(X_test_sc)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%} ({pca_dim} components)")

    # ---- Step 3b: Train T-OSVAE (conditional) ----
    print(f"\n--- STEP 3b: Train T-OSVAE (dim={args.embedding_dim}) ---")
    feature_groups = build_feature_groups(feature_names)
    tosvae_result = train_tosvae(
        X_all_sc, y, n_features, args.embedding_dim,
        feature_groups, ckpt_dir, args.epochs, batch_size=256, patience=args.patience,
        seed=args.seed, gamma=args.gamma, beta=args.beta, alpha=args.alpha,
        max_train_samples=args.tosvae_max_samples,
    )
    Z_all_tosvae = None
    Z_train_tosvae = None
    Z_test_tosvae = None
    if tosvae_result is not None:
        tosvae_model, Z_all_tosvae = tosvae_result
        with torch.no_grad():
            device = next(tosvae_model.parameters()).device
            Z_train_tosvae = tosvae_model.encode(
                torch.tensor(X_train_sc, dtype=torch.float32).to(device)
            ).cpu().numpy()
            Z_test_tosvae = tosvae_model.encode(
                torch.tensor(X_test_sc, dtype=torch.float32).to(device)
            ).cpu().numpy()

    # ---- Step 4: VIF analysis for embedding models ----
    print("\n--- STEP 4: VIF Analysis ---")

    raw_vif = vif_analysis(X_all_sc, f"Raw ({n_features}D)")
    pca_vif = vif_analysis(Z_all_pca, f"PCA ({pca_dim}D)")
    svae_vif = vif_analysis(Z_all_svae, f"SVAE ({args.embedding_dim}D)")
    all_vif = [raw_vif, pca_vif, svae_vif]

    if Z_all_tosvae is not None:
        tosvae_vif = vif_analysis(Z_all_tosvae, f"T-OSVAE ({args.embedding_dim}D)")
        all_vif.append(tosvae_vif)

    vif_rows = [
        ("Dimensions", "dims", "d"),
        ("Mean VIF", "vif_mean", ".2f"),
        ("Max VIF", "vif_max", ".2f"),
        ("Severe VIF (>10)", "vif_severe", "d"),
        ("Moderate VIF (>5)", "vif_moderate", "d"),
        ("Condition Number", "condition_number", ".1f"),
    ]
    print_comparison_table("VIF ANALYSIS", all_vif, vif_rows)

    # ---- Step 5: Cross-validated 6-model comparison ----
    print("\n--- STEP 5: Cross-Validated Regression Comparison ---")

    seed = args.seed
    ridge_alpha = args.ridge_alpha
    ridge_factory = lambda: make_ridge(ridge_alpha)
    rf_factory = lambda: make_rf_regressor(seed)
    lgbm_factory = lambda: make_lgbm_regressor(seed)

    cv_raw_ridge = cv_evaluate_regression(ridge_factory, X_all_sc, y, "Raw+Ridge", seed=seed)
    cv_pca_ridge = cv_evaluate_regression(ridge_factory, Z_all_pca, y, "PCA+Ridge", seed=seed)
    cv_raw_rf = cv_evaluate_regression(rf_factory, X_all_sc, y, "Raw+RF", seed=seed)
    lgbm_label = "Raw+LGBM" if HAS_LGBM else "Raw+GBM"
    cv_raw_lgbm = cv_evaluate_regression(lgbm_factory, X_all_sc, y, lgbm_label, seed=seed)
    cv_svae_ridge = cv_evaluate_regression(ridge_factory, Z_all_svae, y, "SVAE+Ridge", seed=seed)

    all_cv = [cv_raw_ridge, cv_pca_ridge, cv_raw_rf, cv_raw_lgbm, cv_svae_ridge]

    if Z_all_tosvae is not None:
        cv_tosvae_ridge = cv_evaluate_regression(ridge_factory, Z_all_tosvae, y, "T-OSVAE+Ridge", seed=seed)
        all_cv.append(cv_tosvae_ridge)

    cv_rows = [
        ("R-squared", "r2", ".4f"),
        ("MAE (log-price)", "mae", ".4f"),
        ("RMSE (log-price)", "rmse", ".4f"),
        ("MAPE", "mape", ".4f"),
        ("Median AE", "median_ae", ".4f"),
        ("Within 10%", "within_10pct", ".4f"),
        ("Within 25%", "within_25pct", ".4f"),
    ]
    print_comparison_table("CROSS-VALIDATED REGRESSION RESULTS", all_cv, cv_rows)

    best_cv = max(all_cv, key=lambda r: r["r2"])
    print(f"\n  BEST CV R-squared: {best_cv['label']} = {best_cv['r2']:.4f}")
    best_mae = min(all_cv, key=lambda r: r["mae"])
    print(f"  BEST CV MAE:      {best_mae['label']} = {best_mae['mae']:.4f}")

    # ---- Step 6: Out-of-sample test ----
    print("\n--- STEP 6: Out-of-Sample Test ---")

    test_predictions = {}

    def oos_evaluate(X_tr, X_te, y_tr, y_te, factory, label):
        reg = factory()
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        test_predictions[label] = y_pred
        metrics = compute_regression_metrics(y_te, y_pred)
        metrics["label"] = label
        return metrics

    oos_raw_ridge = oos_evaluate(X_train_sc, X_test_sc, y_train, y_test, ridge_factory, "Raw+Ridge")
    oos_pca_ridge = oos_evaluate(Z_train_pca, Z_test_pca, y_train, y_test, ridge_factory, "PCA+Ridge")
    oos_raw_rf = oos_evaluate(X_train_sc, X_test_sc, y_train, y_test, rf_factory, "Raw+RF")
    oos_raw_lgbm = oos_evaluate(X_train_sc, X_test_sc, y_train, y_test, lgbm_factory, lgbm_label)
    oos_svae_ridge = oos_evaluate(Z_train_svae, Z_test_svae, y_train, y_test, ridge_factory, "SVAE+Ridge")

    all_oos = [oos_raw_ridge, oos_pca_ridge, oos_raw_rf, oos_raw_lgbm, oos_svae_ridge]

    if Z_train_tosvae is not None and Z_test_tosvae is not None:
        oos_tosvae = oos_evaluate(Z_train_tosvae, Z_test_tosvae, y_train, y_test,
                                  ridge_factory, "T-OSVAE+Ridge")
        all_oos.append(oos_tosvae)

    oos_rows = [
        ("R-squared", "r2", ".4f"),
        ("MAE (log-price)", "mae", ".4f"),
        ("RMSE (log-price)", "rmse", ".4f"),
        ("MAPE", "mape", ".4f"),
        ("Median AE", "median_ae", ".4f"),
        ("Within 10%", "within_10pct", ".4f"),
        ("Within 25%", "within_25pct", ".4f"),
    ]
    print_comparison_table("OUT-OF-SAMPLE TEST RESULTS", all_oos, oos_rows)

    best_oos = max(all_oos, key=lambda r: r["r2"])
    print(f"\n  BEST OOS R-squared: {best_oos['label']} = {best_oos['r2']:.4f}")

    # ---- Step 7: Walk-forward temporal backtest ----
    print("\n--- STEP 7: Walk-Forward Temporal Backtest ---")

    min_train_size = max(100, int(n_samples * 0.3))
    # Ensure at least args.walk_forward_windows windows
    max_step = (n_samples - min_train_size) // args.walk_forward_windows
    step_size = max(20, min(max_step, int(n_samples * 0.05)))
    print(f"  min_train={min_train_size}, step={step_size}")

    wf_keys = ["Raw+Ridge", "PCA+Ridge", "Raw+RF", "Raw+LGBM", "SVAE+Ridge"]
    if Z_all_tosvae is not None:
        wf_keys.append("T-OSVAE+Ridge")
    wf_results: dict[str, list[float]] = {k: [] for k in wf_keys}

    for start in range(min_train_size, n_samples - step_size, step_size):
        test_end = min(start + step_size, n_samples)
        X_tr, y_tr = X[:start], y[:start]
        X_te, y_te = X[start:test_end], y[start:test_end]

        if len(y_te) < 5:
            continue

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # Raw+Ridge
        reg = make_ridge(ridge_alpha)
        reg.fit(X_tr_sc, y_tr)
        wf_results["Raw+Ridge"].append(float(r2_score(y_te, reg.predict(X_te_sc))))

        # PCA+Ridge
        pc = PCA(n_components=pca_dim, random_state=seed)
        Z_tr = pc.fit_transform(X_tr_sc)
        Z_te = pc.transform(X_te_sc)
        reg = make_ridge(ridge_alpha)
        reg.fit(Z_tr, y_tr)
        wf_results["PCA+Ridge"].append(float(r2_score(y_te, reg.predict(Z_te))))

        # Raw+RF
        rf = make_rf_regressor(seed)
        rf.fit(X_tr_sc, y_tr)
        wf_results["Raw+RF"].append(float(r2_score(y_te, rf.predict(X_te_sc))))

        # Raw+LGBM
        gb = make_lgbm_regressor(seed)
        gb.fit(X_tr_sc, y_tr)
        wf_results["Raw+LGBM"].append(float(r2_score(y_te, gb.predict(X_te_sc))))

        # SVAE+Ridge (using global model -- same methodology as classification experiment)
        with torch.no_grad():
            Z_tr_s = model.get_embedding(torch.tensor(X_tr_sc, dtype=torch.float32))
            Z_te_s = model.get_embedding(torch.tensor(X_te_sc, dtype=torch.float32))
        reg = make_ridge(ridge_alpha)
        reg.fit(Z_tr_s, y_tr)
        wf_results["SVAE+Ridge"].append(float(r2_score(y_te, reg.predict(Z_te_s))))

        # T-OSVAE+Ridge
        if tosvae_result is not None:
            tosvae_model_wf = tosvae_result[0]
            device = next(tosvae_model_wf.parameters()).device
            with torch.no_grad():
                Z_tr_t = tosvae_model_wf.encode(
                    torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
                ).cpu().numpy()
                Z_te_t = tosvae_model_wf.encode(
                    torch.tensor(X_te_sc, dtype=torch.float32).to(device)
                ).cpu().numpy()
            reg = make_ridge(ridge_alpha)
            reg.fit(Z_tr_t, y_tr)
            wf_results["T-OSVAE+Ridge"].append(float(r2_score(y_te, reg.predict(Z_te_t))))

    if wf_results["Raw+Ridge"]:
        n_windows = len(wf_results["Raw+Ridge"])
        print(f"  Windows: {n_windows}")
        for name, vals in wf_results.items():
            if vals:
                print(f"  {name:<18s} mean R2: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    else:
        print("  Not enough data for walk-forward.")

    # ---- Step 8: Visualizations ----
    print("\n--- STEP 8: Generating Figures ---")

    try:
        plot_model_comparison_bars(all_cv, output_dir)
        plot_vif_comparison(all_vif, output_dir)
        if wf_results["Raw+Ridge"]:
            plot_walk_forward(wf_results, output_dir)
        if test_predictions:
            plot_residuals(y_test, test_predictions, output_dir)
        print(f"  Figures saved to {output_dir}")
    except Exception as e:
        print(f"  WARNING: Figure generation failed: {e}")

    # ---- Step 9: Verdict ----
    print(f"\n{'='*90}")
    print("  VERDICT: Art Market Regression Experiment")
    print(f"{'='*90}")

    svae_r2 = cv_svae_ridge["r2"]
    pca_r2 = cv_pca_ridge["r2"]
    raw_ridge_r2 = cv_raw_ridge["r2"]
    rf_r2 = cv_raw_rf["r2"]
    lgbm_r2 = cv_raw_lgbm["r2"]

    svae_vs_pca = svae_r2 - pca_r2
    svae_vs_raw = svae_r2 - raw_ridge_r2
    svae_vs_rf = svae_r2 - rf_r2
    svae_vs_lgbm = svae_r2 - lgbm_r2

    vif_reduction = raw_vif_result.max_vif / max(svae_vif["vif_max"], 0.01)
    cond_reduction = raw_cond / max(svae_vif["condition_number"], 0.01)

    print(f"\n  Dataset: Art Market Auctions (estimate features excluded)")
    print(f"  Samples: {n_samples}, Features: {n_features}")
    print(f"  Target: log_hammer_price, range [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Excluded features ({len(excluded)}): {sorted(excluded)}")

    print(f"\n  MULTICOLLINEARITY (after excluding estimates):")
    print(f"    Raw max VIF:       {raw_vif_result.max_vif:.2f} ({raw_vif_result.n_severe} severe)")
    print(f"    PCA max VIF:       {pca_vif['vif_max']:.2f}")
    print(f"    Orth-SVAE max VIF: {svae_vif['vif_max']:.2f}")
    if Z_all_tosvae is not None:
        print(f"    T-OSVAE max VIF:   {tosvae_vif['vif_max']:.2f}")
    print(f"    VIF reduction (SVAE): {vif_reduction:.1f}x")
    print(f"    Cond# reduction:      {cond_reduction:.1f}x")

    print(f"\n  REGRESSION COMPARISON (CV R-squared):")
    for r in all_cv:
        print(f"    {r['label']:<18s} R2: {r['r2']:.4f}  MAE: {r['mae']:.4f}  Within-25%: {r['within_25pct']:.1%}")
    print(f"\n    SVAE vs PCA:   {svae_vs_pca:+.4f}")
    print(f"    SVAE vs Ridge: {svae_vs_raw:+.4f}")
    print(f"    SVAE vs RF:    {svae_vs_rf:+.4f}")
    print(f"    SVAE vs LGBM:  {svae_vs_lgbm:+.4f}")

    if Z_all_tosvae is not None:
        tosvae_r2 = cv_tosvae_ridge["r2"]
        tosvae_vs_svae = tosvae_r2 - svae_r2
        tosvae_vs_rf = tosvae_r2 - rf_r2
        tosvae_vs_lgbm = tosvae_r2 - lgbm_r2
        print(f"\n    T-OSVAE vs SVAE: {tosvae_vs_svae:+.4f}")
        print(f"    T-OSVAE vs RF:   {tosvae_vs_rf:+.4f}")
        print(f"    T-OSVAE vs LGBM: {tosvae_vs_lgbm:+.4f}")

    # Determine verdict
    svae_beats_rf = svae_vs_rf > 0
    svae_beats_lgbm = svae_vs_lgbm > 0
    svae_beats_pca = svae_vs_pca > 0

    print(f"\n  --- Q1: Does SVAE+Ridge beat standard ML on regression? ---")
    if svae_beats_rf and svae_beats_lgbm:
        q1_verdict = "YES: SVAE+Ridge outperforms RF and LGBM"
        q1_detail = (f"R2 deltas: SVAE vs RF {svae_vs_rf:+.4f}, SVAE vs LGBM {svae_vs_lgbm:+.4f}. "
                     f"Embedding approach adds value for price prediction.")
    elif svae_beats_rf or svae_beats_lgbm:
        beaten = "RF" if svae_beats_rf else "LGBM"
        lost_to = "LGBM" if svae_beats_rf else "RF"
        q1_verdict = f"MIXED: SVAE beats {beaten} but loses to {lost_to}"
        q1_detail = (f"R2 deltas: SVAE vs RF {svae_vs_rf:+.4f}, SVAE vs LGBM {svae_vs_lgbm:+.4f}.")
    else:
        q1_verdict = "NO: Standard ML outperforms SVAE on regression"
        q1_detail = (f"R2 deltas: SVAE vs RF {svae_vs_rf:+.4f}, SVAE vs LGBM {svae_vs_lgbm:+.4f}. "
                     f"Tree-based models handle raw features better for price prediction.")
    print(f"  {q1_verdict}")
    print(f"  {q1_detail}")

    print(f"\n  --- Q2: Does SVAE produce better-conditioned embeddings? ---")
    if vif_reduction > 5.0:
        q2_verdict = f"YES: {vif_reduction:.0f}x VIF reduction, {cond_reduction:.0f}x condition# reduction"
    elif vif_reduction > 2.0:
        q2_verdict = f"MODERATE: {vif_reduction:.1f}x VIF reduction"
    else:
        q2_verdict = f"MINIMAL: only {vif_reduction:.1f}x VIF reduction"
    print(f"  {q2_verdict}")

    if Z_all_tosvae is not None:
        print(f"\n  --- Q3: Does T-OSVAE improve over Orth-SVAE? ---")
        if tosvae_vs_svae > 0.01:
            q3_verdict = f"YES: T-OSVAE R2 {tosvae_r2:.4f} vs SVAE {svae_r2:.4f} ({tosvae_vs_svae:+.4f})"
        elif tosvae_vs_svae > -0.01:
            q3_verdict = f"COMPARABLE: T-OSVAE R2 {tosvae_r2:.4f} vs SVAE {svae_r2:.4f} ({tosvae_vs_svae:+.4f})"
        else:
            q3_verdict = f"NO: SVAE R2 {svae_r2:.4f} > T-OSVAE {tosvae_r2:.4f} ({tosvae_vs_svae:+.4f})"
        print(f"  {q3_verdict}")

    # Final combined verdict
    best_model = max(all_cv, key=lambda r: r["r2"])
    print(f"\n  BEST OVERALL: {best_model['label']} (R2={best_model['r2']:.4f}, "
          f"MAE={best_model['mae']:.4f}, Within-25%={best_model['within_25pct']:.1%})")
    print(f"{'='*90}")

    # ---- Save report ----
    report = {
        "experiment": "art_market_regression",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "Art Market Auctions (estimates excluded)",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_features_dropped_zero_var": len(dropped),
            "target": "log_hammer_price",
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
            "feature_names": feature_names,
            "excluded_features": {
                "category_D_estimate": sorted(ESTIMATE_FEATURES_D & set(excluded)),
                "category_P_estimate_accuracy": sorted(ESTIMATE_FEATURES_P & set(excluded)),
                "other_target_leaking": sorted(OTHER_ESTIMATE_DERIVED & set(excluded)),
                "total_excluded": len(excluded),
                "all_excluded": sorted(excluded),
            },
            "split": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test),
            },
        },
        "hyperparameters": {
            "embedding_dim": args.embedding_dim,
            "beta": args.beta,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "ridge_alpha": ridge_alpha,
            "seed": args.seed,
        },
        "training": {
            "svae_best_epoch": result["best_epoch"],
            "svae_best_val_loss": result["best_val_loss"],
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "tosvae_available": HAS_TOSVAE,
        },
        "multicollinearity": {
            "raw": {
                "vif_mean": raw_vif_result.mean_vif,
                "vif_max": raw_vif_result.max_vif,
                "vif_severe": raw_vif_result.n_severe,
                "condition_number": raw_cond,
                "max_correlation": raw_max_corr,
            },
            "pca": pca_vif,
            "svae": svae_vif,
        },
        "cv_results": {r["label"]: r for r in all_cv},
        "oos_results": {r["label"]: r for r in all_oos},
        "walk_forward": {
            method: {
                "mean_r2": float(np.mean(vals)) if vals else None,
                "std_r2": float(np.std(vals)) if vals else None,
                "n_windows": len(vals),
            }
            for method, vals in wf_results.items()
        },
        "vif_analysis": {r["label"]: r for r in all_vif},
        "verdict": {
            "q1_svae_vs_ml": q1_verdict,
            "q1_detail": q1_detail,
            "q2_conditioning": q2_verdict,
            "best_model": best_model["label"],
            "best_r2": best_model["r2"],
        },
        "elapsed_seconds": time.perf_counter() - total_start,
    }

    if Z_all_tosvae is not None:
        report["multicollinearity"]["tosvae"] = tosvae_vif
        report["verdict"]["q3_tosvae_vs_svae"] = q3_verdict

    report_path = output_dir / "art_regression_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  Figures: {output_dir}")
    print(f"  Completed in {report['elapsed_seconds']:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
