#!/usr/bin/env python3
"""End-to-end experiment runner: embeddings vs raw features for disentanglement.

Orchestrates the full pipeline:
    1. Extract resolved market data from ClickHouse
    2. Summarize features and check raw-space multicollinearity
    3. Train VAE / autoencoder
    4. Extract embeddings for all markets
    5. Compare multicollinearity: raw features vs embeddings (THE KEY TEST)
    6. Run linear probes on both representations
    7. Statistical significance tests
    8. Generate visualizations
    9. Print final summary report with verdict

Usage:
    python run_experiment.py                          # full pipeline
    python run_experiment.py --skip-extract           # use cached data
    python run_experiment.py --skip-train             # use saved model
    python run_experiment.py --skip-extract --skip-train  # analysis only
    python run_experiment.py --config custom.yaml     # custom config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(cfg_path: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    p = Path(cfg_path)
    if not p.is_absolute():
        p = EXPERIMENT_DIR / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager for timing steps."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def print_header(step: int, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Step {step}: {title}")
    print(f"{'='*70}\n")


def print_metric(name: str, value, fmt: str = ".4f") -> None:
    if isinstance(value, float):
        print(f"  {name:<40s} {value:{fmt}}")
    else:
        print(f"  {name:<40s} {value}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_extract(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Step 1: Extract features from ClickHouse.

    Calls data.extract.main() which queries ClickHouse, builds feature vectors,
    performs temporal split, and saves results to output_dir.
    """
    print_header(1, "Data Extraction")

    from data.extract import main as extract_main

    output_dir = resolve_path(config["data"]["output_dir"])

    with Timer("extraction") as t:
        extract_main(
            min_trades=config["data"]["min_trades"],
            output_dir=output_dir,
        )

    # Load the extracted data back
    return _load_extracted_data(output_dir, t.elapsed)


def step_load_cached(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load previously extracted data from disk."""
    print_header(1, "Loading Cached Data")

    output_dir = resolve_path(config["data"]["output_dir"])
    return _load_extracted_data(output_dir)


def _load_extracted_data(
    output_dir: Path,
    extract_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load features.npz and metadata.json from the output directory.

    The extraction pipeline saves split arrays (X_train, X_val, X_test, etc.).
    We reassemble the full X and y for autoencoder training and analysis.
    """
    features_file = output_dir / "features.npz"
    metadata_file = output_dir / "metadata.json"

    if not features_file.exists():
        print(f"  ERROR: {features_file} not found. Run without --skip-extract first.")
        sys.exit(1)

    data = np.load(features_file, allow_pickle=True)

    # Reassemble full arrays from temporal splits
    X_parts = []
    y_parts = []
    for split in ("train", "val", "test"):
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        if x_key in data:
            X_parts.append(data[x_key])
            y_parts.append(data[y_key])

    if X_parts:
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
    else:
        # Fallback: look for unsplit X/y
        X = data["X"]
        y = data["y"]

    # Load metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Also store feature names from npz if available
    if "feature_names" in data:
        fn = data["feature_names"]
        if fn.dtype.kind in ("U", "S", "O"):
            metadata["feature_names"] = fn.tolist()

    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Outcome distribution: Yes={np.sum(y >= 0.5)}, No={np.sum(y < 0.5)}")
    if extract_time is not None:
        print(f"  Extraction time: {extract_time:.1f}s")

    # Handle NaNs: replace with column median
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  NaN values found: {nan_count}. Imputing with column medians.")
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if np.isfinite(median_val) else 0.0

    # Drop zero-variance features (e.g., liquidity=0 for all closed markets,
    # num_outcomes=2 for all binary markets). These cause infinite VIF.
    feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])
    variances = np.var(X, axis=0)
    zero_var_mask = variances < 1e-10
    if zero_var_mask.any():
        dropped = [feature_names[i] for i in range(len(feature_names)) if zero_var_mask[i]]
        print(f"  Dropping {len(dropped)} zero-variance features: {dropped}")
        keep_mask = ~zero_var_mask
        X = X[:, keep_mask]
        feature_names = [fn for fn, keep in zip(feature_names, keep_mask) if keep]
        metadata["feature_names"] = feature_names
        metadata["n_features"] = len(feature_names)
        metadata["dropped_features"] = dropped

    return X, y, metadata


def step_feature_summary(X: np.ndarray, feature_names: list[str]) -> dict:
    """Step 2: Summarize features and check raw-space multicollinearity."""
    print_header(2, "Feature Summary & Raw Multicollinearity")

    from models.statistics import compute_vif, compute_condition_number

    # Basic stats
    print("  Feature statistics:")
    print(f"    Samples:  {X.shape[0]}")
    print(f"    Features: {X.shape[1]}")
    print()

    # Correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr, 0)
    max_corr = float(np.max(np.abs(corr)))
    mean_corr = float(np.mean(np.abs(corr)))

    print("  Correlation analysis (raw features):")
    print_metric("Max |correlation|", max_corr)
    print_metric("Mean |correlation|", mean_corr)

    # Find top correlated pairs
    n_features = X.shape[1]
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append((abs(corr[i, j]), feature_names[i], feature_names[j]))
    pairs.sort(reverse=True)

    print("\n  Top 5 correlated feature pairs:")
    for r, f1, f2 in pairs[:5]:
        print(f"    |r|={r:.3f}  {f1} <-> {f2}")

    # VIF (returns VIFResult dataclass)
    print("\n  Variance Inflation Factors (raw features):")
    vif_result = compute_vif(X, feature_names)
    top_vifs = sorted(
        zip(vif_result.feature_names, vif_result.vif_values),
        key=lambda x: -x[1],
    )
    for name, vif in top_vifs[:10]:
        flag = " *** PROBLEMATIC" if vif > 10 else ""
        print(f"    {name:<35s} VIF = {vif:>10.2f}{flag}")

    print(f"\n    Max VIF: {vif_result.max_vif:.2f}")
    print(f"    Severe (VIF > 10): {vif_result.n_severe}/{len(vif_result.vif_values)}")
    print(f"    Moderate (VIF > 5): {vif_result.n_moderate}/{len(vif_result.vif_values)}")

    # Condition number
    cond = compute_condition_number(X)
    print(f"\n  Condition number: {cond:.2f}")

    raw_stats = {
        "max_correlation": max_corr,
        "mean_correlation": mean_corr,
        "max_vif": vif_result.max_vif,
        "mean_vif": vif_result.mean_vif,
        "problematic_vif_count": vif_result.n_severe,
        "condition_number": cond,
        "vif_values": {
            n: float(v) for n, v in zip(vif_result.feature_names, vif_result.vif_values)
        },
    }
    return raw_stats


def step_train(
    X: np.ndarray, metadata: dict, config: dict,
) -> tuple:
    """Step 3: Train VAE / autoencoder.

    Saves standardized data to disk for models.train, then runs training.
    Returns the trained model and embeddings.
    """
    print_header(3, "Autoencoder Training")

    from sklearn.preprocessing import StandardScaler

    from models.autoencoder import AutoencoderConfig, MarketAutoencoder
    from models.train import TrainConfig, train

    output_dir = resolve_path(config["model"]["checkpoint_dir"])

    # Standardize features and save as features.npy for the training module
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    data_dir = resolve_path(config["data"]["output_dir"])
    np.save(data_dir / "features.npy", X_scaled)

    # Save metadata expected by train.load_data
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Map config model type: config uses "vae"/"autoencoder", code uses "ae"/"vae"
    model_type = config["model"]["type"]
    if model_type == "autoencoder":
        model_type = "ae"

    train_config = TrainConfig(
        model_type=model_type,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=config["model"]["dropout"],
        beta=config["model"]["beta"],
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        patience=config["model"]["patience"],
        data_dir=str(data_dir),
        output_dir=str(output_dir),
    )

    with Timer("training") as t:
        result = train(train_config)

    embeddings = result["embeddings"]
    history = result["history"]
    best_epoch = result["best_epoch"]
    best_val_loss = result["best_val_loss"]

    print(f"  Training complete in {t.elapsed:.1f}s")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Embedding shape: {embeddings.shape}")

    # Reconstruct model for later use
    ae_config = AutoencoderConfig(
        input_dim=X.shape[1],
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=config["model"]["dropout"],
        model_type=model_type,
        beta=config["model"]["beta"],
    )
    model = MarketAutoencoder(ae_config)
    checkpoint_path = output_dir / f"best_model_{model_type}.pt"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    return model, embeddings, history


def step_load_model(X: np.ndarray, config: dict) -> tuple:
    """Load a previously trained model and extract embeddings."""
    print_header(3, "Loading Saved Model")

    from sklearn.preprocessing import StandardScaler

    from models.autoencoder import AutoencoderConfig, MarketAutoencoder

    checkpoint_dir = resolve_path(config["model"]["checkpoint_dir"])

    model_type = config["model"]["type"]
    if model_type == "autoencoder":
        model_type = "ae"

    checkpoint_path = checkpoint_dir / f"best_model_{model_type}.pt"
    if not checkpoint_path.exists():
        print(f"  ERROR: {checkpoint_path} not found. Run without --skip-train first.")
        sys.exit(1)

    ae_config = AutoencoderConfig(
        input_dim=X.shape[1],
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=config["model"]["dropout"],
        model_type=model_type,
        beta=config["model"]["beta"],
    )
    model = MarketAutoencoder(ae_config)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    print(f"  Loaded model from {checkpoint_path}")

    # Also check for pre-saved embeddings
    embeddings_path = checkpoint_dir / "embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"  Loaded cached embeddings: {embeddings.shape}")
    else:
        # Re-extract embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        embeddings = model.get_embedding(X_tensor)
        print(f"  Extracted embeddings: {embeddings.shape}")

    return model, embeddings


def step_embedding_analysis(embeddings: np.ndarray) -> None:
    """Step 4: Analyze embedding quality."""
    print_header(4, "Embedding Analysis")

    from models.train import compute_embedding_stats

    stats = compute_embedding_stats(embeddings)

    print_metric("Mean activation", stats["mean_activation"])
    print_metric("Std activation", stats["std_activation"])
    print_metric("Max inter-dim correlation", stats["max_inter_dim_correlation"])
    print_metric("Mean inter-dim correlation", stats["mean_inter_dim_correlation"])
    print_metric("Dead dimensions (std < 1e-6)", str(stats["dead_dimensions"]))

    active_dims = embeddings.shape[1] - stats["dead_dimensions"]
    print_metric("Active dimensions", f"{active_dims}/{embeddings.shape[1]}")


def step_compare_multicollinearity(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: list[str],
    raw_stats: dict,
    config: dict,
) -> dict:
    """Step 5: Compare multicollinearity -- raw vs embedding. THE KEY TEST."""
    print_header(5, "Multicollinearity Comparison (THE KEY TEST)")

    from models.statistics import compare_multicollinearity

    with Timer("comparison") as t:
        comparison = compare_multicollinearity(X, Z, feature_names)

    # Print the comparison table (the __str__ method is well-formatted)
    print(comparison)
    print(f"\n  Time: {t.elapsed:.1f}s")

    # Extract summary for downstream use
    vif_reduction = raw_stats["max_vif"] / max(comparison.embed_vif.max_vif, 1e-6)
    cond_reduction = raw_stats["condition_number"] / max(comparison.embed_condition, 1e-6)

    print(f"\n  VIF reduction factor:              {vif_reduction:.1f}x")
    print(f"  Condition number reduction factor:  {cond_reduction:.1f}x")

    embed_stats = {
        "max_vif": comparison.embed_vif.max_vif,
        "mean_vif": comparison.embed_vif.mean_vif,
        "problematic_vif_count": comparison.embed_vif.n_severe,
        "condition_number": comparison.embed_condition,
        "max_correlation": comparison.embed_max_corr,
        "vif_reduction_factor": vif_reduction,
        "condition_reduction_factor": cond_reduction,
    }
    return embed_stats


def step_linear_probes(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    metadata: dict,
    config: dict,
) -> list:
    """Step 6: Linear probes -- compare raw vs embedding representations."""
    print_header(6, "Linear Probes")

    from models.probes import run_standard_probes

    # Build labels dict from metadata and y
    labels: dict[str, np.ndarray] = {}

    # Outcome is always available (binary resolution)
    labels["outcome"] = (y >= 0.5).astype(int)

    # Category labels from metadata if available
    category_map = metadata.get("category_map", {})
    if category_map:
        # Categories are encoded as one-hot in the feature matrix;
        # reconstruct categorical labels from the cat_ columns
        feature_names = metadata.get("feature_names", [])
        cat_cols = [i for i, fn in enumerate(feature_names) if fn.startswith("cat_")]
        if cat_cols:
            cat_names = [feature_names[i].replace("cat_", "") for i in cat_cols]
            cat_matrix = X[:, cat_cols]
            labels["category"] = np.array([
                cat_names[np.argmax(row)] if np.any(row > 0) else "unknown"
                for row in cat_matrix
            ])

    # Volatility regime: median split on realized volatility if available
    feature_names = metadata.get("feature_names", [])
    vol_cols = [i for i, fn in enumerate(feature_names) if "volatility" in fn.lower()]
    if vol_cols:
        vol_values = X[:, vol_cols[0]]
        median_vol = np.median(vol_values)
        labels["volatility_regime"] = (vol_values > median_vol).astype(int)

    # Scale raw features for probe logistic regression convergence
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    with Timer("probes") as t:
        comparisons = run_standard_probes(
            X_raw=X_scaled,
            X_embed=Z,
            labels=labels,
            n_permutations=config["probes"]["n_permutations"],
        )

    print(f"\n  Probe results (time: {t.elapsed:.1f}s):")
    print("  ┌──────────────────────┬───────────┬───────────┬──────────┐")
    print("  │ Concept              │  Raw      │  Embed    │ Delta    │")
    print("  ├──────────────────────┼───────────┼───────────┼──────────┤")

    for comp in comparisons:
        m = comp.primary_metric
        raw_val = comp.raw_result.metrics[m]
        emb_val = comp.embed_result.metrics[m]
        delta = comp.improvement
        sign = "+" if delta >= 0 else ""
        print(f"  │ {comp.concept:<20s} │ {raw_val:>8.4f} │ {emb_val:>8.4f} │ {sign}{delta:>7.4f} │")

    print("  └──────────────────────┴───────────┴───────────┴──────────┘")

    return comparisons


def step_statistical_tests(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict,
) -> dict:
    """Step 7: Statistical significance tests."""
    print_header(7, "Statistical Tests")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    from models.statistics import test_orthogonality, test_predictive_power

    with Timer("stats") as t:
        # Orthogonality test
        orth = test_orthogonality(Z)

        # Per-dimension predictive power (Wald test)
        y_binary = (y >= 0.5).astype(int)
        pred_power = test_predictive_power(Z, y_binary, alpha=config["probes"]["significance_threshold"])

        # Overall predictive comparison: raw vs embedding logistic regression
        from sklearn.preprocessing import StandardScaler
        X_sc = StandardScaler().fit_transform(X)
        raw_clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        raw_clf.fit(X_sc, y_binary)
        raw_acc = accuracy_score(y_binary, raw_clf.predict(X_sc))
        raw_auc = roc_auc_score(y_binary, raw_clf.predict_proba(X_sc)[:, 1])

        emb_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        emb_clf.fit(Z, y_binary)
        emb_acc = accuracy_score(y_binary, emb_clf.predict(Z))
        emb_auc = roc_auc_score(y_binary, emb_clf.predict_proba(Z)[:, 1])

    print(f"  Time: {t.elapsed:.1f}s\n")

    # Orthogonality
    print(f"  Embedding orthogonality:")
    print(orth)

    # Predictive power
    print(f"\n  Per-dimension predictive power (Wald test):")
    print(pred_power)

    # Overall prediction comparison
    print(f"\n  Outcome prediction (logistic regression):")
    print_metric("Raw features accuracy", raw_acc)
    print_metric("Embedding accuracy", emb_acc)
    print_metric("Raw features AUC", raw_auc)
    print_metric("Embedding AUC", emb_auc)

    stats_results = {
        "orthogonality": {
            "mean_cosine_similarity": orth.mean_off_diagonal,
            "max_cosine_similarity": orth.max_off_diagonal,
            "n_correlated_pairs": orth.n_correlated_pairs,
        },
        "predictive_power": {
            "raw_accuracy": raw_acc,
            "embed_accuracy": emb_acc,
            "raw_auc": raw_auc,
            "embed_auc": emb_auc,
        },
        "significant_dimensions": {
            "count": pred_power.n_significant,
            "total": len(pred_power.dimension_names),
            "fraction": pred_power.n_significant / max(len(pred_power.dimension_names), 1),
        },
    }
    return stats_results


def step_visualize(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    metadata: dict,
    feature_names: list[str],
    raw_stats: dict,
    embed_stats: dict,
    probe_comparisons: list,
    config: dict,
) -> None:
    """Step 8: Generate all visualizations."""
    print_header(8, "Visualization")

    figures_dir = resolve_path(config["output"]["figures_dir"])
    out = str(figures_dir)

    try:
        from models.visualize import (
            plot_correlation_heatmap,
            plot_embedding_space,
            plot_probe_comparison,
            plot_vif_comparison,
        )
    except ImportError:
        print("  models.visualize not available, generating basic plots...")
        _generate_basic_figures(X, Z, y, feature_names, figures_dir)
        return

    with Timer("viz") as t:
        # 1. Correlation heatmaps (raw vs embedding)
        plot_correlation_heatmap(X, feature_names=feature_names, title="Raw Features", output_dir=out)
        emb_names = [f"emb_{i}" for i in range(Z.shape[1])]
        plot_correlation_heatmap(Z, feature_names=emb_names, title="Embeddings", output_dir=out)

        # 2. VIF comparison
        from models.statistics import compute_vif
        raw_vif = compute_vif(X, feature_names)
        emb_vif = compute_vif(Z, emb_names)
        plot_vif_comparison(
            raw_vif.vif_values, emb_vif.vif_values,
            raw_names=raw_vif.feature_names, embed_names=emb_vif.feature_names,
            output_dir=out,
        )

        # 3. Embedding space projections
        y_binary = (y >= 0.5).astype(int)
        plot_embedding_space(Z, y_binary, label_name="outcome", method="tsne", output_dir=out)

        # Category coloring if available
        cat_cols = [i for i, fn in enumerate(feature_names) if fn.startswith("cat_")]
        if cat_cols:
            cat_names_list = [feature_names[i].replace("cat_", "") for i in cat_cols]
            cat_matrix = X[:, cat_cols]
            cat_labels = np.array([
                cat_names_list[np.argmax(row)] if np.any(row > 0) else "other"
                for row in cat_matrix
            ])
            plot_embedding_space(Z, cat_labels, label_name="category", method="tsne", output_dir=out)

        # 4. Probe comparison chart
        if probe_comparisons:
            plot_probe_comparison(probe_comparisons, output_dir=out)

    print(f"  Figures saved to {figures_dir}")
    print(f"  Time: {t.elapsed:.1f}s")


def _generate_basic_figures(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Fallback: generate basic comparison plots with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Raw feature correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_raw = np.corrcoef(X, rowvar=False)
    sns.heatmap(corr_raw, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    ax.set_title("Raw Feature Correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_raw.png", dpi=150)
    plt.close(fig)

    # 2. Embedding correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_emb = np.corrcoef(Z, rowvar=False)
    sns.heatmap(corr_emb, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    ax.set_title("Embedding Dimension Correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_embed.png", dpi=150)
    plt.close(fig)

    # 3. VIF comparison
    from models.statistics import compute_vif

    raw_vif = compute_vif(X, feature_names)
    emb_names = [f"emb_{i}" for i in range(Z.shape[1])]
    emb_vif = compute_vif(Z, emb_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(raw_vif.feature_names, raw_vif.vif_values)
    axes[0].axvline(x=10, color="r", linestyle="--", label="VIF=10")
    axes[0].set_title("Raw Feature VIF")
    axes[0].set_xlabel("VIF")
    axes[0].legend()

    # Show only top 30 embedding dims by VIF for readability
    sorted_idx = np.argsort(emb_vif.vif_values)[-30:]
    axes[1].barh(
        [emb_vif.feature_names[i] for i in sorted_idx],
        emb_vif.vif_values[sorted_idx],
    )
    axes[1].axvline(x=10, color="r", linestyle="--", label="VIF=10")
    axes[1].set_title("Embedding VIF (top 30)")
    axes[1].set_xlabel("VIF")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "vif_comparison.png", dpi=150)
    plt.close(fig)

    # 4. t-SNE of embeddings colored by outcome
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, Z.shape[0] - 1))
        Z_2d = tsne.fit_transform(Z)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap="RdYlGn", alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label="Outcome")
        ax.set_title("t-SNE of Embeddings (colored by outcome)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(output_dir / "tsne_outcome.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: t-SNE failed: {e}")

    print(f"  Basic figures saved to {output_dir}")


def step_verdict(
    raw_stats: dict,
    embed_stats: dict,
    probe_comparisons: list,
    stats_results: dict,
    config: dict,
) -> dict:
    """Step 9: Final verdict -- does the hypothesis hold?"""
    print_header(9, "VERDICT")

    vif_threshold = config["statistics"]["vif_threshold"]

    # Criterion 1: VIF reduction
    embed_max_vif = embed_stats["max_vif"]
    vif_pass = embed_max_vif < vif_threshold
    vif_reduction = embed_stats["vif_reduction_factor"]

    # Criterion 2: Probe accuracy preserved (embedding >= 95% of raw)
    probes_preserved = True
    if probe_comparisons:
        for comp in probe_comparisons:
            m = comp.primary_metric
            raw_val = comp.raw_result.metrics[m]
            emb_val = comp.embed_result.metrics[m]
            if emb_val < raw_val * 0.95:
                probes_preserved = False
                break

    # Criterion 3: Stable significant dimensions
    sig_fraction = stats_results["significant_dimensions"]["fraction"]
    sig_pass = sig_fraction > 0.3

    # Criterion 4: Condition number improvement
    cond_pass = embed_stats["condition_reduction_factor"] > 5

    # Overall verdict
    criteria = {
        "VIF below threshold": vif_pass,
        "Probe accuracy preserved": probes_preserved,
        "Significant dimensions (>30%)": sig_pass,
        "Condition number improved 5x+": cond_pass,
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print("  Criteria evaluation:")
    for name, result in criteria.items():
        status = "PASS" if result else "FAIL"
        print(f"    [{status}] {name}")

    print(f"\n  Score: {passed}/{total} criteria met")

    if passed == total:
        verdict = "STRONG SUPPORT"
        print("\n  VERDICT: STRONG SUPPORT for H1")
        print("  Embeddings resolve multicollinearity while preserving information.")
    elif passed >= 3:
        verdict = "MODERATE SUPPORT"
        print("\n  VERDICT: MODERATE SUPPORT for H1")
        print("  Embeddings substantially improve feature space properties.")
    elif passed >= 2:
        verdict = "WEAK SUPPORT"
        print("\n  VERDICT: WEAK SUPPORT for H1")
        print("  Some improvement, but the evidence is not conclusive.")
    else:
        verdict = "NOT SUPPORTED"
        print("\n  VERDICT: H1 NOT SUPPORTED")
        print("  Embeddings do not meaningfully improve over raw features.")

    print(f"\n  Key metrics:")
    print_metric("Raw max VIF", raw_stats["max_vif"])
    print_metric("Embedding max VIF", embed_max_vif)
    print_metric("VIF reduction", f"{vif_reduction:.1f}x")
    print_metric("Condition # reduction", f"{embed_stats['condition_reduction_factor']:.1f}x")

    # Build serializable probe results
    probe_summary = {}
    for comp in probe_comparisons:
        m = comp.primary_metric
        probe_summary[comp.concept] = {
            "primary_metric": m,
            "raw_value": comp.raw_result.metrics[m],
            "embed_value": comp.embed_result.metrics[m],
            "improvement": comp.improvement,
            "raw_p_value": comp.raw_result.p_value,
            "embed_p_value": comp.embed_result.p_value,
        }

    report = {
        "verdict": verdict,
        "criteria": criteria,
        "score": f"{passed}/{total}",
        "raw_stats": raw_stats,
        "embed_stats": embed_stats,
        "probe_results": probe_summary,
        "stats_results": stats_results,
    }
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the embedding disentanglement experiment end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip data extraction, use cached features",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training, use saved checkpoint",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override model.embedding_dim from config",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override model.beta from config",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Label for this run (used in report filename)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.embedding_dim is not None:
        config["model"]["embedding_dim"] = args.embedding_dim
    if args.beta is not None:
        config["model"]["beta"] = args.beta

    print("=" * 70)
    print("  Disentangled Feature Analysis via Neural Embeddings")
    print("  Polymarket Testbed Experiment")
    print("=" * 70)
    print(f"\n  Config: {args.config}")
    print(f"  Skip extract: {args.skip_extract}")
    print(f"  Skip train:   {args.skip_train}")
    print(f"  embedding_dim: {config['model']['embedding_dim']}")
    print(f"  beta:          {config['model']['beta']}")
    if args.run_label:
        print(f"  Run label:     {args.run_label}")

    total_start = time.perf_counter()

    # --- Step 1: Data ---
    if args.skip_extract:
        X, y, metadata = step_load_cached(config)
    else:
        X, y, metadata = step_extract(config)

    feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])

    # --- Step 2: Feature summary & raw multicollinearity ---
    raw_stats = step_feature_summary(X, feature_names)

    # --- Step 3: Train model / load checkpoint ---
    if args.skip_train:
        model, embeddings = step_load_model(X, config)
    else:
        model, embeddings, _history = step_train(X, metadata, config)

    Z = embeddings

    # --- Step 4: Embedding analysis ---
    step_embedding_analysis(Z)

    # --- Step 5: Compare multicollinearity (THE KEY TEST) ---
    embed_stats = step_compare_multicollinearity(X, Z, feature_names, raw_stats, config)

    # --- Step 6: Linear probes ---
    probe_comparisons = step_linear_probes(X, Z, y, metadata, config)

    # --- Step 7: Statistical tests ---
    stats_results = step_statistical_tests(X, Z, y, feature_names, config)

    # --- Step 8: Visualizations ---
    step_visualize(
        X, Z, y, metadata, feature_names,
        raw_stats, embed_stats, probe_comparisons, config,
    )

    # --- Step 9: Verdict ---
    report = step_verdict(raw_stats, embed_stats, probe_comparisons, stats_results, config)

    # Save report
    reports_dir = resolve_path(config["output"]["reports_dir"])
    label = args.run_label or "default"
    report_path = reports_dir / f"experiment_report_{label}.json"
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["config"] = config

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'='*70}")
    print(f"  Experiment complete in {total_elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print(f"  Figures: {resolve_path(config['output']['figures_dir'])}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
