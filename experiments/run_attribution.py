#!/usr/bin/env python3
"""Feature Attribution: trace embedding dims back to interpretable input features.

For the best orthogonality-regularized supervised VAE (b1_a1_g1), computes:
1. Jacobian matrix d(embedding)/d(input) via torch autograd
2. Average |Jacobian| across samples -> 8x25 attribution matrix
3. Top driving features per dim with plain-language interpretations
4. Attribution heatmap (the paper's key figure)
5. Mutual information between each dim and the outcome
6. Cross-reference: Wald significance vs economic interpretability

Usage:
    python run_attribution.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Publication-quality style
STYLE_CONFIG = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
plt.rcParams.update(STYLE_CONFIG)
sns.set_style("whitegrid")

# Best config from sweep
BEST_CONFIG_NAME = "b1_a1_g1"
EMBEDDING_DIM = 8
TOP_K_FEATURES = 5


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the exact data used during sweep training.

    Uses the saved features.npy/labels.npy from the best config's data dir,
    which already has NaN imputation and zero-variance dropping applied.
    """
    data_dir = EXPERIMENT_DIR / "results" / f"orth_{BEST_CONFIG_NAME}_data"
    X = np.load(data_dir / "features.npy")
    y = np.load(data_dir / "labels.npy")

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])

    return X, y, feature_names


def load_model(input_dim: int) -> torch.nn.Module:
    """Load the best model checkpoint."""
    from models.autoencoder import AutoencoderConfig, MarketAutoencoder

    config = AutoencoderConfig(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=(256, 128),
        dropout=0.1,
        model_type="supervised_vae",
        beta=1.0,
        alpha=1.0,
        gamma=1.0,
    )
    model = MarketAutoencoder(config)

    ckpt_path = EXPERIMENT_DIR / "results" / f"orth_{BEST_CONFIG_NAME}_ckpt" / "best_model_supervised_vae.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model


def compute_jacobian(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute mean |d(embedding)/d(input)| across all samples.

    For each sample x, computes the Jacobian J[i,j] = d(z_i)/d(x_j)
    using torch.autograd. Returns the sample-averaged |J| matrix
    of shape (embedding_dim, input_dim).

    Args:
        model: Trained model with .encode() method.
        X_scaled: Scaled input features (N, D_in).
        batch_size: Batch size for Jacobian computation.

    Returns:
        Attribution matrix of shape (embedding_dim, input_dim).
    """
    device = next(model.parameters()).device
    n_samples = X_scaled.shape[0]
    input_dim = X_scaled.shape[1]

    # Accumulate attributions
    attr_sum = np.zeros((EMBEDDING_DIM, input_dim), dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = torch.tensor(
            X_scaled[start:end], dtype=torch.float32, device=device
        )
        batch.requires_grad_(True)

        # Forward pass to get mu (deterministic embedding)
        z = model.encode(batch)  # (B, embed_dim)

        # Compute gradient for each embedding dimension
        for dim_i in range(EMBEDDING_DIM):
            model.zero_grad()
            if batch.grad is not None:
                batch.grad.zero_()

            z_i_sum = z[:, dim_i].sum()
            z_i_sum.backward(retain_graph=(dim_i < EMBEDDING_DIM - 1))

            # |dz_i/dx| averaged over batch
            grad = batch.grad.abs().cpu().numpy()  # (B, input_dim)
            attr_sum[dim_i] += grad.sum(axis=0)

    # Average over all samples
    attr_matrix = attr_sum / n_samples
    return attr_matrix


def compute_mutual_information(
    embeddings: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Estimate mutual information between each embedding dim and outcome.

    Uses histogram-based MI estimation:
        MI(Z_i, Y) = H(Z_i) + H(Y) - H(Z_i, Y)

    where H is Shannon entropy computed from binned distributions.

    Args:
        embeddings: Embedding matrix (N, D).
        y: Binary labels (N,).
        n_bins: Number of bins for continuous variables.

    Returns:
        MI values array of shape (D,).
    """
    n_dims = embeddings.shape[1]
    mi_values = np.zeros(n_dims)

    # H(Y)
    py = np.array([np.mean(y == 0), np.mean(y == 1)])
    py = py[py > 0]
    h_y = -np.sum(py * np.log2(py))

    for i in range(n_dims):
        z_i = embeddings[:, i]

        # H(Z_i) via histogram
        counts_z, bin_edges = np.histogram(z_i, bins=n_bins)
        pz = counts_z / counts_z.sum()
        pz = pz[pz > 0]
        h_z = -np.sum(pz * np.log2(pz))

        # H(Z_i, Y) via 2D histogram
        counts_zy, _, _ = np.histogram2d(z_i, y, bins=[n_bins, 2])
        pzy = counts_zy / counts_zy.sum()
        pzy = pzy[pzy > 0]
        h_zy = -np.sum(pzy * np.log2(pzy))

        mi_values[i] = h_z + h_y - h_zy

    return mi_values


def interpret_dimension(
    dim_idx: int,
    top_features: list[tuple[str, float]],
    wald_coef: float,
    wald_p: float,
    mi: float,
) -> str:
    """Generate plain-language interpretation for an embedding dimension.

    Groups features by economic category and identifies the dominant theme.

    Args:
        dim_idx: Dimension index.
        top_features: List of (feature_name, attribution_score) tuples.
        wald_coef: Wald test coefficient for this dimension.
        wald_p: Wald test p-value.
        mi: Mutual information with outcome.

    Returns:
        Plain-language interpretation string.
    """
    # Feature category mapping
    categories = {
        "volume": ["volume_24h", "volume_1wk", "volume_total", "volume_acceleration",
                    "volume_vs_category_median"],
        "price": ["last_price", "one_day_price_change", "one_week_price_change",
                   "price_at_75pct_life", "price_range", "neg_risk",
                   "final_price_velocity"],
        "trading": ["trade_count", "trades_per_day", "avg_trade_size",
                     "max_trade_size", "trade_size_gini", "buy_sell_ratio",
                     "buy_volume_ratio", "late_volume_ratio"],
        "wallet": ["unique_wallet_count", "top_wallet_concentration",
                    "whale_buy_ratio", "avg_insider_score"],
        "structure": ["market_duration_days"],
    }

    # Count category hits in top features
    feat_names = [f for f, _ in top_features]
    cat_counts: dict[str, int] = {}
    cat_features: dict[str, list[str]] = {}
    for cat, members in categories.items():
        hits = [f for f in feat_names if f in members]
        if hits:
            cat_counts[cat] = len(hits)
            cat_features[cat] = hits

    # Determine dominant category
    if not cat_counts:
        dominant = "mixed"
    else:
        dominant = max(cat_counts, key=cat_counts.get)

    # Direction from Wald coefficient
    direction = "positive" if wald_coef > 0 else "negative"
    sig_str = f"p={wald_p:.2e}" if wald_p > 0 else "p<1e-300"

    # Build interpretation
    theme_labels = {
        "volume": "volume/liquidity",
        "price": "price dynamics",
        "trading": "trading behavior",
        "wallet": "wallet intelligence",
        "structure": "market structure",
        "mixed": "mixed signals",
    }

    theme = theme_labels.get(dominant, dominant)
    top_3_str = ", ".join(feat_names[:3])

    interp = (
        f"Dim {dim_idx}: {theme} ({direction} predictor, {sig_str}). "
        f"Driven by {top_3_str}."
    )

    # Add secondary category if present
    secondary_cats = [c for c in cat_counts if c != dominant and cat_counts[c] >= 2]
    if secondary_cats:
        sec = secondary_cats[0]
        sec_feats = ", ".join(cat_features[sec][:2])
        interp += f" Secondary: {theme_labels.get(sec, sec)} ({sec_feats})."

    return interp


def plot_attribution_heatmap(
    attr_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    wald_data: dict | None = None,
    mi_values: np.ndarray | None = None,
) -> Path:
    """Create the key figure: 8 dims x 25 features attribution heatmap.

    Features are ordered by total attribution. Dimensions are annotated
    with Wald significance and MI values on the right margin.

    Args:
        attr_matrix: Attribution matrix (embed_dim, input_dim).
        feature_names: Input feature names.
        output_dir: Directory to save figure.
        wald_data: Optional dict with per-dim Wald test results.
        mi_values: Optional mutual information per dim.

    Returns:
        Path to saved figure.
    """
    # Order features by total attribution (descending)
    total_attr = attr_matrix.sum(axis=0)
    feat_order = np.argsort(total_attr)[::-1]

    attr_ordered = attr_matrix[:, feat_order]
    names_ordered = [feature_names[i] for i in feat_order]

    # Normalize rows for better visual contrast
    row_max = attr_ordered.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1e-10)
    attr_norm = attr_ordered / row_max

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        attr_norm,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=names_ordered,
        yticklabels=[f"Dim {i}" for i in range(EMBEDDING_DIM)],
        cbar_kws={"label": "Normalized Attribution (row-scaled)"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Input Feature (ordered by total attribution)")
    ax.set_ylabel("Embedding Dimension")
    ax.set_title("Feature Attribution: Which Inputs Drive Each Embedding Dimension")
    plt.xticks(rotation=45, ha="right")

    # Add significance annotations on right margin
    if wald_data is not None:
        for dim_i in range(EMBEDDING_DIM):
            dim_key = f"dim_{dim_i}"
            if dim_key in wald_data:
                p = wald_data[dim_key]["p"]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(
                    len(names_ordered) + 0.3, dim_i + 0.5,
                    stars, ha="left", va="center", fontsize=10, fontweight="bold",
                )

    fig.tight_layout()
    fpath = output_dir / "attribution_heatmap.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_attribution_raw(
    attr_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> Path:
    """Create un-normalized attribution heatmap showing absolute gradient magnitudes."""
    total_attr = attr_matrix.sum(axis=0)
    feat_order = np.argsort(total_attr)[::-1]

    attr_ordered = attr_matrix[:, feat_order]
    names_ordered = [feature_names[i] for i in feat_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        attr_ordered,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=names_ordered,
        yticklabels=[f"Dim {i}" for i in range(EMBEDDING_DIM)],
        cbar_kws={"label": "Mean |d(emb)/d(input)|"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Input Feature (ordered by total attribution)")
    ax.set_ylabel("Embedding Dimension")
    ax.set_title("Feature Attribution: Absolute Gradient Magnitudes")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    fpath = output_dir / "attribution_heatmap_raw.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_dim_profiles(
    attr_matrix: np.ndarray,
    feature_names: list[str],
    wald_data: dict,
    mi_values: np.ndarray,
    output_dir: Path,
) -> Path:
    """Per-dimension bar charts showing top features, arranged in a 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for dim_i in range(EMBEDDING_DIM):
        ax = axes[dim_i // 4][dim_i % 4]
        attr = attr_matrix[dim_i]
        top_idx = np.argsort(attr)[::-1][:TOP_K_FEATURES]

        names = [feature_names[i] for i in top_idx]
        values = attr[top_idx]

        # Normalize to percentage of max
        if values.max() > 0:
            values_pct = values / values.max() * 100
        else:
            values_pct = values

        bars = ax.barh(range(len(names)), values_pct, color="#ef5350", alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Attribution (%)", fontsize=8)

        # Title with Wald info
        dim_key = f"dim_{dim_i}"
        p_val = wald_data[dim_key]["p"]
        coef = wald_data[dim_key]["coef"]
        sign = "+" if coef > 0 else "-"
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        ax.set_title(f"Dim {dim_i} ({sign}, MI={mi_values[dim_i]:.3f}) {stars}", fontsize=10)

    fig.suptitle(
        "Per-Dimension Feature Attribution Profiles\n(top 5 features per dim, Wald significance noted)",
        fontsize=14,
    )
    fig.tight_layout()

    fpath = output_dir / "dim_profiles.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_mi_vs_wald(
    mi_values: np.ndarray,
    wald_data: dict,
    output_dir: Path,
) -> Path:
    """Scatter plot: mutual information vs Wald |z-score| per dimension."""
    fig, ax = plt.subplots(figsize=(8, 6))

    z_scores = []
    for i in range(EMBEDDING_DIM):
        z_scores.append(abs(wald_data[f"dim_{i}"]["z"]))

    ax.scatter(mi_values, z_scores, s=120, c="#5c6bc0", edgecolors="white", linewidth=1.5, zorder=5)

    for i in range(EMBEDDING_DIM):
        ax.annotate(
            f"Dim {i}", (mi_values[i], z_scores[i]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Mutual Information with Outcome (bits)")
    ax.set_ylabel("|Wald z-score|")
    ax.set_title("Dimension Significance vs Information Content")
    ax.axhline(y=1.96, color="red", linestyle="--", alpha=0.5, label="p=0.05 threshold")
    ax.legend()

    fig.tight_layout()
    fpath = output_dir / "mi_vs_wald.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 70)
    print("  Feature Attribution Analysis")
    print(f"  Model: Orth-Reg Supervised VAE ({BEST_CONFIG_NAME})")
    print("=" * 70)

    # Step 1: Load data and model
    print("\n[1/7] Loading data and model...")
    X, y_binary, feature_names = load_data()
    n_features = X.shape[1]
    print(f"  Data: {X.shape[0]} samples, {n_features} features")
    print(f"  Features: {feature_names}")

    # X is already scaled (StandardScaler applied during sweep)
    X_scaled = X.astype(np.float32)

    model = load_model(input_dim=n_features)
    print(f"  Model loaded from orth_{BEST_CONFIG_NAME}_ckpt/")

    # Extract embeddings
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Z = model.get_embedding(X_tensor)
    print(f"  Embeddings: shape {Z.shape}")

    # Load Wald test results from sweep report
    report_path = EXPERIMENT_DIR / "results" / "orth_sweep_reports" / "orth_sweep_report.json"
    with open(report_path) as f:
        sweep_report = json.load(f)
    best_config = next(c for c in sweep_report["configs"] if c["name"] == BEST_CONFIG_NAME)
    wald_data = best_config["wald_test"]["per_dim"]

    output_dir = EXPERIMENT_DIR / "results" / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Compute Jacobian
    print("\n[2/7] Computing Jacobian d(embedding)/d(input)...")
    t0 = time.perf_counter()
    attr_matrix = compute_jacobian(model, X_scaled, batch_size=128)
    t_jac = time.perf_counter() - t0
    print(f"  Jacobian computed in {t_jac:.1f}s")
    print(f"  Attribution matrix shape: {attr_matrix.shape} (embed_dim x input_dim)")

    # Step 3: Top features per dim
    print("\n[3/7] Identifying top driving features per dimension...")
    dim_interpretations = []
    dim_top_features = {}

    for dim_i in range(EMBEDDING_DIM):
        attr = attr_matrix[dim_i]
        top_idx = np.argsort(attr)[::-1][:TOP_K_FEATURES]
        top_feats = [(feature_names[j], float(attr[j])) for j in top_idx]
        dim_top_features[f"dim_{dim_i}"] = top_feats

        dim_key = f"dim_{dim_i}"
        print(f"\n  Dim {dim_i} (Wald coef={wald_data[dim_key]['coef']:.3f}, "
              f"p={wald_data[dim_key]['p']:.2e}):")
        for fname, score in top_feats:
            pct = score / attr.max() * 100 if attr.max() > 0 else 0
            print(f"    {fname:<30} {score:.6f} ({pct:.0f}%)")

    # Step 4: Mutual information
    print("\n[4/7] Computing mutual information between each dim and outcome...")
    mi_values = compute_mutual_information(Z, y_binary, n_bins=20)
    for i in range(EMBEDDING_DIM):
        print(f"  Dim {i}: MI = {mi_values[i]:.4f} bits")

    # Step 5: Plain-language interpretations
    print("\n[5/7] Generating interpretations...")
    for dim_i in range(EMBEDDING_DIM):
        dim_key = f"dim_{dim_i}"
        interp = interpret_dimension(
            dim_i,
            dim_top_features[dim_key],
            wald_data[dim_key]["coef"],
            wald_data[dim_key]["p"],
            mi_values[dim_i],
        )
        dim_interpretations.append(interp)
        print(f"  {interp}")

    # Step 6: Cross-reference significance vs interpretability
    print("\n[6/7] Cross-referencing Wald significance vs MI...")
    wald_z = np.array([abs(wald_data[f"dim_{i}"]["z"]) for i in range(EMBEDDING_DIM)])
    # Rank by Wald |z|
    wald_rank = np.argsort(wald_z)[::-1]
    # Rank by MI
    mi_rank = np.argsort(mi_values)[::-1]

    print(f"\n  {'Dim':<6} {'|Wald z|':>10} {'MI (bits)':>10} {'Wald rank':>10} {'MI rank':>10}")
    print(f"  {'-'*48}")
    for i in range(EMBEDDING_DIM):
        wr = int(np.where(wald_rank == i)[0][0]) + 1
        mr = int(np.where(mi_rank == i)[0][0]) + 1
        print(f"  Dim {i:<2} {wald_z[i]:>10.2f} {mi_values[i]:>10.4f} {wr:>10} {mr:>10}")

    # Rank correlation
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(wald_z, mi_values)
    print(f"\n  Spearman rank correlation (Wald |z| vs MI): rho={rho:.3f}, p={p_rho:.4f}")
    if rho > 0.5 and p_rho < 0.1:
        print("  -> Strong alignment: statistically significant dims also carry the most outcome info")
    elif rho > 0:
        print("  -> Moderate/weak alignment: significance and MI partially correlated")
    else:
        print("  -> No alignment: significance and MI are independent aspects")

    # Step 7: Generate figures
    print("\n[7/7] Generating figures...")

    plot_attribution_heatmap(attr_matrix, feature_names, output_dir, wald_data, mi_values)
    plot_attribution_raw(attr_matrix, feature_names, output_dir)
    plot_dim_profiles(attr_matrix, feature_names, wald_data, mi_values, output_dir)
    plot_mi_vs_wald(mi_values, wald_data, output_dir)

    # Save results
    results = {
        "experiment": "feature_attribution",
        "model": BEST_CONFIG_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_names": feature_names,
        "n_features": n_features,
        "n_dims": EMBEDDING_DIM,
        "attribution_matrix": attr_matrix.tolist(),
        "dim_top_features": {
            k: [{"feature": f, "attribution": s} for f, s in v]
            for k, v in dim_top_features.items()
        },
        "mutual_information": {
            f"dim_{i}": float(mi_values[i]) for i in range(EMBEDDING_DIM)
        },
        "wald_significance": {
            f"dim_{i}": {
                "coef": wald_data[f"dim_{i}"]["coef"],
                "z": wald_data[f"dim_{i}"]["z"],
                "p": wald_data[f"dim_{i}"]["p"],
            }
            for i in range(EMBEDDING_DIM)
        },
        "interpretations": {
            f"dim_{i}": dim_interpretations[i] for i in range(EMBEDDING_DIM)
        },
        "cross_reference": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
        },
    }

    report_path = output_dir / "attribution_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save raw attribution matrix as npy for downstream use
    np.save(output_dir / "attribution_matrix.npy", attr_matrix)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}")
    print(f"  Attribution analysis complete in {total_elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print(f"  Figures: {output_dir}/")
    print(f"    - attribution_heatmap.png (key figure)")
    print(f"    - attribution_heatmap_raw.png (absolute gradients)")
    print(f"    - dim_profiles.png (per-dim top features)")
    print(f"    - mi_vs_wald.png (significance vs information)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
