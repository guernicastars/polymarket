"""Publication-quality visualizations for the embedding experiment.

All figures are saved to the results/ directory and designed for
inclusion in a paper's supplementary materials: proper axis labels,
legends, font sizes, and tight layouts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)

# Global style settings for publication quality
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


def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_name: str = "label",
    method: str = "tsne",
    output_dir: str = "results",
    perplexity: int = 30,
) -> Path:
    """2D projection of embedding space colored by label.

    Projects high-dimensional embeddings to 2D using t-SNE or UMAP,
    then creates a scatter plot colored by the provided labels.

    Args:
        embeddings: Embedding matrix (N, D).
        labels: Label array (N,) for coloring — categorical or numeric.
        label_name: Name of the label for the legend/title.
        method: 'tsne' or 'umap'.
        output_dir: Directory to save the figure.
        perplexity: Perplexity for t-SNE (ignored for UMAP).

    Returns:
        Path to saved figure.
    """
    out = _ensure_dir(output_dir)

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(embeddings)
        method_label = f"t-SNE (perplexity={perplexity})"
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = np.unique(labels)
    is_categorical = isinstance(labels[0], str) or len(unique_labels) <= 20

    if is_categorical:
        palette = sns.color_palette("husl", n_colors=len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[palette[i]], label=str(lbl), s=15, alpha=0.6, edgecolors="none")
        ax.legend(title=label_name, bbox_to_anchor=(1.05, 1), loc="upper left",
                  markerscale=2, frameon=True)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=labels.astype(float), cmap="viridis", s=15, alpha=0.6,
                             edgecolors="none")
        plt.colorbar(scatter, ax=ax, label=label_name)

    ax.set_xlabel(f"{method_label} Dimension 1")
    ax.set_ylabel(f"{method_label} Dimension 2")
    ax.set_title(f"Embedding Space Colored by {label_name}")

    fpath = out / f"embedding_space_{label_name}_{method}.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_correlation_heatmap(
    X: np.ndarray,
    title: str = "Feature Correlation Matrix",
    feature_names: list[str] | None = None,
    output_dir: str = "results",
    filename: str | None = None,
) -> Path:
    """Correlation matrix heatmap.

    Args:
        X: Feature matrix (N, D).
        title: Plot title.
        feature_names: Optional feature names for axis labels.
        output_dir: Directory to save the figure.
        filename: Custom filename (auto-generated if None).

    Returns:
        Path to saved figure.
    """
    out = _ensure_dir(output_dir)
    corr = np.corrcoef(X.T)
    d = corr.shape[0]

    fig_size = max(6, min(d * 0.4, 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    show_labels = d <= 30
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", norm=norm, center=0,
        square=True, linewidths=0.5 if d <= 30 else 0,
        xticklabels=feature_names if show_labels else False,
        yticklabels=feature_names if show_labels else False,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    )
    ax.set_title(title)
    if show_labels:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    fname = filename or title.lower().replace(" ", "_") + ".png"
    fpath = out / fname
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_vif_comparison(
    vif_raw: np.ndarray,
    vif_embed: np.ndarray,
    raw_names: list[str] | None = None,
    embed_names: list[str] | None = None,
    output_dir: str = "results",
) -> Path:
    """Side-by-side VIF bar charts for raw features vs embeddings.

    Args:
        vif_raw: VIF values for raw features (D_raw,).
        vif_embed: VIF values for embeddings (D_embed,).
        raw_names: Feature names for raw features.
        embed_names: Dimension names for embeddings.
        output_dir: Directory to save the figure.

    Returns:
        Path to saved figure.
    """
    out = _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw features
    ax = axes[0]
    idx_raw = np.argsort(vif_raw)[::-1][:30]  # Top 30 by VIF
    names_raw = raw_names or [f"feat_{i}" for i in range(len(vif_raw))]
    colors_raw = ["#d32f2f" if v > 10 else "#ff9800" if v > 5 else "#4caf50"
                  for v in vif_raw[idx_raw]]
    ax.barh(range(len(idx_raw)), vif_raw[idx_raw], color=colors_raw)
    ax.set_yticks(range(len(idx_raw)))
    ax.set_yticklabels([names_raw[i] for i in idx_raw], fontsize=8)
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7, label="Moderate (5)")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="Severe (10)")
    ax.set_xlabel("VIF")
    ax.set_title(f"Raw Features (n={len(vif_raw)})")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Embeddings
    ax = axes[1]
    idx_emb = np.argsort(vif_embed)[::-1][:30]
    names_emb = embed_names or [f"emb_{i}" for i in range(len(vif_embed))]
    colors_emb = ["#d32f2f" if v > 10 else "#ff9800" if v > 5 else "#4caf50"
                  for v in vif_embed[idx_emb]]
    ax.barh(range(len(idx_emb)), vif_embed[idx_emb], color=colors_emb)
    ax.set_yticks(range(len(idx_emb)))
    ax.set_yticklabels([names_emb[i] for i in idx_emb], fontsize=8)
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7, label="Moderate (5)")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="Severe (10)")
    ax.set_xlabel("VIF")
    ax.set_title(f"Embedding Dimensions (n={len(vif_embed)})")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig.suptitle("Variance Inflation Factor: Raw Features vs Embeddings", fontsize=14)
    fig.tight_layout()

    fpath = out / "vif_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_probe_comparison(
    probe_results: list,
    output_dir: str = "results",
) -> Path:
    """Grouped bar chart comparing probe performance: raw vs embedding.

    Args:
        probe_results: List of ProbeComparison objects from probes.py.
        output_dir: Directory to save the figure.

    Returns:
        Path to saved figure.
    """
    out = _ensure_dir(output_dir)

    concepts = []
    raw_scores = []
    embed_scores = []
    metric_names = []

    for comp in probe_results:
        m = comp.primary_metric
        concepts.append(comp.concept)
        raw_scores.append(comp.raw_result.metrics[m])
        embed_scores.append(comp.embed_result.metrics[m])
        metric_names.append(m)

    x = np.arange(len(concepts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_raw = ax.bar(x - width / 2, raw_scores, width, label="Raw Features",
                      color="#5c6bc0", alpha=0.85)
    bars_emb = ax.bar(x + width / 2, embed_scores, width, label="Embeddings",
                      color="#26a69a", alpha=0.85)

    # Value labels on bars
    for bars in [bars_raw, bars_emb]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Probe Task")
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Performance: Raw Features vs Embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({m})" for c, m in zip(concepts, metric_names)])
    ax.legend()
    ax.set_ylim(0, max(max(raw_scores), max(embed_scores)) * 1.15)

    fig.tight_layout()
    fpath = out / "probe_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_feature_attribution(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 20,
    output_dir: str = "results",
) -> Path:
    """Gradient-based feature attribution: which inputs activate which embedding dims.

    Computes the Jacobian dz/dx averaged over the dataset and visualizes
    the attribution matrix as a heatmap.

    For input feature j and embedding dimension i:

        A_{ij} = E_x [ |d z_i / d x_j| ]

    High values indicate that input feature j strongly influences
    embedding dimension i.

    Args:
        model: Trained MarketAutoencoder (PyTorch).
        X: Input feature matrix (N, D) as numpy array.
        feature_names: Names of input features.
        top_k: Number of top input features to show.
        output_dir: Directory to save the figure.

    Returns:
        Path to saved figure.
    """
    import torch

    out = _ensure_dir(output_dir)
    model.eval()

    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)

    # Forward pass to get embeddings
    z = model.encode(X_tensor)
    embed_dim = z.shape[1]

    # Compute gradients for each embedding dimension
    attributions = np.zeros((X.shape[1], embed_dim))
    for i in range(embed_dim):
        model.zero_grad()
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()
        z_i = z[:, i].sum()
        z_i.backward(retain_graph=(i < embed_dim - 1))
        grad = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        attributions[:, i] = grad

    # Select top-k input features by total attribution
    total_attr = attributions.sum(axis=1)
    top_idx = np.argsort(total_attr)[::-1][:top_k]

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    fig, ax = plt.subplots(figsize=(max(8, embed_dim * 0.3), max(6, top_k * 0.3)))
    sns.heatmap(
        attributions[top_idx],
        ax=ax, cmap="YlOrRd",
        xticklabels=[f"emb_{i}" for i in range(embed_dim)],
        yticklabels=[feature_names[i] for i in top_idx],
        cbar_kws={"label": "Mean |gradient|"},
    )
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Input Feature")
    ax.set_title("Feature Attribution: Input -> Embedding Mapping")

    fpath = out / "feature_attribution.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath


def plot_training_history(
    history: dict[str, list],
    output_dir: str = "results",
) -> Path:
    """Plot training and validation loss curves.

    Args:
        history: Dict from train.py with 'train_loss', 'val_loss', etc.
        output_dir: Directory to save the figure.

    Returns:
        Path to saved figure.
    """
    out = _ensure_dir(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    n_plots = 2 if "kl_loss" in history and any(v > 0 for v in history["kl_loss"]) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Total loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#5c6bc0")
    ax.plot(epochs, history["val_loss"], label="Validation", color="#ef5350")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")

    # KL divergence (VAE only)
    if n_plots > 1:
        ax = axes[1]
        ax.plot(epochs, history["kl_loss"], label="KL Divergence", color="#ff9800")
        ax.plot(epochs, history["recon_loss"], label="Reconstruction", color="#4caf50")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Component")
        ax.set_title("Loss Components")
        ax.legend()

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()

    fpath = out / "training_history.png"
    fig.savefig(fpath)
    plt.close(fig)
    logger.info("Saved: %s", fpath)
    return fpath
