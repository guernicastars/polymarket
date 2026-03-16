"""Publication-quality figures for the dissertation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_VENN = False

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
except ImportError:
    pass

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_blind_spot_venn(
    blind_spot_sizes: Dict[str, int],
    intersections: Dict[str, int],
    total_samples: int,
    save_path: Optional[str] = None,
    title: str = "Blind Spot Complementarity",
):
    """
    Venn diagram of agent blind spots.
    Figure 1 of the dissertation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    agents = list(blind_spot_sizes.keys())
    n = len(agents)

    if n == 2 and HAS_VENN:
        v = venn2(
            subsets=(
                blind_spot_sizes[agents[0]],
                blind_spot_sizes[agents[1]],
                intersections.get(f"{agents[0]}&{agents[1]}", 0),
            ),
            set_labels=tuple(agents),
            ax=ax,
        )
    elif n == 3 and HAS_VENN:
        a, b, c = agents
        v = venn3(
            subsets=(
                blind_spot_sizes[a],
                blind_spot_sizes[b],
                intersections.get(f"{a}&{b}", 0),
                blind_spot_sizes[c],
                intersections.get(f"{a}&{c}", 0),
                intersections.get(f"{b}&{c}", 0),
                intersections.get(f"{a}&{b}&{c}", 0),
            ),
            set_labels=(a, b, c),
            ax=ax,
        )
    else:
        # Fallback: bar chart
        ax.bar(range(n), [blind_spot_sizes[a] for a in agents], color="steelblue")
        collective = intersections.get("collective", 0)
        ax.axhline(y=collective, color="red", linestyle="--", label=f"Collective: {collective}")
        ax.set_xticks(range(n))
        ax.set_xticklabels(agents, rotation=45)
        ax.set_ylabel("Blind Spot Size")
        ax.legend()

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_pi_n_vs_pi_u(
    method_names: List[str],
    pi_n_scores: List[float],
    pi_u_scores: List[float],
    metric_name: str = "MSE",
    save_path: Optional[str] = None,
    title: str = r"Performance: $\Pi_N$ vs $\Pi_U$",
):
    """
    Bar chart comparing high-evidence vs low-evidence performance.
    Figure 2 of the dissertation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(method_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, pi_n_scores, width, label=r"$\Pi_N$ (high evidence)",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, pi_u_scores, width, label=r"$\Pi_U$ (low evidence)",
                   color="coral", alpha=0.8)

    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_complementarity_over_training(
    steps: List[int],
    curves: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Complementarity Score vs Training Step",
):
    """
    Line plot of complementarity C increasing over training.
    Figure 3 of the dissertation.

    curves: dict mapping method name -> list of C values over training.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for idx, (name, values) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(steps[:len(values)], values, label=name, color=color, linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Complementarity Score C")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_cumulative_regret(
    steps: List[int],
    curves: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Cumulative Regret",
):
    """Plot cumulative regret for different methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for idx, (name, values) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(steps[:len(values)], values, label=name, color=color, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_reliability_diagram(
    diagram_data: dict,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
):
    """Plot calibration reliability diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    centers = diagram_data["bin_centers"]
    accuracies = diagram_data["bin_accuracies"]
    counts = diagram_data["bin_counts"]

    ax.bar(centers, accuracies, width=0.1, alpha=0.5, color="steelblue",
           edgecolor="navy", label="Empirical")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Empirical Probability")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_evidence_heatmap(
    evidence_matrix: np.ndarray,
    agent_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Agent Evidence Weights",
):
    """Heatmap of evidence weights across agents and samples."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    im = ax.imshow(evidence_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names)
    ax.set_xlabel("Sample Index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="V(x)")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
