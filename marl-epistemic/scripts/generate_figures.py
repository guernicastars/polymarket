"""
Generate publication-quality figures for the dissertation.

Three key figures:
1. Blind spot Venn diagram
2. Pi_N vs Pi_U performance gap
3. Complementarity score vs training step
"""

import argparse
import os

import numpy as np
import torch

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.environments.prediction_market import SyntheticEventGenerator, PredictionMarketEnv
from src.metrics.blind_spot import (
    compute_blind_spot,
    blind_spot_overlap,
    complementarity_score,
    pairwise_overlap_matrix,
)
from src.utils.plotting import (
    plot_blind_spot_venn,
    plot_pi_n_vs_pi_u,
    plot_complementarity_over_training,
    plot_reliability_diagram,
)


def generate_figure1(save_dir: str):
    """Figure 1: Blind Spot Venn Diagram."""
    print("Generating Figure 1: Blind Spot Venn Diagram...")

    # Create a synthetic task where different architectures have different blind spots
    torch.manual_seed(42)
    n_features = 20
    n_samples = 1000

    x = torch.randn(n_samples, n_features)

    # True function: mix of linear + nonlinear + long-range
    w = torch.randn(n_features) * 0.3
    y_linear = x @ w
    y_local = (x[:, :3].prod(dim=-1)) * 0.5
    y_longrange = (x[:, 0] * x[:, -1]) * 0.4
    y_true = (y_linear + y_local + y_longrange).unsqueeze(-1)

    # Create agents
    agents = [
        LinearAgent(n_features, 1),
        MLPAgent(n_features, 1),
        CNNAgent(n_features, 1),
        AttentionAgent(n_features, 1),
    ]

    # Train each agent briefly
    for agent in agents:
        agent.setup_optimizer(lr=1e-3)
        agent.train_mode()
        for _ in range(200):
            pred = agent.predict(x)
            loss = torch.nn.functional.mse_loss(pred, y_true)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        agent.eval_mode()

    # Compute blind spots
    threshold = 0.1
    blind_spots = {}
    for i, agent in enumerate(agents):
        mask = compute_blind_spot(agent, x, y_true, threshold)
        name = agent.hypothesis_class_name
        blind_spots[name] = mask.sum().item()

    # Compute pairwise intersections
    names = [a.hypothesis_class_name for a in agents]
    masks = [compute_blind_spot(a, x, y_true, threshold) for a in agents]
    intersections = {}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            key = f"{names[i]}&{names[j]}"
            intersections[key] = (masks[i] & masks[j]).sum().item()

    # Collective
    collective = masks[0]
    for m in masks[1:]:
        collective = collective & m
    intersections["collective"] = collective.sum().item()

    fig = plot_blind_spot_venn(
        blind_spots, intersections, n_samples,
        save_path=os.path.join(save_dir, "figure1_blind_spot_venn.png"),
        title="Blind Spot Complementarity Across Hypothesis Classes",
    )
    print(f"  Saved. Individual: {blind_spots}, Collective: {intersections['collective']}")


def generate_figure2(save_dir: str):
    """Figure 2: Pi_N vs Pi_U Performance Gap."""
    print("Generating Figure 2: Pi_N vs Pi_U Performance Gap...")

    # Placeholder data (would come from Experiment 2 runs)
    methods = ["Best Single", "Simple Avg", "Accuracy Wt.", "Keynesian Wt.", "Ev-LOLA"]
    pi_n = [0.042, 0.035, 0.031, 0.028, 0.025]
    pi_u = [0.185, 0.162, 0.148, 0.112, 0.095]

    plot_pi_n_vs_pi_u(
        methods, pi_n, pi_u,
        metric_name="MSE",
        save_path=os.path.join(save_dir, "figure2_pi_n_vs_pi_u.png"),
    )
    print("  Saved (using placeholder data).")


def generate_figure3(save_dir: str):
    """Figure 3: Complementarity Score vs Training Step."""
    print("Generating Figure 3: Complementarity vs Training...")

    steps = list(range(0, 10001, 100))
    n = len(steps)

    # Simulated curves
    curves = {
        "Homogeneous (MLP x4)": [0.05 + 0.1 * (1 - np.exp(-s / 3000)) + np.random.randn() * 0.02 for s in steps],
        "Diverse (no LOLA)": [0.1 + 0.4 * (1 - np.exp(-s / 2000)) + np.random.randn() * 0.02 for s in steps],
        "Diverse + LOLA": [0.1 + 0.55 * (1 - np.exp(-s / 1500)) + np.random.randn() * 0.02 for s in steps],
        "Diverse + Ev-LOLA": [0.15 + 0.7 * (1 - np.exp(-s / 1200)) + np.random.randn() * 0.02 for s in steps],
    }
    # Clip to [0, 1]
    for k in curves:
        curves[k] = [max(0, min(1, v)) for v in curves[k]]

    plot_complementarity_over_training(
        steps, curves,
        save_path=os.path.join(save_dir, "figure3_complementarity.png"),
    )
    print("  Saved (using simulated curves).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generate_figure1(args.output_dir)
    generate_figure2(args.output_dir)
    generate_figure3(args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
