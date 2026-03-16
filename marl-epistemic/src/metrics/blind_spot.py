"""
Blind spot detection and complementarity metrics.

B_i = {x : irreducible error of agent i on x exceeds threshold}

The key insight: if agents have DIFFERENT hypothesis classes,
their blind spots B_i are different, and the collective blind spot
B = intersection(B_i) is strictly smaller than any individual B_i.
"""

from typing import List

import torch
from torch import Tensor

from src.agents.base import BaseAgent


def compute_blind_spot(
    agent: BaseAgent,
    x_test: Tensor,
    y_test: Tensor,
    threshold: float = 0.1,
) -> Tensor:
    """
    Identify inputs in agent's blind spot.

    B_i = {x in test_data : irreducible_error(agent, x) > threshold}

    After training to convergence, residual error reflects the
    expressiveness limit of the hypothesis class, not data scarcity.

    Args:
        agent: Trained agent.
        x_test: Test inputs. Shape: (N, input_dim).
        y_test: Test targets. Shape: (N, output_dim).
        threshold: Error threshold for blind spot membership.

    Returns:
        Boolean mask of shape (N,). True = in blind spot.
    """
    agent.eval_mode()
    with torch.no_grad():
        y_pred = agent.predict(x_test)
        errors = (y_pred - y_test).pow(2).mean(dim=-1)  # (N,)
    return errors > threshold


def blind_spot_overlap(B_i: Tensor, B_j: Tensor) -> float:
    """
    Jaccard index of two blind spots: |B_i ∩ B_j| / |B_i ∪ B_j|.

    J = 0: completely disjoint blind spots (ideal for complementarity).
    J = 1: identical blind spots (no benefit from diversity).

    Args:
        B_i, B_j: Boolean masks of shape (N,).

    Returns:
        Jaccard index in [0, 1].
    """
    intersection = (B_i & B_j).sum().float()
    union = (B_i | B_j).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def collective_blind_spot(
    agents: List[BaseAgent],
    x_test: Tensor,
    y_test: Tensor,
    threshold: float = 0.1,
) -> Tensor:
    """
    Collective blind spot B = intersection of all B_i.

    x is in B iff ALL agents fail on x (no agent can represent the
    true function at x within their hypothesis class).

    Returns:
        Boolean mask of shape (N,). True = in collective blind spot.
    """
    masks = [compute_blind_spot(a, x_test, y_test, threshold) for a in agents]
    # Intersection: all agents must have x in their blind spot
    collective = masks[0]
    for m in masks[1:]:
        collective = collective & m
    return collective


def complementarity_score(
    agents: List[BaseAgent],
    x_test: Tensor,
    y_test: Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Complementarity score C = 1 - |B| / min_i |B_i|.

    C = 0: no complementarity (all agents have same blind spot).
    C = 1: perfect complementarity (collective has no blind spot).

    This is the key metric for Claim 1 of the dissertation:
    agents with diverse hypothesis classes collectively learn strictly
    more than any single agent.

    Returns:
        C in [0, 1].
    """
    individual_masks = [
        compute_blind_spot(a, x_test, y_test, threshold) for a in agents
    ]
    individual_sizes = [m.sum().item() for m in individual_masks]

    if min(individual_sizes) == 0:
        return 1.0  # no blind spots at all

    collective_mask = individual_masks[0]
    for m in individual_masks[1:]:
        collective_mask = collective_mask & m

    collective_size = collective_mask.sum().item()
    min_individual = min(individual_sizes)

    return 1.0 - collective_size / min_individual


def blind_spot_error_profile(
    agents: List[BaseAgent],
    x_test: Tensor,
    y_test: Tensor,
) -> dict:
    """
    Compute detailed error profile for blind spot analysis.

    Returns dict with per-agent errors and boolean masks at multiple thresholds.
    Useful for generating the Venn diagram figure.
    """
    errors = {}
    for i, agent in enumerate(agents):
        agent.eval_mode()
        with torch.no_grad():
            y_pred = agent.predict(x_test)
            err = (y_pred - y_test).pow(2).mean(dim=-1)
        errors[f"agent_{i}"] = err
        errors[f"agent_{i}_name"] = agent.hypothesis_class_name

    # Compute masks at multiple thresholds for sensitivity analysis
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    for t in thresholds:
        for i in range(len(agents)):
            errors[f"B_{i}_t{t}"] = errors[f"agent_{i}"] > t

    return errors


def pairwise_overlap_matrix(
    agents: List[BaseAgent],
    x_test: Tensor,
    y_test: Tensor,
    threshold: float = 0.1,
) -> Tensor:
    """
    Compute NxN matrix of pairwise blind spot Jaccard overlaps.

    Low off-diagonal values = agents have complementary blind spots.
    """
    masks = [compute_blind_spot(a, x_test, y_test, threshold) for a in agents]
    n = len(agents)
    overlap = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            overlap[i, j] = blind_spot_overlap(masks[i], masks[j])
    return overlap
