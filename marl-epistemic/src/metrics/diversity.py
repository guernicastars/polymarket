"""Hypothesis class diversity and prediction disagreement measures."""

from typing import List

import torch
from torch import Tensor

from src.agents.base import BaseAgent


def prediction_disagreement(
    agents: List[BaseAgent],
    x: Tensor,
) -> float:
    """
    Mean pairwise disagreement between agent predictions.

    D = (2 / N(N-1)) * sum_{i<j} ||y_hat_i - y_hat_j||^2

    High disagreement suggests agents have learned different functions,
    which is a necessary (but not sufficient) condition for blind spot
    complementarity.

    Returns: scalar disagreement value.
    """
    predictions = []
    for agent in agents:
        agent.eval_mode()
        with torch.no_grad():
            predictions.append(agent.predict(x))

    n = len(predictions)
    if n < 2:
        return 0.0

    total_disagreement = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = (predictions[i] - predictions[j]).pow(2).mean().item()
            total_disagreement += diff
            count += 1

    return total_disagreement / count


def hypothesis_class_diversity(agents: List[BaseAgent]) -> dict:
    """
    Structural diversity metrics of the agent ensemble.

    Returns:
        - n_unique_classes: number of distinct hypothesis class names
        - class_names: list of hypothesis class names
        - param_count_variance: variance of parameter counts (proxy for capacity diversity)
        - architecture_types: set of unique types
    """
    class_names = [a.hypothesis_class_name for a in agents]
    unique_classes = set(class_names)
    param_counts = [sum(p.numel() for p in a.parameters) for a in agents]
    mean_params = sum(param_counts) / len(param_counts)
    var_params = sum((p - mean_params) ** 2 for p in param_counts) / len(param_counts)

    return {
        "n_unique_classes": len(unique_classes),
        "class_names": class_names,
        "unique_classes": list(unique_classes),
        "param_counts": param_counts,
        "param_count_variance": var_params,
    }


def functional_diversity(
    agents: List[BaseAgent],
    x: Tensor,
) -> dict:
    """
    Measure functional diversity: how different are the learned functions?

    Computes:
    - prediction_covariance: covariance matrix of agent predictions
    - effective_ensemble_size: N_eff = (sum lambda_i)^2 / sum lambda_i^2
      where lambda_i are eigenvalues of the prediction covariance.
      N_eff close to N = all agents contribute distinctly.
      N_eff close to 1 = agents are redundant.
    """
    predictions = []
    for agent in agents:
        agent.eval_mode()
        with torch.no_grad():
            pred = agent.predict(x)
            predictions.append(pred.mean(dim=-1))  # (batch,)

    # (N, batch)
    pred_matrix = torch.stack(predictions, dim=0)

    # Covariance across agents (N x N)
    pred_centered = pred_matrix - pred_matrix.mean(dim=1, keepdim=True)
    cov = (pred_centered @ pred_centered.T) / pred_matrix.shape[1]

    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=0)

    total = eigenvalues.sum().item()
    sum_sq = eigenvalues.pow(2).sum().item()
    n_eff = (total ** 2) / (sum_sq + 1e-8) if sum_sq > 0 else 1.0

    return {
        "covariance": cov,
        "eigenvalues": eigenvalues,
        "effective_ensemble_size": n_eff,
        "n_agents": len(agents),
    }
