"""
Weight of evidence V_i computation methods.

V_i is NOT prediction confidence (softmax entropy).
V_i measures how much DATA supports the prediction at x.

Two methods:
- MC dropout: inverse predictive variance under dropout noise
- Kernel: effective sample size in neighbourhood of x
"""

from typing import List

import torch
from torch import Tensor

from src.agents.base import BaseAgent


def weight_of_evidence_mc_dropout(
    agent: BaseAgent,
    x: Tensor,
    n_samples: int = 20,
) -> Tensor:
    """
    MC dropout estimate of weight of evidence.

    V_i(x) = 1 / Var[f_hat(x; theta_i + noise)]

    Run n_samples forward passes with dropout enabled, compute variance.
    Low variance = high evidence (model is confident because it has
    seen similar data, not just because of architecture bias).

    Returns: (batch,) tensor of evidence weights.
    """
    agent.model.train()  # enable dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = agent.predict(x)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # (S, batch, output_dim)
    variance = predictions.var(dim=0).mean(dim=-1)  # (batch,)
    evidence = 1.0 / (variance + 1e-8)

    agent.model.eval()
    return evidence


def weight_of_evidence_kernel(
    agent: BaseAgent,
    x: Tensor,
    bandwidth: float = 1.0,
) -> Tensor:
    """
    Kernel density estimate of weight of evidence.

    V_i(x) = sum_j K(x, x_j) for training points x_j

    Uses Gaussian kernel. More training data near x = higher evidence.
    This method requires agent.store_training_points() to have been called.

    Returns: (batch,) tensor of evidence weights.
    """
    return agent.weight_of_evidence(x, method="kernel")


def pooled_evidence(
    agents: List[BaseAgent],
    x: Tensor,
    method: str = "mc_dropout",
) -> Tensor:
    """
    Compute pooled evidence V_pool = sum_i V_i(x).

    Key claim: V_pool > max_i V_i when agents have complementary evidence.
    This means the collective has MORE information than any individual,
    analogous to combining measurements from different instruments.

    Returns: (batch,) tensor.
    """
    evidences = [a.weight_of_evidence(x, method=method) for a in agents]
    return sum(evidences)


def evidence_complementarity(
    agents: List[BaseAgent],
    x: Tensor,
    method: str = "mc_dropout",
) -> dict:
    """
    Measure how much agents' evidence complements each other.

    Returns:
        - v_pool: pooled evidence
        - v_max: max individual evidence
        - complementarity_ratio: v_pool / v_max (should be > 1)
        - per_agent: individual V_i values
    """
    evidences = [a.weight_of_evidence(x, method=method) for a in agents]
    stacked = torch.stack(evidences, dim=0)  # (N, batch)

    v_pool = stacked.sum(dim=0)  # (batch,)
    v_max = stacked.max(dim=0).values  # (batch,)

    ratio = (v_pool / (v_max + 1e-8)).mean().item()

    return {
        "v_pool": v_pool,
        "v_max": v_max,
        "complementarity_ratio": ratio,
        "per_agent": {
            agents[i].hypothesis_class_name: evidences[i].mean().item()
            for i in range(len(agents))
        },
    }
