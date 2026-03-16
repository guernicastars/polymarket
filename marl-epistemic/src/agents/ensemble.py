"""Ensemble methods: simple average, Keynesian-weighted, and LOLA-trained."""

from typing import List, Optional

import torch
from torch import Tensor

from .base import BaseAgent


class SimpleEnsemble:
    """Simple average ensemble: y_hat = mean(y_hat_i)."""

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def predict(self, x: Tensor) -> Tensor:
        predictions = torch.stack([a.predict(x) for a in self.agents], dim=0)
        return predictions.mean(dim=0)

    def predict_with_individual(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        individual = [a.predict(x) for a in self.agents]
        ensemble = torch.stack(individual, dim=0).mean(dim=0)
        return ensemble, individual


class AccuracyWeightedEnsemble:
    """Accuracy-weighted ensemble: y_hat = sum(w_i * y_hat_i) / sum(w_i)."""

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.weights: Optional[Tensor] = None

    def fit_weights(self, x_val: Tensor, y_val: Tensor):
        """Compute weights proportional to 1/MSE on validation set."""
        mse_scores = []
        for agent in self.agents:
            agent.eval_mode()
            with torch.no_grad():
                pred = agent.predict(x_val)
                mse = (pred - y_val).pow(2).mean().item()
            mse_scores.append(mse)

        # w_i proportional to 1/MSE_i
        inv_mse = torch.tensor([1.0 / (m + 1e-8) for m in mse_scores])
        self.weights = inv_mse / inv_mse.sum()

    def predict(self, x: Tensor) -> Tensor:
        if self.weights is None:
            raise RuntimeError("Call fit_weights() first.")
        predictions = torch.stack([a.predict(x) for a in self.agents], dim=0)
        w = self.weights.to(x.device).view(-1, 1, 1)
        return (predictions * w).sum(dim=0)


class KeynesianEnsemble:
    """
    Keynesian-weighted ensemble: y_hat = sum(V_i * y_hat_i) / sum(V_i).

    Weights are per-sample evidence weights V_i(x), not global accuracy.
    This means the ensemble upweights agents that have MORE EVIDENCE
    for the specific input x, not just agents that are globally better.

    Key advantage over accuracy-weighted: on low-evidence inputs (Pi_U),
    the ensemble correctly downweights all agents and can signal low
    confidence, rather than trusting the globally best agent.
    """

    def __init__(self, agents: List[BaseAgent], evidence_method: str = "mc_dropout"):
        self.agents = agents
        self.evidence_method = evidence_method

    def predict(self, x: Tensor) -> Tensor:
        predictions = []
        evidences = []
        for agent in self.agents:
            pred = agent.predict(x)
            v = agent.weight_of_evidence(x, method=self.evidence_method)
            predictions.append(pred)
            evidences.append(v)

        # predictions: list of (batch, output_dim)
        # evidences: list of (batch,)
        preds = torch.stack(predictions, dim=0)  # (N, batch, output_dim)
        evs = torch.stack(evidences, dim=0)  # (N, batch)

        # Normalize evidence weights per sample
        w = evs / (evs.sum(dim=0, keepdim=True) + 1e-8)  # (N, batch)
        w = w.unsqueeze(-1)  # (N, batch, 1)
        return (preds * w).sum(dim=0)

    def predict_with_evidence(self, x: Tensor) -> tuple[Tensor, Tensor, List[Tensor]]:
        """Return ensemble prediction, pooled evidence, and individual evidences."""
        predictions = []
        evidences = []
        for agent in self.agents:
            pred = agent.predict(x)
            v = agent.weight_of_evidence(x, method=self.evidence_method)
            predictions.append(pred)
            evidences.append(v)

        preds = torch.stack(predictions, dim=0)
        evs = torch.stack(evidences, dim=0)

        w = evs / (evs.sum(dim=0, keepdim=True) + 1e-8)
        w_expanded = w.unsqueeze(-1)
        ensemble_pred = (preds * w_expanded).sum(dim=0)

        # Pooled evidence: sum of individual evidences
        v_pool = evs.sum(dim=0)

        return ensemble_pred, v_pool, evidences


class LOLAEnsemble:
    """
    Ensemble of LOLA-trained agents.

    Agents are trained with opponent-shaping (and optionally evidence-seeking),
    then ensembled using Keynesian weights at inference time.
    This is a container -- training is handled by the LOLA/EvidenceLOLA trainers.
    """

    def __init__(self, agents: List[BaseAgent], evidence_method: str = "mc_dropout"):
        self.agents = agents
        self._keynesian = KeynesianEnsemble(agents, evidence_method)

    def predict(self, x: Tensor) -> Tensor:
        return self._keynesian.predict(x)

    def predict_with_evidence(self, x: Tensor) -> tuple[Tensor, Tensor, List[Tensor]]:
        return self._keynesian.predict_with_evidence(x)
