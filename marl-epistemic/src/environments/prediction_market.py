"""
Experiment 2: Multi-Agent Prediction Market.

Synthetic prediction market with controlled evidence density.
High-evidence events have many correlated features and clean signal.
Low-evidence events have few features, noisy signal, ambiguous.

This is the key experiment for validating Keynesian evidence weighting.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class SyntheticEvent:
    """A synthetic prediction market event with controlled properties."""
    event_id: int
    features: Tensor          # (n_features,) input features
    true_probability: float   # ground truth probability in [0, 1]
    evidence_density: float   # in [0, 1], how much data supports this event
    evidence_class: str       # "high" (Pi_N) or "low" (Pi_U)
    feature_type: str         # which component generated the signal


class SyntheticEventGenerator:
    """
    Generate synthetic events with controlled evidence density.

    Events are generated from a mixture of components:
    - Linear component: signal is a linear function of features
    - Local pattern component: signal depends on local feature neighborhoods
    - Long-range component: signal depends on feature interactions across distance
    - Noisy component: weak/ambiguous signal (low evidence)

    By controlling the mixture, we create a clean separation between
    high-evidence (Pi_N) and low-evidence (Pi_U) events.
    """

    def __init__(
        self,
        n_features: int = 20,
        n_events: int = 1000,
        high_evidence_fraction: float = 0.6,
        noise_scale: float = 0.1,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.n_events = n_events
        self.high_evidence_fraction = high_evidence_fraction
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

        # Generate fixed signal parameters
        self._linear_weights = self.rng.randn(n_features) * 0.5
        self._local_kernels = self.rng.randn(3, 3) * 0.3  # 3 local kernels of size 3
        self._interaction_pairs = [
            (self.rng.randint(0, n_features), self.rng.randint(0, n_features))
            for _ in range(5)
        ]

    def generate(self) -> List[SyntheticEvent]:
        """Generate all synthetic events."""
        events = []
        n_high = int(self.n_events * self.high_evidence_fraction)

        for i in range(self.n_events):
            is_high = i < n_high
            event = self._generate_one(i, is_high)
            events.append(event)

        # Shuffle
        indices = self.rng.permutation(len(events))
        return [events[i] for i in indices]

    def _generate_one(self, event_id: int, high_evidence: bool) -> SyntheticEvent:
        """Generate a single event."""
        features = self.rng.randn(self.n_features).astype(np.float32)

        if high_evidence:
            # Choose a signal type with strong signal
            signal_type = self.rng.choice(["linear", "local", "interaction"])
            evidence_density = 0.5 + 0.5 * self.rng.random()
            noise = self.noise_scale
        else:
            # Low evidence: weak signal or ambiguous
            signal_type = self.rng.choice(["weak_linear", "noisy", "ambiguous"])
            evidence_density = 0.1 + 0.3 * self.rng.random()
            noise = self.noise_scale * 5.0

        # Compute true probability from features
        if signal_type == "linear":
            logit = np.dot(features, self._linear_weights)
        elif signal_type == "local":
            # Local pattern: convolve nearby features
            logit = 0.0
            for k_idx, kernel in enumerate(self._local_kernels):
                start = (k_idx * self.n_features // 3) % (self.n_features - 3)
                window = features[start:start + 3]
                logit += np.dot(window, kernel)
        elif signal_type == "interaction":
            # Long-range: product interactions
            logit = 0.0
            for i_idx, j_idx in self._interaction_pairs:
                logit += features[i_idx] * features[j_idx] * 0.5
        elif signal_type == "weak_linear":
            logit = np.dot(features, self._linear_weights) * 0.1
        elif signal_type == "noisy":
            logit = self.rng.randn() * 0.5
        elif signal_type == "ambiguous":
            logit = 0.0  # true probability near 0.5
        else:
            logit = 0.0

        # Add noise and convert to probability
        logit += self.rng.randn() * noise
        true_prob = 1.0 / (1.0 + np.exp(-logit))
        true_prob = np.clip(true_prob, 0.01, 0.99)

        return SyntheticEvent(
            event_id=event_id,
            features=torch.tensor(features),
            true_probability=float(true_prob),
            evidence_density=float(evidence_density),
            evidence_class="high" if high_evidence else "low",
            feature_type=signal_type,
        )

    def generate_tensors(self) -> dict:
        """Generate events and return as tensors for training."""
        events = self.generate()
        features = torch.stack([e.features for e in events])
        targets = torch.tensor([e.true_probability for e in events]).unsqueeze(-1)
        evidence_classes = [e.evidence_class for e in events]
        high_mask = torch.tensor([c == "high" for c in evidence_classes])
        low_mask = ~high_mask

        return {
            "features": features,
            "targets": targets,
            "high_evidence_mask": high_mask,
            "low_evidence_mask": low_mask,
            "events": events,
        }


class PredictionMarketEnv:
    """
    Multi-agent prediction market environment.

    Each round, agents observe event features and submit probability predictions.
    Events resolve (0 or 1) according to their true probability.
    Agents are scored by Brier score: (prediction - outcome)^2.
    """

    def __init__(
        self,
        generator: SyntheticEventGenerator,
        n_rounds: Optional[int] = None,
    ):
        self.generator = generator
        self.data = generator.generate_tensors()
        self.n_events = self.data["features"].shape[0]
        self.n_rounds = n_rounds or self.n_events
        self.current_round = 0

        # Pre-sample outcomes
        probs = self.data["targets"].squeeze(-1)
        self.outcomes = torch.bernoulli(probs)

    def reset(self):
        self.current_round = 0
        probs = self.data["targets"].squeeze(-1)
        self.outcomes = torch.bernoulli(probs)

    def get_batch(self, batch_size: int = 32) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a batch of events.

        Returns:
            features: (batch, n_features)
            targets: (batch, 1) true probabilities
            outcomes: (batch,) binary outcomes
        """
        indices = torch.randint(0, self.n_events, (batch_size,))
        return (
            self.data["features"][indices],
            self.data["targets"][indices],
            self.outcomes[indices],
        )

    def get_high_evidence_batch(self, batch_size: int = 32) -> Tuple[Tensor, Tensor, Tensor]:
        """Get batch only from high-evidence (Pi_N) events."""
        mask = self.data["high_evidence_mask"]
        indices = torch.where(mask)[0]
        sample = indices[torch.randint(0, len(indices), (batch_size,))]
        return (
            self.data["features"][sample],
            self.data["targets"][sample],
            self.outcomes[sample],
        )

    def get_low_evidence_batch(self, batch_size: int = 32) -> Tuple[Tensor, Tensor, Tensor]:
        """Get batch only from low-evidence (Pi_U) events."""
        mask = self.data["low_evidence_mask"]
        indices = torch.where(mask)[0]
        sample = indices[torch.randint(0, len(indices), (batch_size,))]
        return (
            self.data["features"][sample],
            self.data["targets"][sample],
            self.outcomes[sample],
        )

    def evaluate_predictions(
        self,
        predictions: Tensor,
        indices: Optional[Tensor] = None,
    ) -> dict:
        """
        Evaluate prediction quality with stratified metrics.

        Returns dict with overall, Pi_N, and Pi_U performance.
        """
        if indices is None:
            indices = torch.arange(min(predictions.shape[0], self.n_events))

        targets = self.data["targets"][indices].squeeze(-1)
        outcomes = self.outcomes[indices]
        preds = predictions.squeeze(-1)

        # Brier score
        brier = (preds - outcomes).pow(2).mean().item()

        # MSE against true probability
        mse = (preds - targets).pow(2).mean().item()

        # Stratified metrics
        high_mask = self.data["high_evidence_mask"][indices]
        low_mask = self.data["low_evidence_mask"][indices]

        metrics = {
            "brier_score": brier,
            "mse": mse,
        }

        if high_mask.any():
            metrics["brier_high"] = (preds[high_mask] - outcomes[high_mask]).pow(2).mean().item()
            metrics["mse_high"] = (preds[high_mask] - targets[high_mask]).pow(2).mean().item()

        if low_mask.any():
            metrics["brier_low"] = (preds[low_mask] - outcomes[low_mask]).pow(2).mean().item()
            metrics["mse_low"] = (preds[low_mask] - targets[low_mask]).pow(2).mean().item()

        return metrics
