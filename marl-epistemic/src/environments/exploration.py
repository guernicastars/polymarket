"""
Experiment 3: Multi-Agent Contextual Bandits.

Contextual bandit with d=20 dimensional context, K=10 arms.
True reward function is a mixture of:
- Linear component: captured by LinearAgent
- Local pattern component: captured by CNNAgent
- Long-range dependency component: captured by AttentionAgent
- Interaction terms: partially captured by MLPAgent

This creates provably different blind spots for each architecture.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class ContextualBanditEnv:
    """
    Contextual bandit where the optimal arm depends on multi-component
    reward function that different architectures capture differently.

    True reward for arm k given context x:
        r(x, k) = r_linear(x, k) + r_local(x, k) + r_longrange(x, k) + r_interaction(x, k)

    Each component is designed so a specific architecture excels at it:
    - Linear: w_k^T x
    - Local: conv patterns in adjacent features
    - Long-range: attention over distant feature pairs
    - Interaction: product terms (partially captured by MLP)
    """

    def __init__(
        self,
        context_dim: int = 20,
        n_arms: int = 10,
        component_weights: Optional[Dict[str, float]] = None,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        if component_weights is None:
            component_weights = {
                "linear": 0.3,
                "local": 0.25,
                "longrange": 0.25,
                "interaction": 0.2,
            }
        self.component_weights = component_weights

        # Initialize reward function parameters
        self._init_reward_params()

        self.step_count = 0
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        self.component_regret: Dict[str, List[float]] = {
            k: [] for k in component_weights
        }

    def _init_reward_params(self):
        """Initialize parameters for each reward component."""
        d = self.context_dim
        K = self.n_arms

        # Linear: per-arm weight vectors
        self.linear_weights = self.rng.randn(K, d).astype(np.float32) * 0.3

        # Local: per-arm convolution kernels (size 3, stride 1)
        self.local_kernels = self.rng.randn(K, 5, 3).astype(np.float32) * 0.2

        # Long-range: per-arm pair interactions between distant features
        n_pairs = 4
        self.lr_indices = [
            (self.rng.randint(0, d // 2), self.rng.randint(d // 2, d))
            for _ in range(n_pairs)
        ]
        self.lr_weights = self.rng.randn(K, n_pairs).astype(np.float32) * 0.4

        # Interaction: per-arm nonlinear feature interactions
        self.interaction_pairs = [
            (self.rng.randint(0, d), self.rng.randint(0, d))
            for _ in range(6)
        ]
        self.interaction_weights = self.rng.randn(K, 6).astype(np.float32) * 0.3

    def sample_context(self, batch_size: int = 1) -> Tensor:
        """Sample random context vectors."""
        return torch.randn(batch_size, self.context_dim)

    def compute_reward_components(
        self, context: np.ndarray, arm: int
    ) -> Dict[str, float]:
        """Compute each reward component separately (for analysis)."""
        d = self.context_dim

        # Linear
        r_linear = float(np.dot(context, self.linear_weights[arm]))

        # Local: apply kernels to adjacent features
        r_local = 0.0
        for k_idx in range(self.local_kernels.shape[1]):
            start = (k_idx * d // 5) % (d - 3)
            window = context[start:start + 3]
            r_local += float(np.dot(window, self.local_kernels[arm, k_idx]))

        # Long-range: product of distant features
        r_longrange = 0.0
        for p_idx, (i, j) in enumerate(self.lr_indices):
            r_longrange += float(
                context[i] * context[j] * self.lr_weights[arm, p_idx]
            )

        # Interaction: nonlinear combinations
        r_interaction = 0.0
        for p_idx, (i, j) in enumerate(self.interaction_pairs):
            r_interaction += float(
                np.tanh(context[i] * context[j]) * self.interaction_weights[arm, p_idx]
            )

        return {
            "linear": r_linear,
            "local": r_local,
            "longrange": r_longrange,
            "interaction": r_interaction,
        }

    def true_reward(self, context: np.ndarray, arm: int) -> float:
        """Compute the true (noiseless) reward for an arm given context."""
        components = self.compute_reward_components(context, arm)
        return sum(
            self.component_weights[k] * v for k, v in components.items()
        )

    def optimal_arm(self, context: np.ndarray) -> Tuple[int, float]:
        """Find the optimal arm and its reward."""
        rewards = [self.true_reward(context, a) for a in range(self.n_arms)]
        best = int(np.argmax(rewards))
        return best, rewards[best]

    def step(
        self, context: Tensor, arm: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Take an action and receive noisy reward.

        Args:
            context: (context_dim,) context vector.
            arm: chosen arm index.

        Returns:
            reward: scalar noisy reward.
            info: dict with component breakdown, regret, optimal arm.
        """
        ctx = context.numpy() if isinstance(context, Tensor) else context

        # Compute reward
        components = self.compute_reward_components(ctx, arm)
        true_r = sum(self.component_weights[k] * v for k, v in components.items())
        noisy_r = true_r + self.rng.randn() * self.noise_std

        # Compute regret
        opt_arm, opt_reward = self.optimal_arm(ctx)
        regret = opt_reward - true_r
        self.cumulative_regret += max(0, regret)
        self.regret_history.append(self.cumulative_regret)

        # Component regret: how much regret from each component?
        opt_components = self.compute_reward_components(ctx, opt_arm)
        for k in self.component_weights:
            comp_regret = self.component_weights[k] * (opt_components[k] - components[k])
            self.component_regret[k].append(
                (self.component_regret[k][-1] if self.component_regret[k] else 0.0)
                + max(0, comp_regret)
            )

        self.step_count += 1

        return noisy_r, {
            "components": components,
            "true_reward": true_r,
            "optimal_arm": opt_arm,
            "optimal_reward": opt_reward,
            "regret": max(0, regret),
            "cumulative_regret": self.cumulative_regret,
        }

    def batch_true_rewards(self, contexts: Tensor) -> Tensor:
        """
        Compute true reward for all arms given batch of contexts.
        Returns: (batch, n_arms) tensor.
        """
        batch_size = contexts.shape[0]
        rewards = torch.zeros(batch_size, self.n_arms)
        for b in range(batch_size):
            ctx = contexts[b].numpy()
            for a in range(self.n_arms):
                rewards[b, a] = self.true_reward(ctx, a)
        return rewards

    def reset_metrics(self):
        """Reset regret tracking."""
        self.step_count = 0
        self.cumulative_regret = 0.0
        self.regret_history = []
        self.component_regret = {k: [] for k in self.component_weights}
