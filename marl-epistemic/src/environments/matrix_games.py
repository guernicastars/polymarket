"""
Experiment 1: Matrix Games with Diverse Observations.

Iterated Prisoner's Dilemma (IPD) and Stag Hunt where agents observe
DIFFERENT FEATURES of the game state. This tests whether architectural
diversity helps find cooperative equilibria.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# Payoff matrices: (row_action, col_action) -> (row_payoff, col_payoff)
IPD_PAYOFFS = {
    (0, 0): (3.0, 3.0),  # Cooperate, Cooperate -> mutual cooperation
    (0, 1): (0.0, 5.0),  # Cooperate, Defect -> sucker's payoff
    (1, 0): (5.0, 0.0),  # Defect, Cooperate -> temptation
    (1, 1): (1.0, 1.0),  # Defect, Defect -> mutual defection
}

STAG_HUNT_PAYOFFS = {
    (0, 0): (4.0, 4.0),  # Stag, Stag -> best outcome
    (0, 1): (0.0, 3.0),  # Stag, Hare -> nothing vs safe
    (1, 0): (3.0, 0.0),  # Hare, Stag -> safe vs nothing
    (1, 1): (2.0, 2.0),  # Hare, Hare -> safe but suboptimal
}

COORDINATION_PAYOFFS = {
    (0, 0): (2.0, 2.0),  # Match -> good
    (0, 1): (0.0, 0.0),  # Mismatch -> bad
    (1, 0): (0.0, 0.0),
    (1, 1): (2.0, 2.0),
}


class MatrixGame:
    """
    N-player iterated matrix game with diverse observation functions.

    Each agent observes a DIFFERENT FEATURE of the game state:
    - Agent type "temporal": last K actions of each player
    - Agent type "statistical": running cooperation rates
    - Agent type "relative": relative cumulative payoffs

    This creates genuinely different observation spaces, meaning
    different hypothesis classes are needed to process them.
    """

    def __init__(
        self,
        payoff_matrix: Dict[Tuple[int, int], Tuple[float, ...]],
        n_agents: int = 2,
        history_length: int = 5,
        observation_types: Optional[List[str]] = None,
    ):
        self.payoff_matrix = payoff_matrix
        self.n_agents = n_agents
        self.history_length = history_length

        if observation_types is None:
            observation_types = ["temporal", "statistical", "relative"][:n_agents]
        self.observation_types = observation_types

        self.reset()

    def reset(self) -> List[Tensor]:
        """Reset game state and return initial observations."""
        self.action_history: List[List[int]] = [[] for _ in range(self.n_agents)]
        self.payoff_history: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.step_count = 0

        return [self._get_observation(i) for i in range(self.n_agents)]

    def _get_observation(self, agent_idx: int) -> Tensor:
        """
        Get observation for agent based on its observation type.

        Different observation types create fundamentally different input spaces:
        - temporal: sequential pattern -> favors CNN/Attention
        - statistical: summary statistics -> favors Linear/MLP
        - relative: relational information -> favors Attention
        """
        obs_type = self.observation_types[agent_idx]

        if obs_type == "temporal":
            return self._temporal_observation(agent_idx)
        elif obs_type == "statistical":
            return self._statistical_observation(agent_idx)
        elif obs_type == "relative":
            return self._relative_observation(agent_idx)
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

    def _temporal_observation(self, agent_idx: int) -> Tensor:
        """Last K actions of all players. Shape: (n_agents * history_length,)."""
        obs = []
        for i in range(self.n_agents):
            history = self.action_history[i][-self.history_length:]
            padded = [0.0] * (self.history_length - len(history)) + [float(a) for a in history]
            obs.extend(padded)
        return torch.tensor(obs, dtype=torch.float32)

    def _statistical_observation(self, agent_idx: int) -> Tensor:
        """Running cooperation rates + global stats. Shape: (n_agents * 3,)."""
        obs = []
        for i in range(self.n_agents):
            history = self.action_history[i]
            if len(history) == 0:
                obs.extend([0.5, 0.0, 0.0])
            else:
                coop_rate = 1.0 - sum(history) / len(history)  # 0=cooperate, 1=defect
                recent = history[-min(5, len(history)):]
                recent_coop = 1.0 - sum(recent) / len(recent)
                trend = recent_coop - coop_rate
                obs.extend([coop_rate, recent_coop, trend])
        return torch.tensor(obs, dtype=torch.float32)

    def _relative_observation(self, agent_idx: int) -> Tensor:
        """Relative payoffs and standings. Shape: (n_agents * 2 + 1,)."""
        obs = []
        my_total = sum(self.payoff_history[agent_idx]) if self.payoff_history[agent_idx] else 0.0
        for i in range(self.n_agents):
            their_total = sum(self.payoff_history[i]) if self.payoff_history[i] else 0.0
            obs.append(their_total / max(self.step_count, 1))
            obs.append(their_total - my_total)
        obs.append(float(self.step_count) / 100.0)  # time feature
        return torch.tensor(obs, dtype=torch.float32)

    def observation_dims(self) -> List[int]:
        """Return observation dimension for each agent."""
        dims = []
        for obs_type in self.observation_types:
            if obs_type == "temporal":
                dims.append(self.n_agents * self.history_length)
            elif obs_type == "statistical":
                dims.append(self.n_agents * 3)
            elif obs_type == "relative":
                dims.append(self.n_agents * 2 + 1)
        return dims

    def step(self, actions: List[int]) -> Tuple[List[Tensor], List[float], bool]:
        """
        Execute one step of the game.

        Args:
            actions: List of actions, one per agent. 0 = cooperate, 1 = defect.

        Returns:
            observations: List of observation tensors per agent.
            rewards: List of scalar rewards per agent.
            done: Whether the game is over.
        """
        # For 2-player games, use the payoff matrix directly
        # For N-player, generalize via pairwise interactions
        rewards = [0.0] * self.n_agents
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                key = (actions[i], actions[j])
                if key in self.payoff_matrix:
                    rewards[i] += self.payoff_matrix[key][0]
            # Normalize by number of opponents
            rewards[i] /= max(self.n_agents - 1, 1)

        # Update histories
        for i in range(self.n_agents):
            self.action_history[i].append(actions[i])
            self.payoff_history[i].append(rewards[i])
        self.step_count += 1

        observations = [self._get_observation(i) for i in range(self.n_agents)]
        return observations, rewards, False

    def cooperation_rate(self, agent_idx: Optional[int] = None) -> float:
        """Fraction of cooperative actions."""
        if agent_idx is not None:
            history = self.action_history[agent_idx]
            if not history:
                return 0.0
            return 1.0 - sum(history) / len(history)
        # Overall cooperation rate
        all_actions = [a for h in self.action_history for a in h]
        if not all_actions:
            return 0.0
        return 1.0 - sum(all_actions) / len(all_actions)


class IteratedPrisonersDilemma(MatrixGame):
    """Iterated Prisoner's Dilemma with diverse observations."""

    def __init__(self, n_agents: int = 2, **kwargs):
        super().__init__(
            payoff_matrix=IPD_PAYOFFS,
            n_agents=n_agents,
            **kwargs,
        )


class StagHunt(MatrixGame):
    """Stag Hunt with diverse observations."""

    def __init__(self, n_agents: int = 2, **kwargs):
        super().__init__(
            payoff_matrix=STAG_HUNT_PAYOFFS,
            n_agents=n_agents,
            **kwargs,
        )


class PolicyNetwork(torch.nn.Module):
    """Simple policy network for matrix game agents."""

    def __init__(self, input_dim: int, n_actions: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return action logits."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def get_action(self, x: Tensor) -> Tuple[int, Tensor, Tensor]:
        """Sample action, return (action, log_prob, entropy)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob.squeeze(), entropy.squeeze()
