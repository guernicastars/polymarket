"""Standard REINFORCE with baseline for policy gradient training."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.agents.base import BaseAgent


class REINFORCE:
    """
    REINFORCE (Williams 1992) with learned baseline.

    For each agent, computes policy gradient:
        grad_theta J = E[ sum_t (R_t - b_t) * grad_theta log pi(a_t | s_t) ]

    where b_t is a baseline (running average of returns) to reduce variance.
    """

    def __init__(
        self,
        agent: BaseAgent,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.agent.setup_optimizer(lr=lr)
        self._baseline = 0.0
        self._baseline_momentum = 0.99

    def compute_returns(self, rewards: List[float]) -> Tensor:
        """Compute discounted returns for an episode."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.agent.device)
        return returns

    def update(
        self,
        log_probs: List[Tensor],
        rewards: List[float],
        entropies: Optional[List[Tensor]] = None,
    ) -> dict:
        """
        Perform a REINFORCE update for one episode.

        Args:
            log_probs: log pi(a_t | s_t) for each timestep.
            rewards: scalar reward at each timestep.
            entropies: (optional) entropy of the policy at each timestep.

        Returns:
            Dict with loss, mean_return, baseline.
        """
        returns = self.compute_returns(rewards)

        # Update baseline
        mean_return = returns.mean().item()
        self._baseline = (
            self._baseline_momentum * self._baseline
            + (1 - self._baseline_momentum) * mean_return
        )

        # Policy gradient loss
        advantages = returns - self._baseline
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = torch.tensor(0.0, device=self.agent.device)
        for log_p, adv in zip(log_probs, advantages):
            policy_loss = policy_loss - log_p * adv.detach()
        policy_loss = policy_loss / len(log_probs)

        # Entropy bonus
        entropy_loss = torch.tensor(0.0, device=self.agent.device)
        if entropies is not None and self.entropy_coef > 0:
            entropy_loss = -self.entropy_coef * torch.stack(entropies).mean()

        total_loss = policy_loss + entropy_loss

        # Gradient step
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.parameters, self.max_grad_norm
        )
        self.agent.optimizer.step()

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_return": mean_return,
            "baseline": self._baseline,
        }
