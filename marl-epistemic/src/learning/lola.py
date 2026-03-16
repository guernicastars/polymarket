"""
LOLA: Learning with Opponent-Learning Awareness (Foerster et al. 2018).

Key equation:
    grad_theta_i^LOLA R_i = grad_theta_i R_i
        + (dR_i/dtheta_j) * (dtheta_j'/dtheta_i)

where theta_j' = theta_j - eta_j * grad_theta_j R_j(theta_i, theta_j)

This requires differentiating through agent j's gradient step,
using torch.autograd with create_graph=True for second-order gradients.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.agents.base import BaseAgent


class LOLAUpdate:
    """
    LOLA opponent-shaping update for two-player differentiable games.

    In a two-player game, each agent shapes the other's learning:
    agent i accounts for how its parameters affect agent j's gradient step,
    producing a second-order correction to the naive gradient.
    """

    def __init__(
        self,
        agent_i: BaseAgent,
        agent_j: BaseAgent,
        lr_i: float = 1e-3,
        lr_j: float = 1e-3,
        lola_weight: float = 1.0,
        max_grad_norm: float = 5.0,
    ):
        self.agent_i = agent_i
        self.agent_j = agent_j
        self.lr_i = lr_i
        self.lr_j = lr_j
        self.lola_weight = lola_weight
        self.max_grad_norm = max_grad_norm

        self.agent_i.setup_optimizer(lr=lr_i)
        self.agent_j.setup_optimizer(lr=lr_j)

    def compute_lola_gradient(
        self,
        loss_i: Tensor,
        loss_j: Tensor,
        params_i: List[Tensor],
        params_j: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute LOLA gradient for agent i accounting for agent j's learning.

        grad_LOLA_i = grad_i L_i + lola_weight * (d L_i / d theta_j) * (d theta_j' / d theta_i)

        where theta_j' = theta_j - lr_j * grad_j L_j
        """
        # Standard gradient: grad_i L_i
        grad_i_Li = torch.autograd.grad(
            loss_i, params_i, create_graph=True, retain_graph=True,
            allow_unused=True,
        )
        grad_i_Li = [g if g is not None else torch.zeros_like(p)
                      for g, p in zip(grad_i_Li, params_i)]

        # Cross gradient: d L_i / d theta_j
        grad_j_Li = torch.autograd.grad(
            loss_i, params_j, create_graph=True, retain_graph=True,
            allow_unused=True,
        )
        grad_j_Li = [g if g is not None else torch.zeros_like(p)
                      for g, p in zip(grad_j_Li, params_j)]

        # Agent j's gradient: grad_j L_j
        grad_j_Lj = torch.autograd.grad(
            loss_j, params_j, create_graph=True, retain_graph=True,
            allow_unused=True,
        )
        grad_j_Lj = [g if g is not None else torch.zeros_like(p)
                      for g, p in zip(grad_j_Lj, params_j)]

        # The LOLA correction: how j's update (based on j's gradient) affects i's loss
        # theta_j' = theta_j - lr_j * grad_j_Lj
        # d theta_j' / d theta_i = -lr_j * d(grad_j_Lj) / d(theta_i)
        # The full LOLA term: (grad_j_Li)^T * (-lr_j) * d(grad_j_Lj)/d(theta_i)

        # We compute this via vector-Jacobian product:
        # d/d(theta_i) [ sum_k grad_j_Li[k] * grad_j_Lj[k] ] * (-lr_j)
        inner_product = sum(
            (g_li * g_lj).sum()
            for g_li, g_lj in zip(grad_j_Li, grad_j_Lj)
        )

        lola_correction = torch.autograd.grad(
            inner_product, params_i, retain_graph=True,
            allow_unused=True,
        )
        lola_correction = [g if g is not None else torch.zeros_like(p)
                           for g, p in zip(lola_correction, params_i)]

        # Combine: standard gradient + LOLA correction
        lola_grad = []
        for g_std, g_lola in zip(grad_i_Li, lola_correction):
            combined = g_std + self.lola_weight * (-self.lr_j) * g_lola
            lola_grad.append(combined)

        return lola_grad

    def step(
        self,
        loss_i: Tensor,
        loss_j: Tensor,
    ) -> dict:
        """
        Perform one LOLA update step for both agents.

        Args:
            loss_i: Differentiable loss for agent i (scalar).
            loss_j: Differentiable loss for agent j (scalar).

        Returns:
            Dict with loss values and gradient norms.
        """
        params_i = list(self.agent_i.parameters)
        params_j = list(self.agent_j.parameters)

        # LOLA gradient for agent i (accounting for j's learning)
        lola_grad_i = self.compute_lola_gradient(
            loss_i, loss_j, params_i, params_j
        )

        # LOLA gradient for agent j (accounting for i's learning)
        lola_grad_j = self.compute_lola_gradient(
            loss_j, loss_i, params_j, params_i
        )

        # Apply gradients for agent i
        self.agent_i.optimizer.zero_grad()
        for param, grad in zip(params_i, lola_grad_i):
            param.grad = grad.detach()
        grad_norm_i = torch.nn.utils.clip_grad_norm_(
            params_i, self.max_grad_norm
        )
        self.agent_i.optimizer.step()

        # Apply gradients for agent j
        self.agent_j.optimizer.zero_grad()
        for param, grad in zip(params_j, lola_grad_j):
            param.grad = grad.detach()
        grad_norm_j = torch.nn.utils.clip_grad_norm_(
            params_j, self.max_grad_norm
        )
        self.agent_j.optimizer.step()

        return {
            "loss_i": loss_i.item(),
            "loss_j": loss_j.item(),
            "grad_norm_i": grad_norm_i.item() if isinstance(grad_norm_i, Tensor) else grad_norm_i,
            "grad_norm_j": grad_norm_j.item() if isinstance(grad_norm_j, Tensor) else grad_norm_j,
        }


class MultiAgentLOLA:
    """
    LOLA for N > 2 agents. Each agent shapes all others' learning.

    For agent i, the gradient includes LOLA corrections w.r.t. every other agent j.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        lr: float = 1e-3,
        lola_weight: float = 1.0,
        max_grad_norm: float = 5.0,
    ):
        self.agents = agents
        self.lr = lr
        self.lola_weight = lola_weight
        self.max_grad_norm = max_grad_norm

        for agent in agents:
            agent.setup_optimizer(lr=lr)

    def step(self, losses: List[Tensor]) -> dict:
        """
        One LOLA step for all agents.

        Args:
            losses: List of differentiable losses, one per agent.

        Returns:
            Dict with per-agent metrics.
        """
        n = len(self.agents)
        all_params = [list(a.parameters) for a in self.agents]
        all_grads = []

        for i in range(n):
            # Standard gradient for agent i
            grad_i_raw = torch.autograd.grad(
                losses[i], all_params[i], create_graph=True, retain_graph=True,
                allow_unused=True,
            )
            grad_i = [g if g is not None else torch.zeros_like(p)
                       for g, p in zip(grad_i_raw, all_params[i])]

            # LOLA corrections from each opponent j
            for j in range(n):
                if i == j:
                    continue

                # Cross gradient: d L_i / d theta_j
                grad_j_Li = torch.autograd.grad(
                    losses[i], all_params[j], create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                grad_j_Li = [g if g is not None else torch.zeros_like(p)
                              for g, p in zip(grad_j_Li, all_params[j])]

                # j's own gradient: grad_j L_j
                grad_j_Lj = torch.autograd.grad(
                    losses[j], all_params[j], create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                grad_j_Lj = [g if g is not None else torch.zeros_like(p)
                              for g, p in zip(grad_j_Lj, all_params[j])]

                inner = sum(
                    (a * b).sum() for a, b in zip(grad_j_Li, grad_j_Lj)
                )

                correction = torch.autograd.grad(
                    inner, all_params[i], retain_graph=True,
                    allow_unused=True,
                )
                correction = [g if g is not None else torch.zeros_like(p)
                              for g, p in zip(correction, all_params[i])]

                for k in range(len(grad_i)):
                    grad_i[k] = grad_i[k] + self.lola_weight * (-self.lr) * correction[k]

            all_grads.append(grad_i)

        # Apply all gradients
        metrics = {}
        for i, (agent, grads) in enumerate(zip(self.agents, all_grads)):
            agent.optimizer.zero_grad()
            for param, grad in zip(all_params[i], grads):
                param.grad = grad.detach()
            gn = torch.nn.utils.clip_grad_norm_(
                all_params[i], self.max_grad_norm
            )
            agent.optimizer.step()
            metrics[f"loss_{i}"] = losses[i].item()
            metrics[f"grad_norm_{i}"] = gn.item() if isinstance(gn, Tensor) else gn

        return metrics
