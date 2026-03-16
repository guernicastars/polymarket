"""
Evidence-Seeking LOLA -- THE NOVEL CONTRIBUTION.

Standard LOLA shapes opponent's learning to reduce agent i's loss.
Evidence-seeking LOLA shapes opponent's learning to INCREASE COLLECTIVE EVIDENCE:

    R_i^evidence = R_i + mu_ev * (1 / V_pool)

where V_pool = sum_k alpha_k * V_k

Agent i wants to shape agent j's learning so that V_j increases
in regions where V_i is low. This produces COMPLEMENTARY EVIDENCE,
not just complementary predictions.

The gradient:
    grad_theta_i R_i^evidence = grad_theta_i R_i
        + standard LOLA opponent-shaping term
        + mu_ev * d(V_pool^{-1})/dtheta_j * dtheta_j'/dtheta_i
                   ^--- new term: shape j to gather evidence where i lacks it
"""

from typing import List, Optional

import torch
from torch import Tensor

from src.agents.base import BaseAgent


class EvidenceSeekingLOLA:
    """
    LOLA with Keynesian evidence-seeking objective.

    Each agent's loss becomes:
        L_i^ev = L_i + mu_ev / V_pool(x)

    And the LOLA update shapes opponents to increase V_pool,
    particularly in regions where the agent's own V_i is low.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        lr: float = 1e-3,
        lola_weight: float = 1.0,
        mu_ev: float = 0.1,
        evidence_method: str = "mc_dropout",
        max_grad_norm: float = 5.0,
    ):
        self.agents = agents
        self.lr = lr
        self.lola_weight = lola_weight
        self.mu_ev = mu_ev
        self.evidence_method = evidence_method
        self.max_grad_norm = max_grad_norm

        for agent in agents:
            agent.setup_optimizer(lr=lr)

    def compute_evidence_augmented_loss(
        self,
        agent_idx: int,
        x: Tensor,
        y_true: Tensor,
        base_losses: List[Tensor],
    ) -> Tensor:
        """
        Compute evidence-augmented loss for agent i:
            L_i^ev = L_i + mu_ev / V_pool(x)

        where V_pool = sum_j V_j(x)
        """
        base_loss = base_losses[agent_idx]

        # Compute pooled evidence (with gradients flowing through all agents)
        evidences = []
        for agent in self.agents:
            agent.train_mode()
            v = agent.weight_of_evidence(x, method=self.evidence_method)
            evidences.append(v)

        v_pool = sum(evidences)  # (batch,)
        evidence_penalty = self.mu_ev / (v_pool + 1e-8)

        return base_loss + evidence_penalty.mean()

    def step(
        self,
        x: Tensor,
        y_true: Tensor,
        base_losses: List[Tensor],
    ) -> dict:
        """
        One evidence-seeking LOLA step for all agents.

        Args:
            x: Input batch. Shape: (batch, input_dim).
            y_true: Target batch. Shape: (batch, output_dim).
            base_losses: List of differentiable prediction losses per agent.

        Returns:
            Dict with per-agent metrics including evidence values.
        """
        n = len(self.agents)
        all_params = [list(a.parameters) for a in self.agents]

        # Compute evidence-augmented losses
        ev_losses = []
        for i in range(n):
            ev_loss = self.compute_evidence_augmented_loss(
                i, x, y_true, base_losses
            )
            ev_losses.append(ev_loss)

        # Compute LOLA gradients with evidence-augmented objectives
        all_grads = []
        for i in range(n):
            # Standard gradient of evidence-augmented loss
            grad_i_raw = torch.autograd.grad(
                ev_losses[i], all_params[i], create_graph=True, retain_graph=True,
                allow_unused=True,
            )
            grad_i = [g if g is not None else torch.zeros_like(p)
                       for g, p in zip(grad_i_raw, all_params[i])]

            # LOLA corrections from each opponent
            for j in range(n):
                if i == j:
                    continue

                # Cross gradient: d L_i^ev / d theta_j
                grad_j_Li = torch.autograd.grad(
                    ev_losses[i], all_params[j], create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                grad_j_Li = [g if g is not None else torch.zeros_like(p)
                              for g, p in zip(grad_j_Li, all_params[j])]

                # j's evidence-augmented gradient
                grad_j_Lj = torch.autograd.grad(
                    ev_losses[j], all_params[j], create_graph=True, retain_graph=True,
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

        # Apply gradients
        metrics = {}
        for i, (agent, grads) in enumerate(zip(self.agents, all_grads)):
            agent.optimizer.zero_grad()
            for param, grad in zip(all_params[i], grads):
                param.grad = grad.detach()
            gn = torch.nn.utils.clip_grad_norm_(
                all_params[i], self.max_grad_norm
            )
            agent.optimizer.step()

            metrics[f"loss_{i}"] = ev_losses[i].item()
            metrics[f"grad_norm_{i}"] = gn.item() if isinstance(gn, Tensor) else gn

        # Log evidence values
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                v = agent.weight_of_evidence(x, method=self.evidence_method)
                metrics[f"evidence_{i}"] = v.mean().item()
            # Pooled evidence
            total_ev = sum(
                a.weight_of_evidence(x, method=self.evidence_method).mean().item()
                for a in self.agents
            )
            metrics["v_pool"] = total_ev

        return metrics
