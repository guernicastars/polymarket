"""Tests for LOLA and Evidence-Seeking LOLA."""

import pytest
import torch

from src.agents import MLPAgent, LinearAgent
from src.learning.lola import LOLAUpdate, MultiAgentLOLA
from src.learning.evidence_lola import EvidenceSeekingLOLA


class TestLOLAUpdate:
    def test_basic_step(self):
        """LOLA should complete a step without errors."""
        agent_i = MLPAgent(10, 1, hidden_dim=32)
        agent_j = MLPAgent(10, 1, hidden_dim=32)
        lola = LOLAUpdate(agent_i, agent_j, lr_i=1e-3, lr_j=1e-3)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        pred_i = agent_i.predict(x)
        pred_j = agent_j.predict(x)
        loss_i = torch.nn.functional.mse_loss(pred_i, y)
        loss_j = torch.nn.functional.mse_loss(pred_j, y)

        metrics = lola.step(loss_i, loss_j)
        assert "loss_i" in metrics
        assert "loss_j" in metrics
        assert "grad_norm_i" in metrics

    def test_parameters_update(self):
        """LOLA step should change agent parameters."""
        agent_i = MLPAgent(10, 1, hidden_dim=32)
        agent_j = LinearAgent(10, 1)

        params_before_i = {n: p.clone() for n, p in agent_i.model.named_parameters()}

        lola = LOLAUpdate(agent_i, agent_j, lr_i=1e-2, lr_j=1e-2)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        pred_i = agent_i.predict(x)
        pred_j = agent_j.predict(x)
        loss_i = torch.nn.functional.mse_loss(pred_i, y)
        loss_j = torch.nn.functional.mse_loss(pred_j, y)

        lola.step(loss_i, loss_j)

        changed = False
        for n, p in agent_i.model.named_parameters():
            if not torch.allclose(p, params_before_i[n]):
                changed = True
                break
        assert changed, "LOLA should update parameters"


class TestMultiAgentLOLA:
    def test_three_agents(self):
        agents = [
            LinearAgent(10, 1),
            MLPAgent(10, 1, hidden_dim=32),
            MLPAgent(10, 1, hidden_dim=64),
        ]
        multi_lola = MultiAgentLOLA(agents, lr=1e-3)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        losses = [
            torch.nn.functional.mse_loss(a.predict(x), y) for a in agents
        ]

        metrics = multi_lola.step(losses)
        assert "loss_0" in metrics
        assert "loss_1" in metrics
        assert "loss_2" in metrics


class TestEvidenceSeekingLOLA:
    def test_basic_step(self):
        agents = [
            LinearAgent(10, 1),
            MLPAgent(10, 1, hidden_dim=32),
        ]
        ev_lola = EvidenceSeekingLOLA(agents, lr=1e-3, mu_ev=0.1)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        base_losses = [
            torch.nn.functional.mse_loss(a.predict(x), y) for a in agents
        ]

        metrics = ev_lola.step(x, y, base_losses)
        assert "loss_0" in metrics
        assert "loss_1" in metrics
        assert "v_pool" in metrics
        assert "evidence_0" in metrics

    def test_evidence_tracked(self):
        agents = [
            MLPAgent(10, 1, hidden_dim=32),
            MLPAgent(10, 1, hidden_dim=32),
        ]
        ev_lola = EvidenceSeekingLOLA(agents, lr=1e-3, mu_ev=0.1)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        base_losses = [
            torch.nn.functional.mse_loss(a.predict(x), y) for a in agents
        ]

        metrics = ev_lola.step(x, y, base_losses)
        assert metrics["v_pool"] > 0
        assert metrics["evidence_0"] > 0
        assert metrics["evidence_1"] > 0
