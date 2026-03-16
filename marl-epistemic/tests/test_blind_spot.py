"""Tests for blind spot metrics."""

import pytest
import torch

from src.agents import LinearAgent, MLPAgent
from src.metrics.blind_spot import (
    compute_blind_spot,
    blind_spot_overlap,
    collective_blind_spot,
    complementarity_score,
    pairwise_overlap_matrix,
)


@pytest.fixture
def trained_agents():
    """Create and train agents on a simple task."""
    torch.manual_seed(42)
    n_features = 10
    n_samples = 200

    x = torch.randn(n_samples, n_features)
    # Nonlinear target that linear agent can't fit
    y = (x[:, 0] * x[:, 1] + torch.sin(x[:, 2])).unsqueeze(-1)

    agents = [
        LinearAgent(n_features, 1),
        MLPAgent(n_features, 1, hidden_dim=64),
    ]

    for agent in agents:
        agent.setup_optimizer(lr=1e-2)
        agent.train_mode()
        for _ in range(100):
            pred = agent.predict(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        agent.eval_mode()

    return agents, x, y


class TestComputeBlindSpot:
    def test_returns_mask(self, trained_agents):
        agents, x, y = trained_agents
        mask = compute_blind_spot(agents[0], x, y, threshold=0.1)
        assert mask.dtype == torch.bool
        assert mask.shape == (x.shape[0],)

    def test_linear_has_larger_blind_spot(self, trained_agents):
        """Linear agent should have larger blind spot on nonlinear data."""
        agents, x, y = trained_agents
        b_linear = compute_blind_spot(agents[0], x, y, threshold=0.1)
        b_mlp = compute_blind_spot(agents[1], x, y, threshold=0.1)
        assert b_linear.sum() >= b_mlp.sum()


class TestBlindSpotOverlap:
    def test_identical_masks(self):
        mask = torch.tensor([True, False, True, False])
        assert blind_spot_overlap(mask, mask) == 1.0

    def test_disjoint_masks(self):
        a = torch.tensor([True, False, True, False])
        b = torch.tensor([False, True, False, True])
        assert blind_spot_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = torch.tensor([True, True, False, False])
        b = torch.tensor([True, False, True, False])
        # intersection = 1, union = 3
        assert abs(blind_spot_overlap(a, b) - 1 / 3) < 1e-6

    def test_empty_masks(self):
        a = torch.tensor([False, False])
        b = torch.tensor([False, False])
        assert blind_spot_overlap(a, b) == 0.0


class TestComplementarityScore:
    def test_range(self, trained_agents):
        agents, x, y = trained_agents
        C = complementarity_score(agents, x, y, threshold=0.1)
        assert 0.0 <= C <= 1.0

    def test_diverse_better_than_uniform(self, trained_agents):
        """Diverse agents should have higher complementarity than homogeneous."""
        agents, x, y = trained_agents
        diverse_C = complementarity_score(agents, x, y, threshold=0.1)

        # Homogeneous: two linear agents
        homo = [agents[0], agents[0]]
        homo_C = complementarity_score(homo, x, y, threshold=0.1)

        assert diverse_C >= homo_C


class TestPairwiseOverlap:
    def test_shape(self, trained_agents):
        agents, x, y = trained_agents
        matrix = pairwise_overlap_matrix(agents, x, y)
        assert matrix.shape == (len(agents), len(agents))

    def test_diagonal_ones(self, trained_agents):
        agents, x, y = trained_agents
        matrix = pairwise_overlap_matrix(agents, x, y)
        for i in range(len(agents)):
            assert matrix[i, i] == 1.0 or matrix[i, i] == 0.0  # 0 if empty blind spot
