"""Tests for evidence computation and metrics."""

import pytest
import torch

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.metrics.evidence import (
    weight_of_evidence_mc_dropout,
    weight_of_evidence_kernel,
    pooled_evidence,
    evidence_complementarity,
)
from src.metrics.calibration import expected_calibration_error, brier_score
from src.metrics.diversity import (
    prediction_disagreement,
    hypothesis_class_diversity,
    functional_diversity,
)


@pytest.fixture
def agents():
    return [
        LinearAgent(10, 1),
        MLPAgent(10, 1, hidden_dim=32),
        CNNAgent(10, 1),
        AttentionAgent(10, 1),
    ]


class TestWeightOfEvidence:
    def test_mc_dropout_shape(self, agents):
        x = torch.randn(16, 10)
        for agent in agents:
            v = weight_of_evidence_mc_dropout(agent, x, n_samples=10)
            assert v.shape == (16,)
            assert (v > 0).all()

    def test_kernel_with_training_data(self):
        agent = MLPAgent(10, 1)
        x_train = torch.randn(50, 10)
        x_test = torch.randn(8, 10)
        agent.store_training_points(x_train)
        v = weight_of_evidence_kernel(agent, x_test)
        assert v.shape == (8,)
        assert (v > 0).all()


class TestPooledEvidence:
    def test_pooled_greater_than_individual(self, agents):
        x = torch.randn(16, 10)
        v_pool = pooled_evidence(agents, x, method="mc_dropout")
        assert v_pool.shape == (16,)

        # Pooled should be >= max individual
        v_max = max(
            a.weight_of_evidence(x, method="mc_dropout").mean().item()
            for a in agents
        )
        assert v_pool.mean().item() >= v_max * 0.8  # allow some tolerance

    def test_complementarity_ratio(self, agents):
        x = torch.randn(16, 10)
        result = evidence_complementarity(agents, x, method="mc_dropout")
        assert "v_pool" in result
        assert "v_max" in result
        assert "complementarity_ratio" in result
        assert result["complementarity_ratio"] >= 1.0  # pool >= max


class TestCalibration:
    def test_ece_perfect(self):
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        y_pred = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ece = expected_calibration_error(y_true, y_pred, n_bins=5)
        assert ece < 0.1

    def test_ece_range(self):
        y_true = torch.randint(0, 2, (100,)).float()
        y_pred = torch.rand(100)
        ece = expected_calibration_error(y_true, y_pred)
        assert 0 <= ece <= 1

    def test_brier_score_range(self):
        y_true = torch.randint(0, 2, (100,)).float()
        y_pred = torch.rand(100)
        bs = brier_score(y_true, y_pred)
        assert 0 <= bs <= 1


class TestDiversity:
    def test_prediction_disagreement(self, agents):
        x = torch.randn(32, 10)
        d = prediction_disagreement(agents, x)
        assert d >= 0

    def test_hypothesis_class_diversity(self, agents):
        result = hypothesis_class_diversity(agents)
        assert result["n_unique_classes"] == 4
        assert len(result["class_names"]) == 4
        assert result["param_count_variance"] > 0

    def test_functional_diversity(self, agents):
        x = torch.randn(32, 10)
        result = functional_diversity(agents, x)
        assert result["n_agents"] == 4
        assert 0 < result["effective_ensemble_size"] <= 4
