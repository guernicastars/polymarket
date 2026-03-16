"""Tests for agent implementations."""

import pytest
import torch

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.agents.ensemble import SimpleEnsemble, KeynesianEnsemble


@pytest.fixture
def input_dim():
    return 20


@pytest.fixture
def output_dim():
    return 1


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def agents(input_dim, output_dim):
    return [
        LinearAgent(input_dim, output_dim),
        MLPAgent(input_dim, output_dim),
        CNNAgent(input_dim, output_dim),
        AttentionAgent(input_dim, output_dim),
    ]


class TestAgentCreation:
    def test_linear_agent(self, input_dim, output_dim):
        agent = LinearAgent(input_dim, output_dim)
        assert agent.hypothesis_class_name == "linear"
        assert agent._model is not None

    def test_mlp_agent(self, input_dim, output_dim):
        agent = MLPAgent(input_dim, output_dim)
        assert agent.hypothesis_class_name == "mlp"

    def test_cnn_agent(self, input_dim, output_dim):
        agent = CNNAgent(input_dim, output_dim)
        assert agent.hypothesis_class_name == "cnn"

    def test_attention_agent(self, input_dim, output_dim):
        agent = AttentionAgent(input_dim, output_dim)
        assert agent.hypothesis_class_name == "attention"

    def test_different_param_counts(self, agents):
        """Agents should have different parameter counts (different H)."""
        param_counts = [sum(p.numel() for p in a.parameters) for a in agents]
        # Not all the same
        assert len(set(param_counts)) > 1


class TestAgentPrediction:
    def test_predict_shape(self, agents, batch_size, input_dim, output_dim):
        x = torch.randn(batch_size, input_dim)
        for agent in agents:
            pred = agent.predict(x)
            assert pred.shape == (batch_size, output_dim)

    def test_predict_differentiable(self, agents, batch_size, input_dim):
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        for agent in agents:
            agent.train_mode()
            pred = agent.predict(x)
            loss = pred.sum()
            loss.backward()
            assert x.grad is not None
            x.grad = None


class TestWeightOfEvidence:
    def test_mc_dropout_positive(self, agents, batch_size, input_dim):
        x = torch.randn(batch_size, input_dim)
        for agent in agents:
            v = agent.weight_of_evidence(x, method="mc_dropout")
            assert v.shape == (batch_size,)
            assert (v > 0).all()

    def test_kernel_evidence(self, agents, batch_size, input_dim):
        x = torch.randn(batch_size, input_dim)
        x_train = torch.randn(100, input_dim)
        for agent in agents:
            agent.store_training_points(x_train)
            v = agent.weight_of_evidence(x, method="kernel")
            assert v.shape == (batch_size,)
            assert (v > 0).all()


class TestBlindSpotScore:
    def test_blind_spot_range(self, agents, batch_size, input_dim, output_dim):
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, output_dim)
        for agent in agents:
            score = agent.blind_spot_score(x, y)
            assert score.shape == (batch_size,)
            assert (score >= 0).all() and (score <= 1).all()


class TestEnsembles:
    def test_simple_ensemble(self, agents, batch_size, input_dim, output_dim):
        x = torch.randn(batch_size, input_dim)
        ens = SimpleEnsemble(agents)
        pred = ens.predict(x)
        assert pred.shape == (batch_size, output_dim)

    def test_keynesian_ensemble(self, agents, batch_size, input_dim, output_dim):
        x = torch.randn(batch_size, input_dim)
        ens = KeynesianEnsemble(agents)
        pred = ens.predict(x)
        assert pred.shape == (batch_size, output_dim)

    def test_keynesian_with_evidence(self, agents, batch_size, input_dim, output_dim):
        x = torch.randn(batch_size, input_dim)
        ens = KeynesianEnsemble(agents)
        pred, v_pool, individual_v = ens.predict_with_evidence(x)
        assert pred.shape == (batch_size, output_dim)
        assert v_pool.shape == (batch_size,)
        assert len(individual_v) == len(agents)
