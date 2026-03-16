"""Tests for experiment environments."""

import pytest
import torch

from src.environments.matrix_games import (
    MatrixGame,
    IteratedPrisonersDilemma,
    StagHunt,
    PolicyNetwork,
)
from src.environments.prediction_market import (
    SyntheticEventGenerator,
    PredictionMarketEnv,
)
from src.environments.exploration import ContextualBanditEnv


class TestMatrixGames:
    def test_ipd_creation(self):
        game = IteratedPrisonersDilemma(n_agents=2)
        assert game.n_agents == 2

    def test_observation_dims(self):
        game = IteratedPrisonersDilemma(
            n_agents=3,
            observation_types=["temporal", "statistical", "relative"],
        )
        dims = game.observation_dims()
        assert len(dims) == 3
        assert all(d > 0 for d in dims)

    def test_step(self):
        game = IteratedPrisonersDilemma(n_agents=2)
        obs = game.reset()
        assert len(obs) == 2

        obs, rewards, done = game.step([0, 1])  # cooperate, defect
        assert len(obs) == 2
        assert len(rewards) == 2
        assert rewards[1] > rewards[0]  # defector gets more

    def test_cooperation_rate(self):
        game = IteratedPrisonersDilemma(n_agents=2)
        game.reset()
        # All cooperate
        for _ in range(10):
            game.step([0, 0])
        assert game.cooperation_rate() == 1.0

    def test_policy_network(self):
        policy = PolicyNetwork(input_dim=10, n_actions=2)
        x = torch.randn(10)
        action, log_prob, entropy = policy.get_action(x)
        assert action in [0, 1]
        assert log_prob.dim() == 0
        assert entropy.dim() == 0


class TestPredictionMarket:
    def test_event_generation(self):
        gen = SyntheticEventGenerator(n_features=20, n_events=100, seed=42)
        events = gen.generate()
        assert len(events) == 100
        assert all(0 < e.true_probability < 1 for e in events)

    def test_evidence_class_split(self):
        gen = SyntheticEventGenerator(
            n_events=100, high_evidence_fraction=0.6, seed=42
        )
        events = gen.generate()
        high = sum(1 for e in events if e.evidence_class == "high")
        low = sum(1 for e in events if e.evidence_class == "low")
        assert high == 60
        assert low == 40

    def test_tensor_generation(self):
        gen = SyntheticEventGenerator(n_features=20, n_events=200, seed=42)
        data = gen.generate_tensors()
        assert data["features"].shape == (200, 20)
        assert data["targets"].shape == (200, 1)
        assert data["high_evidence_mask"].shape == (200,)

    def test_env_batch(self):
        gen = SyntheticEventGenerator(n_features=10, n_events=100, seed=42)
        env = PredictionMarketEnv(gen)
        features, targets, outcomes = env.get_batch(batch_size=16)
        assert features.shape == (16, 10)
        assert targets.shape == (16, 1)
        assert outcomes.shape == (16,)


class TestContextualBandit:
    def test_creation(self):
        env = ContextualBanditEnv(context_dim=20, n_arms=10, seed=42)
        assert env.context_dim == 20
        assert env.n_arms == 10

    def test_sample_context(self):
        env = ContextualBanditEnv(seed=42)
        ctx = env.sample_context(batch_size=8)
        assert ctx.shape == (8, 20)

    def test_step(self):
        env = ContextualBanditEnv(seed=42)
        ctx = env.sample_context(batch_size=1).squeeze(0)
        reward, info = env.step(ctx, arm=0)
        assert isinstance(reward, float)
        assert "components" in info
        assert "optimal_arm" in info
        assert "regret" in info

    def test_component_rewards(self):
        env = ContextualBanditEnv(seed=42)
        ctx = env.sample_context(batch_size=1).squeeze(0).numpy()
        components = env.compute_reward_components(ctx, arm=0)
        assert "linear" in components
        assert "local" in components
        assert "longrange" in components
        assert "interaction" in components

    def test_regret_tracking(self):
        env = ContextualBanditEnv(seed=42)
        for _ in range(10):
            ctx = env.sample_context(batch_size=1).squeeze(0)
            env.step(ctx, arm=0)
        assert len(env.regret_history) == 10
        assert env.cumulative_regret >= 0
