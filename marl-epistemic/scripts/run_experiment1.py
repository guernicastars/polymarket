"""
Experiment 1: Matrix Games with Diverse Observations.

Show that agents with different observation functions find cooperative
equilibria that homogeneous agents miss.

Expected result: Diverse agents cooperate faster and more stably because
their blind spots don't overlap on the same game states.
"""

import argparse
from pathlib import Path

import torch
import yaml
import numpy as np

from src.environments.matrix_games import (
    IteratedPrisonersDilemma,
    StagHunt,
    MatrixGame,
    PolicyNetwork,
    COORDINATION_PAYOFFS,
)
from src.learning.reinforce import REINFORCE
from src.agents.mlp_agent import MLPAgent
from src.metrics.blind_spot import compute_blind_spot, complementarity_score
from src.utils.logging import ExperimentLogger


def create_game(game_type: str, n_agents: int, obs_types: list, **kwargs) -> MatrixGame:
    if game_type == "ipd":
        return IteratedPrisonersDilemma(n_agents=n_agents, observation_types=obs_types, **kwargs)
    elif game_type == "stag_hunt":
        return StagHunt(n_agents=n_agents, observation_types=obs_types, **kwargs)
    elif game_type == "coordination":
        return MatrixGame(COORDINATION_PAYOFFS, n_agents=n_agents, observation_types=obs_types, **kwargs)
    else:
        raise ValueError(f"Unknown game: {game_type}")


def run_condition(
    game_type: str,
    n_agents: int,
    obs_types: list,
    n_episodes: int,
    episode_length: int,
    hidden_dim: int,
    lr: float,
    gamma: float,
    entropy_coef: float,
    seed: int,
    history_length: int = 5,
) -> dict:
    """Run one experimental condition and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    game = create_game(game_type, n_agents, obs_types, history_length=history_length)
    obs_dims = game.observation_dims()

    # Create policy networks + REINFORCE trainers
    policies = [PolicyNetwork(dim, n_actions=2, hidden_dim=hidden_dim) for dim in obs_dims]
    # Wrap in lightweight agent-like objects for REINFORCE
    class PolicyAgent:
        def __init__(self, model, device="cpu"):
            self._model = model
            self._optimizer = None
            self.device = device
        @property
        def model(self):
            return self._model
        @property
        def parameters(self):
            return self._model.parameters()
        def setup_optimizer(self, lr, **kwargs):
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
            return self._optimizer
        @property
        def optimizer(self):
            return self._optimizer
        def train_mode(self):
            self._model.train()
        def eval_mode(self):
            self._model.eval()

    agents = [PolicyAgent(p) for p in policies]
    trainers = [
        REINFORCE(a, lr=lr, gamma=gamma, entropy_coef=entropy_coef)
        for a in agents
    ]

    # Training loop
    coop_rates = []
    total_payoffs = []

    for ep in range(n_episodes):
        observations = game.reset()
        ep_log_probs = [[] for _ in range(n_agents)]
        ep_rewards = [[] for _ in range(n_agents)]
        ep_entropies = [[] for _ in range(n_agents)]

        for t in range(episode_length):
            actions = []
            for i in range(n_agents):
                action, log_prob, entropy = policies[i].get_action(observations[i])
                actions.append(action)
                ep_log_probs[i].append(log_prob)
                ep_entropies[i].append(entropy)

            observations, rewards, done = game.step(actions)
            for i in range(n_agents):
                ep_rewards[i].append(rewards[i])

        # Update each agent
        for i in range(n_agents):
            trainers[i].update(
                ep_log_probs[i], ep_rewards[i], ep_entropies[i]
            )

        coop_rates.append(game.cooperation_rate())
        total_payoffs.append(
            sum(sum(game.payoff_history[i]) for i in range(n_agents))
        )

    return {
        "cooperation_rates": coop_rates,
        "total_payoffs": total_payoffs,
        "final_coop_rate": np.mean(coop_rates[-50:]),
        "final_total_payoff": np.mean(total_payoffs[-50:]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/experiment1.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "game": {"type": "ipd", "n_agents": 3, "history_length": 5,
                     "n_episodes": 500, "episode_length": 100},
            "agents": {"observation_types": ["temporal", "statistical", "relative"],
                       "hidden_dim": 32, "lr": 0.001, "gamma": 0.99, "entropy_coef": 0.01},
        }

    game_cfg = cfg["game"]
    agent_cfg = cfg["agents"]

    conditions = {
        "diverse": agent_cfg["observation_types"],
        "homogeneous_temporal": ["temporal"] * game_cfg["n_agents"],
        "homogeneous_statistical": ["statistical"] * game_cfg["n_agents"],
    }

    logger = ExperimentLogger(
        experiment_name="exp1_matrix_games",
        config=cfg,
        enabled=args.wandb,
        tags=["experiment1", "matrix_games"],
    )

    results = {}
    for cond_name, obs_types in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            r = run_condition(
                game_type=game_cfg["type"],
                n_agents=game_cfg["n_agents"],
                obs_types=obs_types,
                n_episodes=game_cfg["n_episodes"],
                episode_length=game_cfg["episode_length"],
                hidden_dim=agent_cfg["hidden_dim"],
                lr=agent_cfg["lr"],
                gamma=agent_cfg["gamma"],
                entropy_coef=agent_cfg["entropy_coef"],
                seed=seed,
                history_length=game_cfg["history_length"],
            )
            seed_results.append(r)
            print(f"coop={r['final_coop_rate']:.3f}, payoff={r['final_total_payoff']:.1f}")

        # Aggregate
        final_coops = [r["final_coop_rate"] for r in seed_results]
        final_payoffs = [r["final_total_payoff"] for r in seed_results]
        results[cond_name] = {
            "coop_mean": np.mean(final_coops),
            "coop_std": np.std(final_coops),
            "payoff_mean": np.mean(final_payoffs),
            "payoff_std": np.std(final_payoffs),
            "seed_results": seed_results,
        }

        logger.log({
            f"{cond_name}/coop_mean": np.mean(final_coops),
            f"{cond_name}/coop_std": np.std(final_coops),
            f"{cond_name}/payoff_mean": np.mean(final_payoffs),
        })

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"{name:30s}: coop={r['coop_mean']:.3f}+/-{r['coop_std']:.3f}, "
              f"payoff={r['payoff_mean']:.1f}+/-{r['payoff_std']:.1f}")

    logger.finish()
    return results


if __name__ == "__main__":
    main()
