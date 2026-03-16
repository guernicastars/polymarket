"""
Experiment 3: Multi-Agent Contextual Bandits.

Show that architectural diversity shrinks the collective blind spot
and that evidence-seeking LOLA accelerates this.

Conditions:
(a) Single best agent
(b) 4 homogeneous agents (MLP)
(c) 4 diverse agents (Linear, MLP, CNN, Attention)
(d) Diverse + Keynesian weighting
(e) Diverse + Evidence-seeking LOLA
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.agents.ensemble import KeynesianEnsemble, SimpleEnsemble
from src.environments.exploration import ContextualBanditEnv
from src.metrics.blind_spot import compute_blind_spot, complementarity_score
from src.metrics.evidence import pooled_evidence
from src.utils.logging import ExperimentLogger


def create_agent(arch: str, input_dim: int, output_dim: int, **kwargs):
    if arch == "linear":
        return LinearAgent(input_dim, output_dim, **kwargs)
    elif arch == "mlp":
        return MLPAgent(input_dim, output_dim, **kwargs)
    elif arch == "cnn":
        return CNNAgent(input_dim, output_dim, **kwargs)
    elif arch == "attention":
        return AttentionAgent(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def run_single_agent(
    env: ContextualBanditEnv,
    arch: str,
    n_steps: int,
    lr: float,
    seed: int,
) -> dict:
    """Run single agent on contextual bandit."""
    torch.manual_seed(seed)
    env.reset_metrics()

    agent = create_agent(arch, env.context_dim, env.n_arms)
    agent.setup_optimizer(lr=lr)
    agent.train_mode()

    for step in range(n_steps):
        ctx = env.sample_context(batch_size=1).squeeze(0)
        with torch.no_grad():
            q_values = agent.predict(ctx.unsqueeze(0)).squeeze(0)
        # Epsilon-greedy
        eps = max(0.01, 0.3 * (1 - step / n_steps))
        if np.random.random() < eps:
            arm = np.random.randint(env.n_arms)
        else:
            arm = q_values.argmax().item()

        reward, info = env.step(ctx, arm)

        # Update via MSE on observed reward
        target = torch.zeros(env.n_arms)
        target[arm] = reward
        pred = agent.predict(ctx.unsqueeze(0)).squeeze(0)
        loss = F.mse_loss(pred[arm:arm+1], torch.tensor([reward]))
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    return {
        "cumulative_regret": env.cumulative_regret,
        "regret_history": env.regret_history,
        "component_regret": {k: v[-1] if v else 0 for k, v in env.component_regret.items()},
    }


def run_ensemble_condition(
    env: ContextualBanditEnv,
    architectures: list,
    n_steps: int,
    lr: float,
    use_keynesian: bool,
    seed: int,
) -> dict:
    """Run ensemble of agents on contextual bandit."""
    torch.manual_seed(seed)
    env.reset_metrics()

    agents = [
        create_agent(arch, env.context_dim, env.n_arms)
        for arch in architectures
    ]
    for a in agents:
        a.setup_optimizer(lr=lr)
        a.train_mode()

    complementarity_history = []

    for step in range(n_steps):
        ctx = env.sample_context(batch_size=1).squeeze(0)

        # Ensemble prediction
        with torch.no_grad():
            if use_keynesian:
                ens = KeynesianEnsemble(agents)
                q_values = ens.predict(ctx.unsqueeze(0)).squeeze(0)
            else:
                preds = [a.predict(ctx.unsqueeze(0)).squeeze(0) for a in agents]
                q_values = torch.stack(preds).mean(dim=0)

        # Epsilon-greedy
        eps = max(0.01, 0.3 * (1 - step / n_steps))
        if np.random.random() < eps:
            arm = np.random.randint(env.n_arms)
        else:
            arm = q_values.argmax().item()

        reward, info = env.step(ctx, arm)

        # Update all agents
        for agent in agents:
            agent.train_mode()
            pred = agent.predict(ctx.unsqueeze(0)).squeeze(0)
            loss = F.mse_loss(pred[arm:arm+1], torch.tensor([reward]))
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        # Periodically compute complementarity
        if step % 500 == 0 and step > 0:
            with torch.no_grad():
                test_ctx = env.sample_context(batch_size=100)
                true_rewards = env.batch_true_rewards(test_ctx)
                best_rewards = true_rewards.max(dim=1).values.unsqueeze(-1)
                # Use reward prediction error as blind spot proxy
                C = complementarity_score(agents, test_ctx, best_rewards, threshold=0.5)
                complementarity_history.append((step, C))

    return {
        "cumulative_regret": env.cumulative_regret,
        "regret_history": env.regret_history,
        "component_regret": {k: v[-1] if v else 0 for k, v in env.component_regret.items()},
        "complementarity_history": complementarity_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/experiment3.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "bandit": {"context_dim": 20, "n_arms": 10, "noise_std": 0.1},
            "training": {"n_steps": args.steps, "lr": 0.001},
        }

    bandit_cfg = cfg.get("bandit", {})
    train_cfg = cfg.get("training", {})
    n_steps = train_cfg.get("n_steps", args.steps)
    lr = train_cfg.get("lr", 0.001)

    logger = ExperimentLogger(
        experiment_name="exp3_contextual_bandits",
        config=cfg,
        enabled=args.wandb,
        tags=["experiment3", "contextual_bandits"],
    )

    conditions = {
        "single_mlp": {"type": "single", "arch": "mlp"},
        "homogeneous_4": {"type": "ensemble", "archs": ["mlp"] * 4, "keynesian": False},
        "diverse_4": {"type": "ensemble", "archs": ["linear", "mlp", "cnn", "attention"], "keynesian": False},
        "diverse_keynesian": {"type": "ensemble", "archs": ["linear", "mlp", "cnn", "attention"], "keynesian": True},
    }

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        env = ContextualBanditEnv(
            context_dim=bandit_cfg.get("context_dim", 20),
            n_arms=bandit_cfg.get("n_arms", 10),
            noise_std=bandit_cfg.get("noise_std", 0.1),
            seed=seed,
        )

        for cond_name, cond_cfg in conditions.items():
            print(f"  {cond_name}...", end=" ", flush=True)

            if cond_cfg["type"] == "single":
                result = run_single_agent(env, cond_cfg["arch"], n_steps, lr, seed)
            else:
                result = run_ensemble_condition(
                    env, cond_cfg["archs"], n_steps, lr,
                    cond_cfg["keynesian"], seed,
                )

            print(f"regret={result['cumulative_regret']:.1f}")
            logger.log({
                f"seed_{seed}/{cond_name}/cumulative_regret": result["cumulative_regret"],
            })

    logger.finish()


if __name__ == "__main__":
    main()
