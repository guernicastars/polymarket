"""
Experiment 2: Multi-Agent Prediction Market.

Show Keynesian evidence weighting outperforms simple ensembling,
especially on low-evidence (Pi_U) events.

Comparison: (a) Best single, (b) Simple avg, (c) Accuracy-weighted,
(d) Keynesian-weighted, (e) LOLA + evidence-seeking.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.agents.ensemble import (
    SimpleEnsemble,
    AccuracyWeightedEnsemble,
    KeynesianEnsemble,
)
from src.environments.prediction_market import SyntheticEventGenerator, PredictionMarketEnv
from src.learning.keynesian_loss import KeynesianLossFunction
from src.metrics.blind_spot import compute_blind_spot, complementarity_score
from src.metrics.calibration import expected_calibration_error, brier_score
from src.metrics.evidence import evidence_complementarity
from src.utils.logging import ExperimentLogger


def create_agent(arch: str, input_dim: int, output_dim: int = 1, **kwargs):
    """Factory for creating agents by architecture name."""
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


def train_agent(
    agent,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_keynesian: bool = False,
    mu: float = 0.1,
):
    """Train a single agent on prediction data."""
    agent.setup_optimizer(lr=lr)
    agent.store_training_points(x_train)

    loss_fn = KeynesianLossFunction(mu=mu, mu_schedule="warmup", mu_warmup_steps=500) \
        if use_keynesian else None

    n_samples = x_train.shape[0]
    agent.train_mode()

    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            pred = torch.sigmoid(agent.predict(x_batch))

            if loss_fn:
                evidence = agent.weight_of_evidence(x_batch, method="mc_dropout")
                agent.train_mode()  # re-enable dropout after evidence computation
                loss = loss_fn(y_batch, pred, evidence)
            else:
                loss = F.mse_loss(pred, y_batch)

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters, 5.0)
            agent.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    agent.eval_mode()
    return epoch_loss / max(n_batches, 1)


def evaluate_method(
    method_name: str,
    predict_fn,
    env: PredictionMarketEnv,
    x_test: torch.Tensor,
) -> dict:
    """Evaluate a prediction method."""
    with torch.no_grad():
        preds = torch.sigmoid(predict_fn(x_test))

    targets = env.data["targets"][:x_test.shape[0]].squeeze(-1)
    outcomes = env.outcomes[:x_test.shape[0]]
    high_mask = env.data["high_evidence_mask"][:x_test.shape[0]]
    low_mask = env.data["low_evidence_mask"][:x_test.shape[0]]

    preds_flat = preds.squeeze(-1)

    result = {
        "method": method_name,
        "mse": (preds_flat - targets).pow(2).mean().item(),
        "brier": brier_score(outcomes, preds_flat),
    }

    if high_mask.any():
        result["mse_high"] = (preds_flat[high_mask] - targets[high_mask]).pow(2).mean().item()
        result["brier_high"] = brier_score(outcomes[high_mask], preds_flat[high_mask])

    if low_mask.any():
        result["mse_low"] = (preds_flat[low_mask] - targets[low_mask]).pow(2).mean().item()
        result["brier_low"] = brier_score(outcomes[low_mask], preds_flat[low_mask])

    result["ece"] = expected_calibration_error(outcomes, preds_flat)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/experiment2.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "market": {"n_features": 20, "n_events": 2000,
                       "high_evidence_fraction": 0.6, "noise_scale": 0.1},
            "agents": {"architectures": ["linear", "mlp", "cnn", "attention"],
                       "hidden_dim": 128, "dropout_rate": 0.1},
            "training": {"n_epochs": 100, "batch_size": 64, "lr": 0.001},
            "keynesian": {"mu": 0.1},
        }

    market_cfg = cfg["market"]
    agent_cfg = cfg["agents"]
    train_cfg = cfg["training"]
    key_cfg = cfg["keynesian"]

    logger = ExperimentLogger(
        experiment_name="exp2_prediction_market",
        config=cfg,
        enabled=args.wandb,
        tags=["experiment2", "prediction_market"],
    )

    all_results = {}

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data
        gen = SyntheticEventGenerator(
            n_features=market_cfg["n_features"],
            n_events=market_cfg["n_events"],
            high_evidence_fraction=market_cfg["high_evidence_fraction"],
            noise_scale=market_cfg["noise_scale"],
            seed=seed,
        )
        env = PredictionMarketEnv(gen)
        data = env.data

        # Train/test split (80/20)
        n = data["features"].shape[0]
        n_train = int(0.8 * n)
        perm = torch.randperm(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        x_train = data["features"][train_idx]
        y_train = data["targets"][train_idx]
        x_test = data["features"][test_idx]

        # Create and train agents
        agents = []
        for arch in agent_cfg["architectures"]:
            agent = create_agent(
                arch,
                input_dim=market_cfg["n_features"],
                dropout_rate=agent_cfg.get("dropout_rate", 0.1),
            )
            print(f"  Training {arch}...", end=" ", flush=True)
            final_loss = train_agent(
                agent, x_train, y_train,
                n_epochs=train_cfg["n_epochs"],
                batch_size=train_cfg["batch_size"],
                lr=train_cfg["lr"],
            )
            print(f"loss={final_loss:.4f}")
            agents.append(agent)

        # Also train Keynesian versions
        keynesian_agents = []
        for arch in agent_cfg["architectures"]:
            agent = create_agent(
                arch,
                input_dim=market_cfg["n_features"],
                dropout_rate=agent_cfg.get("dropout_rate", 0.1),
            )
            train_agent(
                agent, x_train, y_train,
                n_epochs=train_cfg["n_epochs"],
                batch_size=train_cfg["batch_size"],
                lr=train_cfg["lr"],
                use_keynesian=True,
                mu=key_cfg["mu"],
            )
            keynesian_agents.append(agent)

        # Evaluate methods
        # (a) Best single agent
        best_mse = float("inf")
        best_agent = agents[0]
        for a in agents:
            with torch.no_grad():
                pred = torch.sigmoid(a.predict(x_test))
                mse = F.mse_loss(pred, data["targets"][test_idx]).item()
            if mse < best_mse:
                best_mse = mse
                best_agent = a

        # (b) Simple ensemble
        simple_ens = SimpleEnsemble(agents)

        # (c) Accuracy-weighted
        acc_ens = AccuracyWeightedEnsemble(agents)
        acc_ens.fit_weights(x_train[:200], y_train[:200])

        # (d) Keynesian-weighted
        key_ens = KeynesianEnsemble(keynesian_agents)

        methods = {
            "best_single": lambda x: best_agent.predict(x),
            "simple_avg": lambda x: simple_ens.predict(x),
            "accuracy_weighted": lambda x: acc_ens.predict(x),
            "keynesian_weighted": lambda x: key_ens.predict(x),
        }

        # Re-index env for test set evaluation
        # Create a test-specific env view
        test_env = PredictionMarketEnv.__new__(PredictionMarketEnv)
        test_env.data = {
            "features": data["features"][test_idx],
            "targets": data["targets"][test_idx],
            "high_evidence_mask": data["high_evidence_mask"][test_idx],
            "low_evidence_mask": data["low_evidence_mask"][test_idx],
        }
        test_env.outcomes = env.outcomes[test_idx]

        for name, pred_fn in methods.items():
            result = evaluate_method(name, pred_fn, test_env, x_test)
            print(f"  {name:25s}: MSE={result['mse']:.4f}, "
                  f"MSE_high={result.get('mse_high', 0):.4f}, "
                  f"MSE_low={result.get('mse_low', 0):.4f}, "
                  f"ECE={result['ece']:.4f}")

            for k, v in result.items():
                if isinstance(v, (int, float)):
                    key = f"seed_{seed}/{name}/{k}"
                    all_results.setdefault(key, []).append(v)
                    logger.log({key: v})

    # Summary
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean +/- std)")
    print(f"{'='*60}")

    logger.finish()


if __name__ == "__main__":
    main()
