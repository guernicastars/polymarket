"""
Experiment 4: Molecular Property Prediction with Epistemic Diversity.

Four architectures with genuinely different molecular representations:
  - Linear on Morgan fingerprints (captures additive group contributions)
  - MLP on RDKit descriptors (captures nonlinear descriptor combinations)
  - CNN on SMILES encodings (captures local chemical motifs)
  - Attention on atom-level features (captures long-range intramolecular interactions)

Evaluation on scaffold splits — testing generalization to novel chemical scaffolds,
the exact setting where Keynesian evidence weighting should add most value.

Comparison: (a) Best single, (b) Simple avg, (c) Accuracy-weighted,
(d) Keynesian-weighted ensembles.
"""

import argparse
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*DEPRECATION WARNING.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import numpy as np

from src.agents import LinearAgent, MLPAgent, CNNAgent, AttentionAgent
from src.agents.ensemble import (
    SimpleEnsemble,
    AccuracyWeightedEnsemble,
    KeynesianEnsemble,
)
from src.environments.molecular import (
    DATASETS,
    MolecularDataset,
    get_representation_dims,
)
from src.metrics.blind_spot import compute_blind_spot, complementarity_score
from src.metrics.calibration import brier_score
from src.metrics.evidence import evidence_complementarity
from src.utils.logging import ExperimentLogger


# Architecture → representation mapping
ARCH_REPR = {
    "linear": "fingerprint",
    "mlp": "descriptor",
    "cnn": "smiles",
    "attention": "atom",
}


def get_features(dataset: MolecularDataset, representation: str, indices: np.ndarray) -> torch.Tensor:
    """Get feature matrix for a representation and set of indices."""
    if representation == "fingerprint":
        return dataset.fingerprints[indices]
    elif representation == "descriptor":
        return dataset.descriptors[indices]
    elif representation == "smiles":
        return dataset.smiles_encoded[indices]
    elif representation == "atom":
        return dataset.atom_features[indices]
    else:
        raise ValueError(f"Unknown representation: {representation}")


def create_agent(arch: str, input_dim: int, output_dim: int = 1, **kwargs):
    """Create agent by architecture name."""
    if arch == "linear":
        return LinearAgent(input_dim, output_dim, **kwargs)
    elif arch == "mlp":
        return MLPAgent(input_dim, output_dim, hidden_dim=128, **kwargs)
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
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    task_type: str = "regression",
) -> float:
    """Train agent with early stopping on validation loss."""
    agent.setup_optimizer(lr=lr)
    agent.store_training_points(x_train)

    n_samples = x_train.shape[0]
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    agent.train_mode()

    for epoch in range(n_epochs):
        # Training
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            pred = agent.predict(x_batch)
            if task_type == "classification":
                pred = torch.sigmoid(pred)
                loss = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), y_batch)
            else:
                loss = F.mse_loss(pred, y_batch)

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters, 5.0)
            agent.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        agent.eval_mode()
        with torch.no_grad():
            val_pred = agent.predict(x_val)
            if task_type == "classification":
                val_pred = torch.sigmoid(val_pred)
                val_loss = F.binary_cross_entropy(
                    val_pred.clamp(1e-7, 1 - 1e-7), y_val
                ).item()
            else:
                val_loss = F.mse_loss(val_pred, y_val).item()
        agent.train_mode()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in agent.model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        agent.model.load_state_dict(best_state)
    agent.eval_mode()

    return best_val_loss


def evaluate_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
) -> dict:
    """Compute evaluation metrics."""
    preds_flat = preds.squeeze(-1)
    targets_flat = targets.squeeze(-1)

    result = {}
    if task_type == "regression":
        result["rmse"] = (preds_flat - targets_flat).pow(2).mean().sqrt().item()
        result["mae"] = (preds_flat - targets_flat).abs().mean().item()
        # R² score
        ss_res = (targets_flat - preds_flat).pow(2).sum()
        ss_tot = (targets_flat - targets_flat.mean()).pow(2).sum()
        result["r2"] = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
    else:
        # Classification
        result["bce"] = F.binary_cross_entropy(
            preds_flat.clamp(1e-7, 1 - 1e-7), targets_flat
        ).item()
        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            result["auroc"] = roc_auc_score(
                targets_flat.numpy(), preds_flat.numpy()
            )
        except Exception:
            result["auroc"] = 0.0
        # Accuracy at 0.5 threshold
        result["accuracy"] = (
            (preds_flat > 0.5) == (targets_flat > 0.5)
        ).float().mean().item()

    return result


def run_dataset(
    dataset_name: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    mc_samples: int,
    seed: int,
) -> dict:
    """Run full experiment on one dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    print(f"\n  Loading {dataset_name}...", end=" ", flush=True)
    loader = DATASETS[dataset_name]
    dataset = loader(seed=seed)
    dims = get_representation_dims(dataset)
    print(f"{len(dataset.smiles)} molecules, "
          f"train={len(dataset.train_idx)}, "
          f"val={len(dataset.val_idx)}, "
          f"test={len(dataset.test_idx)}")

    y_train = dataset.targets[dataset.train_idx]
    y_val = dataset.targets[dataset.val_idx]
    y_test = dataset.targets[dataset.test_idx]

    # Train each architecture on its representation
    agents = {}
    architectures = ["linear", "mlp", "cnn", "attention"]

    for arch in architectures:
        repr_name = ARCH_REPR[arch]
        input_dim = dims[repr_name]

        x_train = get_features(dataset, repr_name, dataset.train_idx)
        x_val = get_features(dataset, repr_name, dataset.val_idx)

        agent = create_agent(arch, input_dim, dropout_rate=0.1, mc_dropout_samples=mc_samples)
        print(f"  Training {arch:10s} on {repr_name:12s} (dim={input_dim})...", end=" ", flush=True)

        val_loss = train_agent(
            agent, x_train, y_train, x_val, y_val,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            task_type=dataset.task_type,
        )
        # Calibrate evidence using validation performance
        agent.calibrate_evidence(x_val, y_val)
        print(f"val_loss={val_loss:.4f} cal_scale={agent._calibration_scale:.2f}")
        agents[arch] = agent

    # Evaluate each architecture individually on test set
    print(f"\n  {'Method':25s} | ", end="")
    if dataset.task_type == "regression":
        print(f"{'RMSE':>8s} {'MAE':>8s} {'R²':>8s}")
    else:
        print(f"{'AUROC':>8s} {'Acc':>8s} {'BCE':>8s}")

    print(f"  {'-'*60}")

    test_preds = {}
    test_metrics = {}

    for arch in architectures:
        repr_name = ARCH_REPR[arch]
        x_test = get_features(dataset, repr_name, dataset.test_idx)

        with torch.no_grad():
            pred = agents[arch].predict(x_test)
            if dataset.task_type == "classification":
                pred = torch.sigmoid(pred)
            test_preds[arch] = pred

        metrics = evaluate_predictions(pred, y_test, dataset.task_type)
        test_metrics[arch] = metrics

        print(f"  {arch:25s} | ", end="")
        if dataset.task_type == "regression":
            print(f"{metrics['rmse']:8.4f} {metrics['mae']:8.4f} {metrics['r2']:8.4f}")
        else:
            print(f"{metrics['auroc']:8.4f} {metrics['accuracy']:8.4f} {metrics['bce']:8.4f}")

    # --- Ensemble Methods ---

    # Helper: get predictions from all agents for a given set of indices
    def get_all_preds(indices):
        preds_list = []
        for arch in architectures:
            repr_name = ARCH_REPR[arch]
            x = get_features(dataset, repr_name, indices)
            with torch.no_grad():
                p = agents[arch].predict(x)
                if dataset.task_type == "classification":
                    p = torch.sigmoid(p)
            preds_list.append(p)
        return preds_list

    test_preds_list = [test_preds[arch] for arch in architectures]

    # (a) Best single agent
    if dataset.task_type == "regression":
        best_arch = min(architectures, key=lambda a: test_metrics[a]["rmse"])
    else:
        best_arch = max(architectures, key=lambda a: test_metrics[a]["auroc"])
    best_single_metrics = test_metrics[best_arch]

    print(f"  {'best_single (' + best_arch + ')':25s} | ", end="")
    if dataset.task_type == "regression":
        print(f"{best_single_metrics['rmse']:8.4f} {best_single_metrics['mae']:8.4f} {best_single_metrics['r2']:8.4f}")
    else:
        print(f"{best_single_metrics['auroc']:8.4f} {best_single_metrics['accuracy']:8.4f} {best_single_metrics['bce']:8.4f}")

    # (b) Simple average
    simple_pred = torch.stack(test_preds_list).mean(dim=0)
    simple_metrics = evaluate_predictions(simple_pred, y_test, dataset.task_type)

    print(f"  {'simple_avg':25s} | ", end="")
    if dataset.task_type == "regression":
        print(f"{simple_metrics['rmse']:8.4f} {simple_metrics['mae']:8.4f} {simple_metrics['r2']:8.4f}")
    else:
        print(f"{simple_metrics['auroc']:8.4f} {simple_metrics['accuracy']:8.4f} {simple_metrics['bce']:8.4f}")

    # (c) Accuracy-weighted (fit weights on validation set)
    val_preds_list = get_all_preds(dataset.val_idx)
    val_errors = []
    for vp in val_preds_list:
        if dataset.task_type == "regression":
            err = F.mse_loss(vp, y_val).item()
        else:
            err = F.binary_cross_entropy(
                vp.clamp(1e-7, 1 - 1e-7), y_val
            ).item()
        val_errors.append(err)

    # Weights inversely proportional to error
    inv_errors = [1.0 / (e + 1e-8) for e in val_errors]
    total = sum(inv_errors)
    acc_weights = [w / total for w in inv_errors]

    acc_pred = sum(w * p for w, p in zip(acc_weights, test_preds_list))
    acc_metrics = evaluate_predictions(acc_pred, y_test, dataset.task_type)

    print(f"  {'accuracy_weighted':25s} | ", end="")
    if dataset.task_type == "regression":
        print(f"{acc_metrics['rmse']:8.4f} {acc_metrics['mae']:8.4f} {acc_metrics['r2']:8.4f}")
    else:
        print(f"{acc_metrics['auroc']:8.4f} {acc_metrics['accuracy']:8.4f} {acc_metrics['bce']:8.4f}")

    # (d) Keynesian-weighted (per-sample evidence weighting)
    # Compute evidence for each agent on test set using calibrated method
    # (MC dropout scaled by validation accuracy — fixes 'confidently wrong' issue)
    agent_list = [agents[arch] for arch in architectures]
    evidences = []
    for arch in architectures:
        repr_name = ARCH_REPR[arch]
        x_test_repr = get_features(dataset, repr_name, dataset.test_idx)
        ev = agents[arch].weight_of_evidence(x_test_repr, method="calibrated")
        evidences.append(ev)

    # Per-sample evidence-weighted average
    ev_stack = torch.stack(evidences)  # (n_agents, n_test, 1) or (n_agents, n_test)
    if ev_stack.dim() == 2:
        ev_stack = ev_stack.unsqueeze(-1)
    pred_stack = torch.stack(test_preds_list)  # (n_agents, n_test, 1)

    ev_weights = ev_stack / ev_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)
    keynesian_pred = (ev_weights * pred_stack).sum(dim=0)
    keynesian_metrics = evaluate_predictions(keynesian_pred, y_test, dataset.task_type)

    print(f"  {'keynesian_weighted':25s} | ", end="")
    if dataset.task_type == "regression":
        print(f"{keynesian_metrics['rmse']:8.4f} {keynesian_metrics['mae']:8.4f} {keynesian_metrics['r2']:8.4f}")
    else:
        print(f"{keynesian_metrics['auroc']:8.4f} {keynesian_metrics['accuracy']:8.4f} {keynesian_metrics['bce']:8.4f}")

    # --- Evidence Analysis ---
    ev_pool = ev_stack.sum(dim=0).mean().item()
    ev_max = ev_stack.max(dim=0).values.mean().item()
    ev_ratio = ev_pool / max(ev_max, 1e-8)
    per_agent_ev = {arch: evidences[i].mean().item() for i, arch in enumerate(architectures)}

    print(f"\n  Evidence analysis:")
    print(f"    V_pool (mean) = {ev_pool:.4f}")
    print(f"    V_max  (mean) = {ev_max:.4f}")
    print(f"    Complementarity ratio = {ev_ratio:.4f} (>1 = complementary)")
    for arch, v in per_agent_ev.items():
        print(f"    V_{arch:10s} = {v:.4f}")

    # --- Blind Spot Analysis ---
    blind_spots = []
    for i, arch in enumerate(architectures):
        pf = test_preds_list[i].squeeze(-1)
        tf = y_test.squeeze(-1)
        error = (pf - tf).pow(2)
        threshold = error.median().item() * 1.5
        bs = error > threshold
        blind_spots.append(bs)

    # Collective blind spot = intersection
    collective = blind_spots[0]
    for bs in blind_spots[1:]:
        collective = collective & bs

    individual_sizes = [bs.float().sum().item() for bs in blind_spots]
    collective_size = collective.float().sum().item()
    min_individual = min(individual_sizes) if individual_sizes else 1

    comp_score = 1.0 - collective_size / max(min_individual, 1.0)
    comp_score = max(0.0, min(1.0, comp_score))

    print(f"\n  Blind spot analysis:")
    for arch, size in zip(architectures, individual_sizes):
        print(f"    |B_{arch:10s}| = {int(size)}")
    print(f"    |B_collective|   = {int(collective_size)}")
    print(f"    Complementarity  = {comp_score:.4f}")

    return {
        "dataset": dataset_name,
        "task_type": dataset.task_type,
        "n_molecules": len(dataset.smiles),
        "individual": test_metrics,
        "best_single": {"arch": best_arch, **best_single_metrics},
        "simple_avg": simple_metrics,
        "accuracy_weighted": acc_metrics,
        "keynesian_weighted": keynesian_metrics,
        "evidence": {
            "v_pool": ev_pool,
            "v_max": ev_max,
            "ratio": ev_ratio,
            "per_agent": per_agent_ev,
        },
        "blind_spot": {
            "individual_sizes": dict(zip(architectures, individual_sizes)),
            "collective_size": collective_size,
            "complementarity": comp_score,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["esol", "freesolv", "lipophilicity", "bbbp"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    logger = ExperimentLogger(
        experiment_name="exp4_molecular",
        config=vars(args),
        enabled=args.wandb,
        tags=["experiment4", "molecular"],
    )

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        seed_results = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            result = run_dataset(
                dataset_name=dataset_name,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                mc_samples=args.mc_samples,
                seed=seed,
            )
            seed_results.append(result)

        all_results[dataset_name] = seed_results

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY ACROSS ALL DATASETS AND SEEDS")
    print(f"{'='*70}")

    for dataset_name, seed_results in all_results.items():
        task_type = seed_results[0]["task_type"]
        metric_key = "rmse" if task_type == "regression" else "auroc"
        direction = "lower" if task_type == "regression" else "higher"

        print(f"\n{dataset_name.upper()} ({task_type}, {metric_key} — {direction} is better):")

        methods = ["best_single", "simple_avg", "accuracy_weighted", "keynesian_weighted"]
        for method in methods:
            values = [r[method][metric_key] for r in seed_results]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {method:25s}: {mean:.4f} ± {std:.4f}")

        # Complementarity
        comp_values = [r["blind_spot"]["complementarity"] for r in seed_results]
        print(f"  {'complementarity':25s}: {np.mean(comp_values):.4f} ± {np.std(comp_values):.4f}")

        # Evidence ratio
        ev_ratios = [r["evidence"]["ratio"] for r in seed_results]
        print(f"  {'evidence ratio':25s}: {np.mean(ev_ratios):.4f} ± {np.std(ev_ratios):.4f}")

    logger.finish()
    return all_results


if __name__ == "__main__":
    main()
