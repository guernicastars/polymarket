#!/usr/bin/env python3
"""Experiment C: Orthogonality-regularized supervised VAE sweep.

Sweeps over hyperparameters (beta, alpha, gamma) with embedding_dim=8.
For each configuration, trains the model, runs full statistical analysis,
and reports whether the 4/4 criteria are met:

    1. VIF: max < 10
    2. Condition number: improved 5x+ over raw
    3. Probe accuracy: >= 83% (outcome)
    4. Significant dims: > 30% (Wald test, p < 0.05)

Loss: L = L_recon + beta*KL + alpha*BCE + gamma*||corr(Z) - I||_F^2

Usage:
    python run_orth_sweep.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """One hyperparameter configuration to evaluate."""
    name: str
    beta: float
    alpha: float
    gamma: float


SWEEP_CONFIGS = [
    SweepConfig("b1_a1_g1", beta=1.0, alpha=1.0, gamma=1.0),
    SweepConfig("b1_a1_g10", beta=1.0, alpha=1.0, gamma=10.0),
    SweepConfig("b1_a05_g5", beta=1.0, alpha=0.5, gamma=5.0),
    SweepConfig("b2_a1_g5", beta=2.0, alpha=1.0, gamma=5.0),
]

EMBEDDING_DIM = 8
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 128
LR = 1e-3


def load_cached_data() -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load cached data from prior extraction."""
    data_dir = EXPERIMENT_DIR / "results" / "data"
    data = np.load(data_dir / "features.npz", allow_pickle=True)

    X_parts, y_parts = [], []
    for split in ("train", "val", "test"):
        X_parts.append(data[f"X_{split}"])
        y_parts.append(data[f"y_{split}"])
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])

    # Impute NaNs
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            med = np.nanmedian(X[:, col])
            X[mask, col] = med if np.isfinite(med) else 0.0

    # Drop zero-variance
    var = np.var(X, axis=0)
    keep = var >= 1e-10
    if not keep.all():
        dropped = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
        X = X[:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    return X, y, feature_names, metadata


def run_one_config(
    cfg: SweepConfig,
    X: np.ndarray,
    y_binary: np.ndarray,
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata: dict,
    raw_cond: float,
) -> dict:
    """Train one config and run full analysis. Returns result dict."""
    from models.autoencoder import AutoencoderConfig, MarketAutoencoder
    from models.probes import LinearProbe
    from models.statistics import (
        compute_condition_number,
        compute_vif,
        test_orthogonality,
        test_predictive_power,
    )
    from models.train import TrainConfig, train, compute_embedding_stats

    print(f"\n{'='*70}")
    print(f"  Config: {cfg.name}  (beta={cfg.beta}, alpha={cfg.alpha}, gamma={cfg.gamma})")
    print(f"{'='*70}")

    # Prepare data dir for this config
    data_dir = EXPERIMENT_DIR / "results" / f"orth_{cfg.name}_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "features.npy", X_scaled)
    np.save(data_dir / "labels.npy", y_binary)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    output_dir = EXPERIMENT_DIR / "results" / f"orth_{cfg.name}_ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        model_type="supervised_vae",
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=(256, 128),
        dropout=0.1,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        patience=PATIENCE,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=42,
    )

    t0 = time.perf_counter()
    result = train(train_config)
    t_train = time.perf_counter() - t0

    Z = result["embeddings"]
    history = result["history"]

    print(f"\n  Training: {t_train:.1f}s, best_epoch={result['best_epoch']}, "
          f"val_loss={result['best_val_loss']:.4f}")

    if history["pred_acc"]:
        best_acc = max(history["pred_acc"])
        print(f"  Best pred accuracy: {best_acc:.4f}")

    # Embedding quality
    emb_stats = compute_embedding_stats(Z)
    print(f"  Max inter-dim corr: {emb_stats['max_inter_dim_correlation']:.4f}")
    print(f"  Dead dims: {emb_stats['dead_dimensions']}/{EMBEDDING_DIM}")

    # VIF
    emb_names = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    emb_vif = compute_vif(Z, emb_names)
    print(f"  VIF: max={emb_vif.max_vif:.2f}, mean={emb_vif.mean_vif:.2f}, "
          f"severe(>10)={emb_vif.n_severe}")

    # Condition number
    emb_cond = compute_condition_number(Z)
    cond_improvement = raw_cond / max(emb_cond, 1e-6)
    print(f"  Condition number: {emb_cond:.1f} ({cond_improvement:.1f}x improvement)")

    # Wald test
    wald = test_predictive_power(Z, y_binary.astype(int), alpha=0.05)
    sig_frac = wald.n_significant / len(wald.dimension_names)
    print(f"  Wald test significant dims: {wald.n_significant}/{len(wald.dimension_names)} "
          f"({sig_frac*100:.0f}%)")

    # Show per-dim details
    sorted_idx = np.argsort(wald.p_values)
    for i in sorted_idx:
        sig = "***" if wald.p_values[i] < 0.001 else "**" if wald.p_values[i] < 0.01 else "*" if wald.p_values[i] < 0.05 else ""
        print(f"    dim_{i}: coef={wald.coefficients[i]:>8.4f} z={wald.z_scores[i]:>7.2f} "
              f"p={wald.p_values[i]:.4e} {sig}")

    # Orthogonality
    orth = test_orthogonality(Z)
    print(f"  Orthogonality: mean_cos={orth.mean_off_diagonal:.4f}, "
          f"max_cos={orth.max_off_diagonal:.4f}")

    # Probe
    probe = LinearProbe(n_folds=5, n_permutations=50, seed=42)
    probe_result = probe.probe_classification(Z, y_binary.astype(int), "outcome", "embedding")
    probe_acc = probe_result.metrics["accuracy"]
    print(f"  Probe accuracy (5-fold CV): {probe_acc:.4f}")

    # Logistic regression AUC
    emb_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    emb_clf.fit(Z, y_binary.astype(int))
    emb_auc = roc_auc_score(y_binary, emb_clf.predict_proba(Z)[:, 1])
    emb_full_acc = accuracy_score(y_binary.astype(int), emb_clf.predict(Z))
    print(f"  Full logistic reg: acc={emb_full_acc:.4f}, AUC={emb_auc:.4f}")

    # Criteria
    vif_pass = emb_vif.max_vif < 10
    cond_pass = cond_improvement > 5
    probe_pass = probe_acc >= 0.83
    sig_pass = sig_frac > 0.3

    criteria = {
        "VIF max < 10": vif_pass,
        "Condition 5x+": cond_pass,
        "Probe >= 83%": probe_pass,
        "Sig dims > 30%": sig_pass,
    }
    passed = sum(criteria.values())

    print(f"\n  CRITERIA: {passed}/4")
    for name, val in criteria.items():
        print(f"    [{'PASS' if val else 'FAIL'}] {name}")

    if passed == 4:
        verdict = "STRONG SUPPORT"
    elif passed >= 3:
        verdict = "MODERATE SUPPORT"
    elif passed >= 2:
        verdict = "WEAK SUPPORT"
    else:
        verdict = "NOT SUPPORTED"
    print(f"  VERDICT: {verdict}")

    return {
        "name": cfg.name,
        "config": {"beta": cfg.beta, "alpha": cfg.alpha, "gamma": cfg.gamma,
                    "embedding_dim": EMBEDDING_DIM},
        "training": {
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "time_s": t_train,
            "best_pred_acc": max(history["pred_acc"]) if history["pred_acc"] else None,
        },
        "embedding_quality": emb_stats,
        "vif": {
            "max": emb_vif.max_vif, "mean": emb_vif.mean_vif,
            "severe": emb_vif.n_severe,
            "per_dim": {n: float(v) for n, v in zip(emb_vif.feature_names, emb_vif.vif_values)},
        },
        "condition_number": emb_cond,
        "condition_improvement": cond_improvement,
        "wald_test": {
            "significant": wald.n_significant,
            "total": len(wald.dimension_names),
            "fraction": sig_frac,
            "per_dim": {
                f"dim_{i}": {"coef": float(wald.coefficients[i]),
                             "z": float(wald.z_scores[i]),
                             "p": float(wald.p_values[i])}
                for i in range(len(wald.dimension_names))
            },
        },
        "orthogonality": {
            "mean_cos": orth.mean_off_diagonal,
            "max_cos": orth.max_off_diagonal,
        },
        "probe_accuracy": probe_acc,
        "logistic_reg": {"accuracy": emb_full_acc, "auc": emb_auc},
        "criteria": criteria,
        "passed": passed,
        "verdict": verdict,
    }


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 70)
    print("  Experiment C: Orthogonality-Regularized Supervised VAE Sweep")
    print("  L = L_recon + beta*KL + alpha*BCE + gamma*||corr(Z)-I||^2")
    print("  dim=8, 4 configurations")
    print("=" * 70)

    # Load data
    X, y, feature_names, metadata = load_cached_data()
    y_binary = (y >= 0.5).astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    print(f"\n  Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Outcome: Yes={int(y_binary.sum())}, No={int((1-y_binary).sum())}")

    # Raw baseline stats
    from models.statistics import compute_condition_number, compute_vif
    raw_vif = compute_vif(X, feature_names)
    raw_cond = compute_condition_number(X)
    print(f"  Raw VIF: max={raw_vif.max_vif:.2f}, mean={raw_vif.mean_vif:.2f}")
    print(f"  Raw condition number: {raw_cond:.1f}")

    # Run sweep
    all_results = []
    for cfg in SWEEP_CONFIGS:
        result = run_one_config(cfg, X, y_binary, X_scaled, feature_names, metadata, raw_cond)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SWEEP SUMMARY")
    print(f"{'='*70}\n")

    header = f"  {'Config':<15} {'beta':>5} {'alpha':>5} {'gamma':>5} | " \
             f"{'VIF max':>8} {'Cond#':>8} {'Sig%':>5} {'Probe':>6} {'AUC':>6} | {'Score':>5} {'Verdict':<18}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_result = None
    best_score = -1

    for r in all_results:
        c = r["config"]
        sig_pct = r["wald_test"]["fraction"] * 100
        row = (f"  {r['name']:<15} {c['beta']:>5.1f} {c['alpha']:>5.1f} {c['gamma']:>5.1f} | "
               f"{r['vif']['max']:>8.2f} {r['condition_number']:>8.1f} "
               f"{sig_pct:>4.0f}% {r['probe_accuracy']:>6.3f} {r['logistic_reg']['auc']:>6.4f} | "
               f"{r['passed']}/4   {r['verdict']:<18}")
        print(row)

        if r["passed"] > best_score or (r["passed"] == best_score and r["probe_accuracy"] > (best_result["probe_accuracy"] if best_result else 0)):
            best_score = r["passed"]
            best_result = r

    # Highlight best
    print(f"\n  BEST: {best_result['name']} ({best_result['verdict']}, {best_result['passed']}/4)")
    print(f"    VIF max:       {best_result['vif']['max']:.2f}")
    print(f"    Condition #:   {best_result['condition_number']:.1f} "
          f"({best_result['condition_improvement']:.1f}x improvement)")
    print(f"    Sig dims:      {best_result['wald_test']['significant']}/{best_result['wald_test']['total']} "
          f"({best_result['wald_test']['fraction']*100:.0f}%)")
    print(f"    Probe acc:     {best_result['probe_accuracy']:.4f}")
    print(f"    AUC:           {best_result['logistic_reg']['auc']:.4f}")

    # Comparison with baselines
    print(f"\n  Comparison with baselines:")
    print(f"    {'Experiment':<30} {'VIF max':>8} {'Cond#':>10} {'Sig dims':>10} {'Probe':>8}")
    print(f"    {'-'*68}")
    print(f"    {'Raw features (25D)':<30} {raw_vif.max_vif:>8.2f} {raw_cond:>10.1f} {'n/a':>10} {'87.4%':>8}")
    print(f"    {'Unsupervised VAE 64D':<30} {'223.39':>8} {'87.1':>10} {'0/64 (0%)':>10} {'87.4%':>8}")
    print(f"    {'Supervised VAE (no orth)':<30} {'1501.72':>8} {'122.4':>10} {'0/8 (0%)':>10} {'97.5%':>8}")
    best_sig = f"{best_result['wald_test']['significant']}/{best_result['wald_test']['total']} ({best_result['wald_test']['fraction']*100:.0f}%)"
    print(f"    {'Orth-reg BEST':<30} {best_result['vif']['max']:>8.2f} "
          f"{best_result['condition_number']:>10.1f} {best_sig:>10} "
          f"{best_result['probe_accuracy']*100:>7.1f}%")

    # Save report
    reports_dir = EXPERIMENT_DIR / "results" / "orth_sweep_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "C_orth_regularized_supervised_vae",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_baseline": {
            "max_vif": raw_vif.max_vif,
            "mean_vif": raw_vif.mean_vif,
            "condition_number": raw_cond,
        },
        "configs": all_results,
        "best": best_result["name"],
    }
    report_path = reports_dir / "orth_sweep_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate visualizations for the best config
    print(f"\n  Generating visualizations for best config...")
    figures_dir = EXPERIMENT_DIR / "results" / "orth_sweep_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Reload best model embeddings
        best_ckpt_dir = EXPERIMENT_DIR / "results" / f"orth_{best_result['name']}_ckpt"
        Z_best = np.load(best_ckpt_dir / "embeddings.npy")

        from models.visualize import (
            plot_correlation_heatmap,
            plot_embedding_space,
            plot_vif_comparison,
        )
        from models.statistics import compute_vif as cv

        emb_names = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
        emb_vif_best = cv(Z_best, emb_names)

        plot_correlation_heatmap(
            Z_best, feature_names=emb_names,
            title=f"Orth-Reg Embeddings ({best_result['name']})",
            output_dir=str(figures_dir),
        )
        plot_correlation_heatmap(
            X, feature_names=feature_names,
            title="Raw Features",
            output_dir=str(figures_dir),
        )
        plot_vif_comparison(
            raw_vif.vif_values, emb_vif_best.vif_values,
            raw_names=raw_vif.feature_names, embed_names=emb_vif_best.feature_names,
            output_dir=str(figures_dir),
        )
        plot_embedding_space(
            Z_best, y_binary.astype(int),
            label_name="outcome", method="tsne",
            output_dir=str(figures_dir),
        )
        print(f"  Figures saved to {figures_dir}")
    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}")
    print(f"  Sweep complete in {total_elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
