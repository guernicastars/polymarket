#!/usr/bin/env python3
"""Experiment B: Supervised VAE with outcome signal in the loss.

Hypothesis: Adding a prediction head to the VAE forces outcome-relevant
information into specific embedding dimensions, making them individually
significant in Wald tests (unlike pure reconstruction VAE where signal
spreads across all dims).

L_SVAE = L_recon + beta * D_KL + alpha * BCE(prediction, outcome)

Uses cached data from the prior extraction run.

Usage:
    python run_supervised_experiment.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_header(step: int, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Step {step}: {title}")
    print(f"{'='*70}\n")


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 70)
    print("  Experiment B: Supervised VAE")
    print("  embedding_dim=8, beta=4.0, alpha=1.0")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Load cached data
    # ---------------------------------------------------------------
    print_header(1, "Load Cached Data")

    data_dir = EXPERIMENT_DIR / "results" / "data"
    features_file = data_dir / "features.npz"
    if not features_file.exists():
        print(f"  ERROR: {features_file} not found. Run the base experiment first.")
        sys.exit(1)

    data = np.load(features_file, allow_pickle=True)
    X_parts, y_parts = [], []
    for split in ("train", "val", "test"):
        X_parts.append(data[f"X_{split}"])
        y_parts.append(data[f"y_{split}"])
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])

    # Handle NaNs
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            median_val = np.nanmedian(X[:, col])
            X[mask, col] = median_val if np.isfinite(median_val) else 0.0

    # Drop zero-variance
    variances = np.var(X, axis=0)
    keep = variances >= 1e-10
    if not keep.all():
        dropped = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
        print(f"  Dropping {len(dropped)} zero-variance features: {dropped}")
        X = X[:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    # Binarize outcome labels
    y_binary = (y >= 0.5).astype(np.float32)

    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Outcome: Yes={int(y_binary.sum())}, No={int((1-y_binary).sum())}")

    # ---------------------------------------------------------------
    # Step 2: Train Supervised VAE
    # ---------------------------------------------------------------
    print_header(2, "Train Supervised VAE (dim=8, beta=4.0, alpha=1.0)")

    from models.autoencoder import AutoencoderConfig, MarketAutoencoder
    from models.train import TrainConfig, train, compute_embedding_stats

    # Prepare data for training module
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Save features + labels for train.load_data
    train_data_dir = EXPERIMENT_DIR / "results" / "supervised_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_data_dir / "features.npy", X_scaled)
    np.save(train_data_dir / "labels.npy", y_binary)
    with open(train_data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    output_dir = EXPERIMENT_DIR / "results" / "supervised_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        model_type="supervised_vae",
        embedding_dim=8,
        hidden_dims=(256, 128),
        dropout=0.1,
        beta=4.0,
        alpha=1.0,
        epochs=300,
        batch_size=128,
        learning_rate=1e-3,
        patience=30,
        data_dir=str(train_data_dir),
        output_dir=str(output_dir),
        seed=42,
    )

    t0 = time.perf_counter()
    result = train(train_config)
    t_train = time.perf_counter() - t0

    embeddings = result["embeddings"]
    history = result["history"]

    print(f"\n  Training complete in {t_train:.1f}s")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val loss: {result['best_val_loss']:.6f}")
    print(f"  Embedding shape: {embeddings.shape}")

    # Final prediction accuracy
    if history["pred_acc"]:
        final_pred_acc = history["pred_acc"][-1]
        best_pred_acc = max(history["pred_acc"])
        print(f"  Final prediction accuracy: {final_pred_acc:.4f}")
        print(f"  Best prediction accuracy:  {best_pred_acc:.4f}")

    Z = embeddings

    # ---------------------------------------------------------------
    # Step 3: Embedding quality analysis
    # ---------------------------------------------------------------
    print_header(3, "Embedding Quality")

    emb_stats = compute_embedding_stats(Z)
    print(f"  Mean activation:         {emb_stats['mean_activation']:.4f}")
    print(f"  Std activation:          {emb_stats['std_activation']:.4f}")
    print(f"  Max inter-dim corr:      {emb_stats['max_inter_dim_correlation']:.4f}")
    print(f"  Mean inter-dim corr:     {emb_stats['mean_inter_dim_correlation']:.4f}")
    print(f"  Dead dimensions:         {emb_stats['dead_dimensions']}/{Z.shape[1]}")

    # ---------------------------------------------------------------
    # Step 4: Multicollinearity comparison (THE KEY TEST)
    # ---------------------------------------------------------------
    print_header(4, "Multicollinearity: Raw vs Supervised VAE Embeddings")

    from models.statistics import (
        compare_multicollinearity,
        compute_vif,
        test_orthogonality,
        test_predictive_power,
    )

    comparison = compare_multicollinearity(X, Z, feature_names)
    print(comparison)

    # ---------------------------------------------------------------
    # Step 5: Wald test on individual dimensions (THE CRITICAL TEST)
    # ---------------------------------------------------------------
    print_header(5, "Per-Dimension Predictive Power (Wald Test)")

    wald_result = test_predictive_power(Z, y_binary.astype(int), alpha=0.05)
    print(wald_result)
    print(f"\n  SIGNIFICANT DIMENSIONS: {wald_result.n_significant}/{len(wald_result.dimension_names)}")

    # Also show all dims sorted by p-value
    print("\n  All dimensions by p-value:")
    sorted_idx = np.argsort(wald_result.p_values)
    for i in sorted_idx:
        sig = "***" if wald_result.p_values[i] < 0.001 else "**" if wald_result.p_values[i] < 0.01 else "*" if wald_result.p_values[i] < 0.05 else ""
        print(f"    dim_{i}: coef={wald_result.coefficients[i]:>8.4f}  "
              f"z={wald_result.z_scores[i]:>7.2f}  "
              f"p={wald_result.p_values[i]:.4e} {sig}")

    # ---------------------------------------------------------------
    # Step 6: Orthogonality test
    # ---------------------------------------------------------------
    print_header(6, "Orthogonality Test")

    orth = test_orthogonality(Z)
    print(orth)

    # ---------------------------------------------------------------
    # Step 7: Linear probes (raw vs supervised embedding)
    # ---------------------------------------------------------------
    print_header(7, "Linear Probes: Raw vs Supervised VAE Embeddings")

    from models.probes import run_standard_probes

    labels = {"outcome": y_binary.astype(int)}
    probe_comparisons = run_standard_probes(
        X_raw=X_scaled, X_embed=Z, labels=labels, n_permutations=100,
    )

    print("\n  Probe comparison:")
    print("  ┌──────────────────────┬───────────┬───────────┬──────────┐")
    print("  │ Concept              │  Raw      │  Embed    │ Delta    │")
    print("  ├──────────────────────┼───────────┼───────────┼──────────┤")
    for comp in probe_comparisons:
        m = comp.primary_metric
        raw_val = comp.raw_result.metrics[m]
        emb_val = comp.embed_result.metrics[m]
        delta = comp.improvement
        sign = "+" if delta >= 0 else ""
        print(f"  │ {comp.concept:<20s} │ {raw_val:>8.4f} │ {emb_val:>8.4f} │ {sign}{delta:>7.4f} │")
    print("  └──────────────────────┴───────────┴───────────┴──────────┘")

    # ---------------------------------------------------------------
    # Step 8: Overall logistic regression comparison
    # ---------------------------------------------------------------
    print_header(8, "Logistic Regression: Raw vs Embedding")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    raw_clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    raw_clf.fit(X_scaled, y_binary.astype(int))
    raw_acc = accuracy_score(y_binary.astype(int), raw_clf.predict(X_scaled))
    raw_auc = roc_auc_score(y_binary, raw_clf.predict_proba(X_scaled)[:, 1])

    emb_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    emb_clf.fit(Z, y_binary.astype(int))
    emb_acc = accuracy_score(y_binary.astype(int), emb_clf.predict(Z))
    emb_auc = roc_auc_score(y_binary, emb_clf.predict_proba(Z)[:, 1])

    print(f"  Raw features ({X.shape[1]}D):  acc={raw_acc:.4f}  AUC={raw_auc:.4f}")
    print(f"  Supervised VAE (8D):   acc={emb_acc:.4f}  AUC={emb_auc:.4f}")

    # ---------------------------------------------------------------
    # Step 9: Visualizations
    # ---------------------------------------------------------------
    print_header(9, "Visualizations")

    figures_dir = EXPERIMENT_DIR / "results" / "supervised_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        from models.visualize import (
            plot_correlation_heatmap,
            plot_embedding_space,
            plot_probe_comparison,
            plot_vif_comparison,
            plot_training_history,
        )
        from models.statistics import compute_vif

        # Correlation heatmaps
        emb_names = [f"emb_{i}" for i in range(Z.shape[1])]
        plot_correlation_heatmap(
            Z, feature_names=emb_names,
            title="Supervised VAE Embeddings",
            output_dir=str(figures_dir),
        )
        plot_correlation_heatmap(
            X, feature_names=feature_names,
            title="Raw Features",
            output_dir=str(figures_dir),
        )

        # VIF comparison
        raw_vif = compute_vif(X, feature_names)
        emb_vif = compute_vif(Z, emb_names)
        plot_vif_comparison(
            raw_vif.vif_values, emb_vif.vif_values,
            raw_names=raw_vif.feature_names, embed_names=emb_vif.feature_names,
            output_dir=str(figures_dir),
        )

        # t-SNE
        plot_embedding_space(
            Z, y_binary.astype(int),
            label_name="outcome", method="tsne",
            output_dir=str(figures_dir),
        )

        # Training history
        plot_training_history(history, output_dir=str(figures_dir))

        # Probe comparison
        if probe_comparisons:
            plot_probe_comparison(probe_comparisons, output_dir=str(figures_dir))

        print(f"  Figures saved to {figures_dir}")
    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")

    # ---------------------------------------------------------------
    # Step 10: VERDICT
    # ---------------------------------------------------------------
    print_header(10, "VERDICT: Supervised VAE Experiment B")

    sig_frac = wald_result.n_significant / len(wald_result.dimension_names)
    vif_pass = comparison.embed_vif.max_vif < 10
    probe_preserved = all(
        comp.embed_result.metrics[comp.primary_metric] >= comp.raw_result.metrics[comp.primary_metric] * 0.95
        for comp in probe_comparisons
    ) if probe_comparisons else True
    sig_pass = sig_frac > 0.3
    cond_improvement = comparison.raw_condition / max(comparison.embed_condition, 1e-6)
    cond_pass = cond_improvement > 5

    criteria = {
        "VIF below 10": vif_pass,
        "Probe accuracy preserved (>= 95% of raw)": probe_preserved,
        f"Significant dimensions > 30% ({wald_result.n_significant}/{len(wald_result.dimension_names)})": sig_pass,
        f"Condition number improved 5x+ ({cond_improvement:.1f}x)": cond_pass,
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print("  Criteria:")
    for crit_name, crit_passed in criteria.items():
        status = "PASS" if crit_passed else "FAIL"
        print(f"    [{status}] {crit_name}")

    print(f"\n  Score: {passed}/{total}")

    if passed == total:
        verdict = "STRONG SUPPORT"
    elif passed >= 3:
        verdict = "MODERATE SUPPORT"
    elif passed >= 2:
        verdict = "WEAK SUPPORT"
    else:
        verdict = "NOT SUPPORTED"

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  Key comparison vs baseline (64D unsupervised VAE):")
    print(f"    Baseline significant dims:   0/64  (0%)")
    print(f"    Supervised VAE significant:  {wald_result.n_significant}/{len(wald_result.dimension_names)} ({sig_frac*100:.0f}%)")
    print(f"    Embedding VIF max:           {comparison.embed_vif.max_vif:.2f}")
    print(f"    Prediction accuracy (embed): {emb_acc:.4f}")

    # Save report
    reports_dir = EXPERIMENT_DIR / "results" / "supervised_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "B_supervised_vae",
        "config": {
            "model_type": "supervised_vae",
            "embedding_dim": 8,
            "beta": 4.0,
            "alpha": 1.0,
            "epochs": 300,
            "patience": 30,
        },
        "verdict": verdict,
        "score": f"{passed}/{total}",
        "criteria": {k: v for k, v in criteria.items()},
        "training": {
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "training_time_s": t_train,
            "final_pred_acc": history["pred_acc"][-1] if history["pred_acc"] else None,
            "best_pred_acc": max(history["pred_acc"]) if history["pred_acc"] else None,
        },
        "wald_test": {
            "significant_dims": wald_result.n_significant,
            "total_dims": len(wald_result.dimension_names),
            "fraction": sig_frac,
            "per_dim": {
                f"dim_{i}": {
                    "coef": float(wald_result.coefficients[i]),
                    "z_score": float(wald_result.z_scores[i]),
                    "p_value": float(wald_result.p_values[i]),
                }
                for i in range(len(wald_result.dimension_names))
            },
        },
        "multicollinearity": {
            "raw_max_vif": comparison.raw_vif.max_vif,
            "raw_mean_vif": comparison.raw_vif.mean_vif,
            "embed_max_vif": comparison.embed_vif.max_vif,
            "embed_mean_vif": comparison.embed_vif.mean_vif,
            "raw_condition": comparison.raw_condition,
            "embed_condition": comparison.embed_condition,
            "raw_max_corr": comparison.raw_max_corr,
            "embed_max_corr": comparison.embed_max_corr,
        },
        "prediction": {
            "raw_accuracy": raw_acc,
            "raw_auc": raw_auc,
            "embed_accuracy": emb_acc,
            "embed_auc": emb_auc,
        },
        "orthogonality": {
            "mean_cosine_sim": orth.mean_off_diagonal,
            "max_cosine_sim": orth.max_off_diagonal,
            "n_correlated_pairs": orth.n_correlated_pairs,
        },
        "embedding_quality": emb_stats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    report_path = reports_dir / "supervised_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.perf_counter() - total_start

    print(f"\n{'='*70}")
    print(f"  Experiment B complete in {elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print(f"  Figures: {figures_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
