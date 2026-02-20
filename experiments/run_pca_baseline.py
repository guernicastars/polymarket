#!/usr/bin/env python3
"""PCA baseline comparison: does the neural net actually add value?

Compares three representations side-by-side:
  1. Raw features (25D)
  2. PCA (8D) — linear dimensionality reduction baseline
  3. Best VAE (8D, beta=1.0) — unsupervised nonlinear
  4. Orth-Supervised-VAE (8D, b1_a1_g1) — best overall

PCA gives orthogonal components by definition, so VIF will be ~1.0.
The question is whether it matches the VAE's probe accuracy and
per-dimension significance.

Usage:
    python run_pca_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from models.statistics import (
    compute_condition_number,
    compute_vif,
    test_orthogonality,
    test_predictive_power,
)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached features, impute NaNs, drop zero-variance."""
    data_dir = EXPERIMENT_DIR / "results" / "data"
    data = np.load(data_dir / "features.npz", allow_pickle=True)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Reassemble splits
    X_parts, y_parts = [], []
    for split in ("train", "val", "test"):
        xk, yk = f"X_{split}", f"y_{split}"
        if xk in data:
            X_parts.append(data[xk])
            y_parts.append(data[yk])
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    # Feature names
    if "feature_names" in data:
        fn = data["feature_names"]
        feature_names = fn.tolist() if fn.dtype.kind in ("U", "S", "O") else metadata.get("feature_names", [])
    else:
        feature_names = metadata.get("feature_names", [f"f_{i}" for i in range(X.shape[1])])

    # Impute NaNs
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                med = np.nanmedian(X[:, col])
                X[mask, col] = med if np.isfinite(med) else 0.0

    # Drop zero-variance
    variances = np.var(X, axis=0)
    keep = variances >= 1e-10
    if not keep.all():
        dropped = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
        print(f"  Dropped {len(dropped)} zero-variance features: {dropped}")
        X = X[:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    return X, y, feature_names


def evaluate_representation(
    Z: np.ndarray,
    y_binary: np.ndarray,
    label: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run the full evaluation battery on a representation.

    Returns dict with VIF, condition number, orthogonality, Wald test,
    probe metrics (accuracy, balanced accuracy, per-class F1, AUC).
    """
    dim = Z.shape[1]
    dim_names = [f"dim_{i}" for i in range(dim)]

    # --- VIF ---
    vif = compute_vif(Z, dim_names)

    # --- Condition number ---
    cond = compute_condition_number(Z)

    # --- Orthogonality ---
    orth = test_orthogonality(Z)

    # --- Wald test ---
    wald = test_predictive_power(Z, y_binary, alpha=0.05)

    # --- Cross-validated probes (accuracy, balanced accuracy, F1, AUC) ---
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, fold_bacc, fold_f1_yes, fold_f1_no, fold_auc = [], [], [], [], []

    for train_idx, test_idx in cv.split(Z, y_binary):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(Z[train_idx], y_binary[train_idx])
        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        fold_acc.append(accuracy_score(y_binary[test_idx], y_pred))
        fold_bacc.append(balanced_accuracy_score(y_binary[test_idx], y_pred))
        fold_f1_yes.append(f1_score(y_binary[test_idx], y_pred, pos_label=1))
        fold_f1_no.append(f1_score(y_binary[test_idx], y_pred, pos_label=0))
        fold_auc.append(roc_auc_score(y_binary[test_idx], y_prob))

    # --- Full-data logistic regression (for Wald-compatible comparison) ---
    clf_full = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf_full.fit(Z, y_binary)
    full_acc = accuracy_score(y_binary, clf_full.predict(Z))
    full_auc = roc_auc_score(y_binary, clf_full.predict_proba(Z)[:, 1])

    return {
        "label": label,
        "dims": dim,
        "vif_mean": vif.mean_vif,
        "vif_max": vif.max_vif,
        "vif_severe": vif.n_severe,
        "condition_number": cond,
        "orth_mean_cos": orth.mean_off_diagonal,
        "orth_max_cos": orth.max_off_diagonal,
        "orth_corr_pairs": orth.n_correlated_pairs,
        "wald_significant": wald.n_significant,
        "wald_total": len(wald.dimension_names),
        "wald_fraction": wald.n_significant / max(len(wald.dimension_names), 1),
        "cv_accuracy": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_balanced_accuracy": float(np.mean(fold_bacc)),
        "cv_balanced_accuracy_std": float(np.std(fold_bacc)),
        "cv_f1_yes": float(np.mean(fold_f1_yes)),
        "cv_f1_no": float(np.mean(fold_f1_no)),
        "cv_auc": float(np.mean(fold_auc)),
        "cv_auc_std": float(np.std(fold_auc)),
        "full_accuracy": full_acc,
        "full_auc": full_auc,
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("  SIDE-BY-SIDE COMPARISON: Raw vs PCA vs VAE vs Orth-Supervised-VAE")
    print("=" * 90)

    # Header
    labels = [r["label"] for r in results]
    header = f"{'Metric':<30s}"
    for lbl in labels:
        header += f" {lbl:>12s}"
    print(f"\n{header}")
    print("-" * (30 + 13 * len(labels)))

    rows = [
        ("Dimensions", "dims", "d"),
        ("Mean VIF", "vif_mean", ".2f"),
        ("Max VIF", "vif_max", ".2f"),
        ("Severe VIF (>10)", "vif_severe", "d"),
        ("Condition Number", "condition_number", ".1f"),
        ("Max |cos_sim|", "orth_max_cos", ".4f"),
        ("Sig Dims (Wald)", "wald_significant", "d"),
        ("Sig Fraction", "wald_fraction", ".1%"),
        ("CV Accuracy", "cv_accuracy", ".4f"),
        ("CV Balanced Acc", "cv_balanced_accuracy", ".4f"),
        ("CV F1 (Yes)", "cv_f1_yes", ".4f"),
        ("CV F1 (No)", "cv_f1_no", ".4f"),
        ("CV AUC", "cv_auc", ".4f"),
        ("Full Accuracy", "full_accuracy", ".4f"),
        ("Full AUC", "full_auc", ".4f"),
    ]

    for name, key, fmt in rows:
        line = f"  {name:<28s}"
        for r in results:
            val = r.get(key, "N/A")
            if val == "N/A":
                line += f" {'N/A':>12s}"
            elif fmt == "d":
                line += f" {int(val):>12d}"
            elif fmt == ".1%":
                line += f" {val:>11.1%} "
            else:
                line += f" {val:>12{fmt}}"
        print(line)

    print("-" * (30 + 13 * len(labels)))


def main() -> None:
    print("=" * 70)
    print("  PCA Baseline Comparison")
    print("  Does the neural network add value over linear PCA?")
    print("=" * 70)

    total_start = time.perf_counter()

    # --- Load data ---
    print("\n  Loading cached data...")
    X, y, feature_names = load_data()
    y_binary = (y >= 0.5).astype(int)
    n_samples, n_features = X.shape
    print(f"  Samples: {n_samples}, Features: {n_features}")
    print(f"  Class balance: Yes={y_binary.sum()}, No={n_samples - y_binary.sum()}")

    # Standardize for PCA and raw probes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 1. Raw features ---
    print("\n  [1/4] Evaluating raw features (scaled)...")
    raw_result = evaluate_representation(X_scaled, y_binary, "Raw (25D)")

    # --- 2. PCA ---
    n_components = 8
    print(f"\n  [2/4] Fitting PCA(n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    Z_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"  Explained variance: {explained.sum():.1%} total")
    for i, ev in enumerate(explained):
        print(f"    PC{i}: {ev:.1%}")
    pca_result = evaluate_representation(Z_pca, y_binary, "PCA (8D)")

    # --- 3. Best VAE (dim=8, beta=1.0) ---
    print("\n  [3/4] Loading best VAE embeddings (dim=8, beta=1.0)...")
    vae_emb_path = EXPERIMENT_DIR / "results" / "checkpoints" / "embeddings.npy"
    if vae_emb_path.exists():
        Z_vae = np.load(vae_emb_path)
        if Z_vae.shape[1] != 8:
            print(f"  WARNING: VAE embeddings are {Z_vae.shape[1]}D, not 8D. Skipping.")
            vae_result = None
        else:
            print(f"  Loaded VAE embeddings: {Z_vae.shape}")
            vae_result = evaluate_representation(Z_vae, y_binary, "VAE (8D)")
    else:
        print(f"  WARNING: {vae_emb_path} not found. Skipping VAE comparison.")
        vae_result = None

    # --- 4. Best Orth-Supervised-VAE (b1_a1_g1) ---
    print("\n  [4/4] Loading Orth-Supervised-VAE embeddings (b1_a1_g1)...")
    orth_emb_path = EXPERIMENT_DIR / "results" / "orth_b1_a1_g1_ckpt" / "embeddings.npy"
    if orth_emb_path.exists():
        Z_orth = np.load(orth_emb_path)
        print(f"  Loaded Orth-SVAE embeddings: {Z_orth.shape}")
        orth_result = evaluate_representation(Z_orth, y_binary, "Orth-SVAE (8D)")
    else:
        print(f"  WARNING: {orth_emb_path} not found. Skipping Orth-SVAE comparison.")
        orth_result = None

    # --- Build comparison ---
    results = [raw_result, pca_result]
    if vae_result:
        results.append(vae_result)
    if orth_result:
        results.append(orth_result)

    print_comparison_table(results)

    # --- Key questions ---
    print("\n" + "=" * 70)
    print("  KEY QUESTIONS")
    print("=" * 70)

    # Q1: Does PCA match VAE on VIF?
    print(f"\n  Q1: Does PCA achieve low VIF?")
    print(f"      PCA max VIF: {pca_result['vif_max']:.4f} (always ~1.0 by construction)")
    if vae_result:
        print(f"      VAE max VIF: {vae_result['vif_max']:.2f}")
    if orth_result:
        print(f"      Orth-SVAE max VIF: {orth_result['vif_max']:.4f}")

    # Q2: Does PCA match probe accuracy?
    print(f"\n  Q2: Does PCA match probe accuracy?")
    print(f"      Raw accuracy:  {raw_result['cv_accuracy']:.4f}")
    print(f"      PCA accuracy:  {pca_result['cv_accuracy']:.4f}")
    if vae_result:
        print(f"      VAE accuracy:  {vae_result['cv_accuracy']:.4f}")
    if orth_result:
        print(f"      Orth-SVAE accuracy: {orth_result['cv_accuracy']:.4f}")

    # Q3: Per-dimension significance?
    print(f"\n  Q3: Does PCA have individually significant dimensions?")
    print(f"      PCA sig dims:  {pca_result['wald_significant']}/{pca_result['wald_total']} ({pca_result['wald_fraction']:.0%})")
    if vae_result:
        print(f"      VAE sig dims:  {vae_result['wald_significant']}/{vae_result['wald_total']} ({vae_result['wald_fraction']:.0%})")
    if orth_result:
        print(f"      Orth-SVAE sig dims: {orth_result['wald_significant']}/{orth_result['wald_total']} ({orth_result['wald_fraction']:.0%})")

    # Q4: Balanced accuracy (class imbalance)
    print(f"\n  Q4: Balanced accuracy (handling 78/22 class imbalance)?")
    print(f"      Raw balanced acc:  {raw_result['cv_balanced_accuracy']:.4f}")
    print(f"      PCA balanced acc:  {pca_result['cv_balanced_accuracy']:.4f}")
    if vae_result:
        print(f"      VAE balanced acc:  {vae_result['cv_balanced_accuracy']:.4f}")
    if orth_result:
        print(f"      Orth-SVAE balanced acc: {orth_result['cv_balanced_accuracy']:.4f}")

    # --- Verdict ---
    print(f"\n{'='*70}")
    print("  VERDICT: Does the neural net add value over PCA?")
    print(f"{'='*70}")

    if orth_result:
        pca_acc = pca_result["cv_accuracy"]
        orth_acc = orth_result["cv_accuracy"]
        pca_sig = pca_result["wald_fraction"]
        orth_sig = orth_result["wald_fraction"]
        pca_bacc = pca_result["cv_balanced_accuracy"]
        orth_bacc = orth_result["cv_balanced_accuracy"]

        acc_delta = orth_acc - pca_acc
        bacc_delta = orth_bacc - pca_bacc
        sig_delta = orth_sig - pca_sig

        print(f"\n  Accuracy advantage (Orth-SVAE vs PCA): {acc_delta:+.4f}")
        print(f"  Balanced accuracy advantage:            {bacc_delta:+.4f}")
        print(f"  Significance advantage:                 {sig_delta:+.0%}")

        if acc_delta > 0.02 and sig_delta > 0.2:
            print("\n  YES - The orthogonal supervised VAE substantially outperforms PCA.")
            print("  The nonlinear + supervised signal provides meaningful improvement.")
            verdict = "NEURAL NET ADDS VALUE"
        elif acc_delta > 0.01 or sig_delta > 0.1:
            print("\n  MARGINAL - Small advantage for the neural net over PCA.")
            verdict = "MARGINAL ADVANTAGE"
        else:
            print("\n  NO - PCA matches the neural net. Occam's razor favors PCA.")
            verdict = "PCA SUFFICIENT"
    else:
        verdict = "INCOMPLETE (no Orth-SVAE to compare)"
        print(f"\n  {verdict}")

    # --- Save results ---
    output_dir = EXPERIMENT_DIR / "results" / "pca_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "pca_baseline_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": n_samples,
        "n_features": n_features,
        "pca_explained_variance": explained.tolist(),
        "pca_total_explained": float(explained.sum()),
        "results": {r["label"]: r for r in results},
        "verdict": verdict,
    }

    report_path = output_dir / "pca_baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save PCA embeddings for downstream use
    np.save(output_dir / "pca_embeddings.npy", Z_pca)

    elapsed = time.perf_counter() - total_start
    print(f"\n  Report: {report_path}")
    print(f"  PCA embeddings: {output_dir / 'pca_embeddings.npy'}")
    print(f"  Completed in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
