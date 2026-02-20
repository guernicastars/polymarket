#!/usr/bin/env python3
"""Honest experiment: Over/Under markets with non-leaking features only.

Addresses two validity concerns from the original experiment:
1. Class imbalance: Yes/No markets have 78% majority class.
   Over/Under markets are ~50/50, making accuracy meaningful.
2. Feature leakage: price/volume features observed AFTER the outcome
   is determined (last_price, price_range, etc.) leak the answer.
   We keep only 13 structural features available BEFORE resolution.

Safe features (13):
  - volume_total, neg_risk, market_duration_days
  - trade_count, avg_trade_size, max_trade_size, trade_size_gini, trades_per_day
  - volume_vs_category_median
  - unique_wallet_count, top_wallet_concentration, avg_insider_score
  - num_outcomes

Usage:
    python run_honest_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
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

from models.autoencoder import AutoencoderConfig, MarketAutoencoder
from models.statistics import (
    compute_condition_number,
    compute_vif,
    test_orthogonality,
    test_predictive_power,
)
from models.train import TrainConfig, train as train_model


# ======================================================================
# Safe feature list
# ======================================================================

SAFE_FEATURES = [
    "volume_total",
    "neg_risk",
    "market_duration_days",
    "trade_count",
    "avg_trade_size",
    "max_trade_size",
    "trade_size_gini",
    "trades_per_day",
    "volume_vs_category_median",
    "unique_wallet_count",
    "top_wallet_concentration",
    "avg_insider_score",
    "num_outcomes",
]


# ======================================================================
# Data extraction from ClickHouse
# ======================================================================

def extract_over_under_data(min_trades: int = 3) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Extract Over/Under markets with safe features from ClickHouse.

    Returns (X, y, feature_names, metadata).
    """
    import clickhouse_connect
    from data.config import (
        CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
        CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE,
    )
    from data.features import (
        extract_market_structure,
        extract_trade_microstructure,
        extract_volume_dynamics,
        extract_wallet_features,
    )

    print("  Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE, secure=True,
        compress="lz4", connect_timeout=30, send_receive_timeout=300,
    )

    # Fetch resolved markets with Over/Under outcomes
    query = """
    SELECT
        m.condition_id,
        m.question,
        m.outcomes,
        m.winning_outcome,
        m.neg_risk,
        m.volume_total,
        m.volume_24h,
        m.volume_1wk,
        m.liquidity,
        m.one_day_price_change,
        m.one_week_price_change,
        m.start_date,
        m.end_date,
        m.category,
        m.token_ids,
        tc.n_trades
    FROM (
        SELECT *
        FROM markets FINAL
        WHERE resolved = 1
          AND winning_outcome != ''
          AND length(outcomes) = 2
    ) AS m
    INNER JOIN (
        SELECT condition_id, count() AS n_trades
        FROM market_trades
        GROUP BY condition_id
        HAVING n_trades >= {min_trades:UInt32}
    ) AS tc ON m.condition_id = tc.condition_id
    ORDER BY m.end_date
    """
    result = client.query(query, parameters={"min_trades": min_trades})
    import pandas as pd
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    print(f"  Total resolved binary markets with >= {min_trades} trades: {len(df)}")

    # Filter to Over/Under only
    def is_over_under(outcomes):
        if not outcomes or len(outcomes) != 2:
            return False
        o = [str(x).lower().strip() for x in outcomes]
        return ("over" in o[0] or "under" in o[0] or
                "over" in o[1] or "under" in o[1])

    over_under_mask = df["outcomes"].apply(is_over_under)
    df_ou = df[over_under_mask].reset_index(drop=True)
    print(f"  Over/Under markets: {len(df_ou)}")

    if len(df_ou) < 20:
        print(f"  WARNING: Very few Over/Under markets ({len(df_ou)}). Falling back to ALL markets.")
        df_ou = df.reset_index(drop=True)
        is_ou_only = False
    else:
        is_ou_only = True

    # Pre-compute category median volumes
    cat_query = """
    SELECT category, median(volume_total) AS med_vol
    FROM markets FINAL
    WHERE resolved = 1 AND winning_outcome != ''
    GROUP BY category
    """
    cat_result = client.query(cat_query)
    cat_medians = {row[0]: float(row[1]) for row in cat_result.result_rows}

    # Batch fetch wallet data
    all_cids = df_ou["condition_id"].tolist()
    from data.extract import fetch_wallet_data_batch
    wallet_data_all: dict = {}
    for i in range(0, len(all_cids), 500):
        chunk = all_cids[i:i + 500]
        wallet_data_all.update(fetch_wallet_data_batch(client, chunk))
    print(f"  Wallet data for {len(wallet_data_all)}/{len(all_cids)} markets")

    # Extract features
    all_features = []
    all_outcomes = []
    valid_indices = []

    for i, row in df_ou.iterrows():
        # Encode outcome
        winning = str(row.get("winning_outcome", ""))
        outcomes = row.get("outcomes", [])
        if not outcomes or not winning:
            continue
        if winning == outcomes[0]:
            y_val = 1.0
        elif winning == outcomes[1]:
            y_val = 0.0
        else:
            continue

        # Fetch trades
        cid = row["condition_id"]
        trade_query = """
        SELECT price, size, side, timestamp
        FROM market_trades
        WHERE condition_id = {cid:String}
        ORDER BY timestamp
        """
        trade_result = client.query(trade_query, parameters={"cid": cid})
        trades_df = pd.DataFrame(trade_result.result_rows, columns=trade_result.column_names)

        # Build features using only safe extractors
        market_meta = row.to_dict()
        fv = {}
        fv.update(extract_market_structure(market_meta))
        fv.update(extract_trade_microstructure(trades_df, market_meta))
        cat = row.get("category", "")
        cat_median = cat_medians.get(cat, 0.0)
        fv.update(extract_volume_dynamics(market_meta, cat_median))
        wallet_data = wallet_data_all.get(cid)
        fv.update(extract_wallet_features(wallet_data or {}))

        all_features.append(fv)
        all_outcomes.append(y_val)
        valid_indices.append(i)

        if (len(all_features)) % 200 == 0:
            print(f"    Processed {len(all_features)} markets...")

    print(f"  Extracted features for {len(all_features)} markets")

    # Build matrix with SAFE features only
    feature_names = [f for f in SAFE_FEATURES if any(f in fv for fv in all_features)]
    X = np.array(
        [[fv.get(fn, np.nan) for fn in feature_names] for fv in all_features],
        dtype=np.float64,
    )
    y = np.array(all_outcomes, dtype=np.float64)

    # Class balance
    n_pos = (y == 1.0).sum()
    n_neg = (y == 0.0).sum()
    print(f"  Features: {len(feature_names)}, Samples: {len(y)}")
    print(f"  Class balance: Over/Yes={n_pos} ({100*n_pos/len(y):.1f}%), Under/No={n_neg} ({100*n_neg/len(y):.1f}%)")

    metadata = {
        "n_markets": len(y),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "is_over_under_only": is_ou_only,
        "class_balance": {"positive": int(n_pos), "negative": int(n_neg)},
        "nan_fraction": float(np.isnan(X).mean()),
        "wallet_coverage": f"{len(wallet_data_all)}/{len(all_cids)}",
    }

    return X, y, feature_names, metadata


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_representation(
    Z: np.ndarray,
    y_binary: np.ndarray,
    label: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Full evaluation: VIF, condition#, orthogonality, Wald, CV probes."""
    dim = Z.shape[1]
    dim_names = [f"dim_{i}" for i in range(dim)]

    vif = compute_vif(Z, dim_names)
    cond = compute_condition_number(Z)
    orth = test_orthogonality(Z)
    wald = test_predictive_power(Z, y_binary, alpha=0.05)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, fold_bacc, fold_f1_pos, fold_f1_neg, fold_auc = [], [], [], [], []

    for train_idx, test_idx in cv.split(Z, y_binary):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(Z[train_idx], y_binary[train_idx])
        y_pred = clf.predict(Z[test_idx])
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        fold_acc.append(accuracy_score(y_binary[test_idx], y_pred))
        fold_bacc.append(balanced_accuracy_score(y_binary[test_idx], y_pred))
        fold_f1_pos.append(f1_score(y_binary[test_idx], y_pred, pos_label=1))
        fold_f1_neg.append(f1_score(y_binary[test_idx], y_pred, pos_label=0))
        fold_auc.append(roc_auc_score(y_binary[test_idx], y_prob))

    return {
        "label": label,
        "dims": dim,
        "vif_mean": vif.mean_vif,
        "vif_max": vif.max_vif,
        "condition_number": cond,
        "orth_max_cos": orth.max_off_diagonal,
        "wald_significant": wald.n_significant,
        "wald_total": len(wald.dimension_names),
        "wald_fraction": wald.n_significant / max(len(wald.dimension_names), 1),
        "cv_accuracy": float(np.mean(fold_acc)),
        "cv_balanced_accuracy": float(np.mean(fold_bacc)),
        "cv_f1_pos": float(np.mean(fold_f1_pos)),
        "cv_f1_neg": float(np.mean(fold_f1_neg)),
        "cv_auc": float(np.mean(fold_auc)),
    }


def evaluate_oos(
    Z_train: np.ndarray, Z_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    label: str,
) -> dict:
    """Out-of-sample: train on train, predict test."""
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    y_prob = clf.predict_proba(Z_test)[:, 1]

    return {
        "label": label,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_pos": float(f1_score(y_test, y_pred, pos_label=1)),
        "f1_neg": float(f1_score(y_test, y_pred, pos_label=0)),
        "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
    }


# ======================================================================
# Printing
# ======================================================================

def print_table(title: str, results: list[dict], rows: list[tuple]) -> None:
    """Print a comparison table."""
    labels = [r["label"] for r in results]
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")

    header = f"  {'Metric':<28s}"
    for lbl in labels:
        header += f" {lbl:>16s}"
    print(header)
    print("  " + "-" * (28 + 17 * len(labels)))

    for name, key, fmt in rows:
        line = f"  {name:<28s}"
        for r in results:
            val = r.get(key, "N/A")
            if val == "N/A":
                line += f" {'N/A':>16s}"
            elif fmt == "d":
                line += f" {int(val):>16d}"
            elif fmt == ".0%":
                line += f" {val:>15.0%} "
            else:
                line += f" {val:>16{fmt}}"
        print(line)

    print("  " + "-" * (28 + 17 * len(labels)))


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    print("=" * 70)
    print("  THE HONEST EXPERIMENT")
    print("  Over/Under markets + non-leaking features only")
    print("=" * 70)

    total_start = time.perf_counter()
    output_dir = EXPERIMENT_DIR / "results" / "honest_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Extract data ----
    print("\n--- STEP 1: Data Extraction ---")
    X, y, feature_names, metadata = extract_over_under_data(min_trades=3)
    y_binary = (y >= 0.5).astype(int)
    n_samples, n_features = X.shape

    # Impute NaNs
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  NaN values: {nan_count}. Imputing with column medians.")
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
        metadata["feature_names"] = feature_names
        metadata["dropped_features"] = dropped

    n_features = len(feature_names)
    print(f"  Final: {n_samples} samples, {n_features} features")
    print(f"  Features: {feature_names}")

    # ---- Temporal split (70/15/15) ----
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.85)
    # Data is already sorted by end_date from the query
    X_train, y_train = X[:n_train], y_binary[:n_train]
    X_val, y_val = X[n_train:n_val], y_binary[n_train:n_val]
    X_test, y_test = X[n_val:], y_binary[n_val:]

    print(f"  Split: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"  Train balance: pos={y_train.sum()}, neg={len(y_train)-y_train.sum()}")
    print(f"  Test balance:  pos={y_test.sum()}, neg={len(y_test)-y_test.sum()}")

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X)

    # ---- Step 2: Train Orth-SVAE ----
    print("\n--- STEP 2: Train Orth-Supervised-VAE (dim=8, beta=1, alpha=1, gamma=1) ---")

    # Save scaled data + labels for training module
    data_dir = output_dir / "train_data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "features.npy", X_all_sc.astype(np.float32))
    np.save(data_dir / "labels.npy", y_binary.astype(np.float32))
    with open(data_dir / "metadata.json", "w") as f:
        json.dump({"feature_names": feature_names, "n_features": n_features}, f)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_config = TrainConfig(
        model_type="supervised_vae",
        embedding_dim=8,
        hidden_dims=(256, 128),
        dropout=0.1,
        beta=1.0,
        alpha=1.0,
        gamma=1.0,
        epochs=300,
        batch_size=128,
        learning_rate=0.001,
        patience=30,
        data_dir=str(data_dir),
        output_dir=str(ckpt_dir),
    )

    result = train_model(train_config)
    print(f"  Best epoch: {result['best_epoch']}, val loss: {result['best_val_loss']:.6f}")

    # Load best model
    model_config = AutoencoderConfig(
        input_dim=n_features, embedding_dim=8,
        hidden_dims=(256, 128), dropout=0.1,
        model_type="supervised_vae", beta=1.0, alpha=1.0, gamma=1.0,
    )
    model = MarketAutoencoder(model_config)
    model.load_state_dict(torch.load(ckpt_dir / "best_model_supervised_vae.pt", weights_only=True))
    model.eval()

    # Encode all splits
    with torch.no_grad():
        Z_train_svae = model.get_embedding(torch.tensor(X_train_sc, dtype=torch.float32))
        Z_val_svae = model.get_embedding(torch.tensor(X_val_sc, dtype=torch.float32))
        Z_test_svae = model.get_embedding(torch.tensor(X_test_sc, dtype=torch.float32))
        Z_all_svae = model.get_embedding(torch.tensor(X_all_sc, dtype=torch.float32))

    # PCA
    pca = PCA(n_components=min(8, n_features), random_state=42)
    Z_train_pca = pca.fit_transform(X_train_sc)
    Z_val_pca = pca.transform(X_val_sc)
    Z_test_pca = pca.transform(X_test_sc)
    Z_all_pca = pca.transform(X_all_sc)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # ---- Step 3: Full analysis (cross-validated on ALL data) ----
    print("\n--- STEP 3: Cross-Validated Analysis (All Data) ---")

    raw_cv = evaluate_representation(X_all_sc, y_binary, f"Raw ({n_features}D)")
    pca_cv = evaluate_representation(Z_all_pca, y_binary, f"PCA ({Z_all_pca.shape[1]}D)")
    svae_cv = evaluate_representation(Z_all_svae, y_binary, "Orth-SVAE (8D)")

    cv_rows = [
        ("Dimensions", "dims", "d"),
        ("Mean VIF", "vif_mean", ".2f"),
        ("Max VIF", "vif_max", ".2f"),
        ("Condition Number", "condition_number", ".1f"),
        ("Sig Dims (Wald)", "wald_significant", "d"),
        ("Sig Fraction", "wald_fraction", ".0%"),
        ("CV Accuracy", "cv_accuracy", ".4f"),
        ("CV Balanced Accuracy", "cv_balanced_accuracy", ".4f"),
        ("CV F1 (Over/Yes)", "cv_f1_pos", ".4f"),
        ("CV F1 (Under/No)", "cv_f1_neg", ".4f"),
        ("CV AUC-ROC", "cv_auc", ".4f"),
    ]
    print_table("CROSS-VALIDATED RESULTS (5-fold, all data)", [raw_cv, pca_cv, svae_cv], cv_rows)

    # ---- Step 4: Out-of-sample test ----
    print("\n--- STEP 4: Out-of-Sample Test (train on train, predict test) ---")

    oos_raw = evaluate_oos(X_train_sc, X_test_sc, y_train, y_test, f"Raw ({n_features}D)")
    oos_pca = evaluate_oos(Z_train_pca, Z_test_pca, y_train, y_test, f"PCA ({Z_all_pca.shape[1]}D)")
    oos_svae = evaluate_oos(Z_train_svae, Z_test_svae, y_train, y_test, "Orth-SVAE (8D)")

    oos_rows = [
        ("Accuracy", "accuracy", ".4f"),
        ("Balanced Accuracy", "balanced_accuracy", ".4f"),
        ("F1 (Over/Yes)", "f1_pos", ".4f"),
        ("F1 (Under/No)", "f1_neg", ".4f"),
        ("AUC-ROC", "auc", ".4f"),
    ]
    print_table("OUT-OF-SAMPLE TEST SET RESULTS", [oos_raw, oos_pca, oos_svae], oos_rows)

    # ---- Step 5: Walk-forward ----
    print("\n--- STEP 5: Walk-Forward Temporal Backtest ---")

    min_train_size = max(50, int(n_samples * 0.3))
    step_size = max(10, int(n_samples * 0.05))
    print(f"  min_train={min_train_size}, step={step_size}")

    raw_wf, pca_wf, svae_wf = [], [], []
    for start in range(min_train_size, n_samples - step_size, step_size):
        test_end = min(start + step_size, n_samples)
        X_tr, y_tr = X[:start], y_binary[:start]
        X_te, y_te = X[start:test_end], y_binary[start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # Raw
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr_sc, y_tr)
        raw_wf.append(accuracy_score(y_te, clf.predict(X_te_sc)))

        # PCA
        pc = PCA(n_components=min(8, n_features), random_state=42)
        Z_tr = pc.fit_transform(X_tr_sc)
        Z_te = pc.transform(X_te_sc)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(Z_tr, y_tr)
        pca_wf.append(accuracy_score(y_te, clf.predict(Z_te)))

        # Orth-SVAE
        with torch.no_grad():
            Z_tr_s = model.get_embedding(torch.tensor(X_tr_sc, dtype=torch.float32))
            Z_te_s = model.get_embedding(torch.tensor(X_te_sc, dtype=torch.float32))
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(Z_tr_s, y_tr)
        svae_wf.append(accuracy_score(y_te, clf.predict(Z_te_s)))

    if raw_wf:
        print(f"  Windows: {len(raw_wf)}")
        print(f"  Raw mean:       {np.mean(raw_wf):.4f}")
        print(f"  PCA mean:       {np.mean(pca_wf):.4f}")
        print(f"  Orth-SVAE mean: {np.mean(svae_wf):.4f}")
    else:
        print("  Not enough data for walk-forward.")

    # ---- Plot ----
    if raw_wf:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(raw_wf))
            ax.plot(x, raw_wf, label=f"Raw (mean={np.mean(raw_wf):.3f})", alpha=0.8)
            ax.plot(x, pca_wf, label=f"PCA (mean={np.mean(pca_wf):.3f})", alpha=0.8)
            ax.plot(x, svae_wf, label=f"Orth-SVAE (mean={np.mean(svae_wf):.3f})", alpha=0.8)
            majority = max(y_binary.mean(), 1 - y_binary.mean())
            ax.axhline(y=majority, color="gray", linestyle="--", alpha=0.5, label=f"Majority class ({majority:.1%})")
            ax.set_xlabel("Window")
            ax.set_ylabel("Accuracy")
            ax.set_title("Honest Experiment: Walk-Forward Accuracy (Over/Under, Non-Leaking Features)")
            ax.legend()
            ax.set_ylim(0.3, 1.0)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "walk_forward.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {output_dir / 'walk_forward.png'}")
        except Exception as e:
            print(f"  Plot failed: {e}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print("  HONEST EXPERIMENT VERDICT")
    print(f"{'='*70}")

    baseline = max(y_binary.mean(), 1 - y_binary.mean())
    print(f"\n  Majority class baseline: {baseline:.1%}")
    print(f"  Raw CV accuracy:       {raw_cv['cv_accuracy']:.4f} (above baseline: {raw_cv['cv_accuracy'] > baseline + 0.01})")
    print(f"  PCA CV accuracy:       {pca_cv['cv_accuracy']:.4f}")
    print(f"  Orth-SVAE CV accuracy: {svae_cv['cv_accuracy']:.4f}")
    print(f"  Orth-SVAE OOS accuracy: {oos_svae['accuracy']:.4f}")

    svae_above_baseline = svae_cv['cv_accuracy'] - baseline
    svae_above_pca = svae_cv['cv_accuracy'] - pca_cv['cv_accuracy']

    if svae_above_baseline > 0.05:
        verdict = "REAL SIGNAL DETECTED"
        print(f"\n  VERDICT: {verdict}")
        print(f"  Orth-SVAE is {svae_above_baseline:.1%} above majority class baseline.")
        if svae_above_pca > 0.02:
            print(f"  Neural net adds +{svae_above_pca:.1%} over PCA. Justified.")
        else:
            print(f"  Neural net adds only +{svae_above_pca:.1%} over PCA. Marginal.")
    elif svae_above_baseline > 0.02:
        verdict = "WEAK SIGNAL"
        print(f"\n  VERDICT: {verdict}")
        print(f"  Orth-SVAE is {svae_above_baseline:.1%} above baseline. Detectable but small.")
    else:
        verdict = "NO SIGNAL"
        print(f"\n  VERDICT: {verdict}")
        print(f"  Orth-SVAE is only {svae_above_baseline:.1%} above baseline. No real signal.")

    # ---- Save ----
    report = {
        "experiment": "honest_experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": metadata,
        "safe_features": feature_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "majority_baseline": float(baseline),
        "cv_results": {
            "raw": raw_cv,
            "pca": pca_cv,
            "orth_svae": svae_cv,
        },
        "oos_results": {
            "raw": oos_raw,
            "pca": oos_pca,
            "orth_svae": oos_svae,
        },
        "walk_forward": {
            "raw_mean": float(np.mean(raw_wf)) if raw_wf else 0.0,
            "pca_mean": float(np.mean(pca_wf)) if pca_wf else 0.0,
            "svae_mean": float(np.mean(svae_wf)) if svae_wf else 0.0,
            "n_windows": len(raw_wf),
        },
        "verdict": verdict,
    }

    report_path = output_dir / "honest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.perf_counter() - total_start
    print(f"\n  Report: {report_path}")
    print(f"  Completed in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
