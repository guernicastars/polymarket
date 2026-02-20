#!/usr/bin/env python3
"""Regression precision analysis: how narrow are the predictions?

Loads the 29K Sotheby's dataset (no estimates), trains Orth-SVAE fresh,
then compares Ridge/RF/LGBM regression on raw features vs SVAE embeddings.

Reports R2, RMSE, MAE in log and dollar terms, MdAPE, and 4-class bucket accuracy.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from models.autoencoder import AutoencoderConfig, MarketAutoencoder
from models.train import TrainConfig, train as train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SILVER_DB = Path("/Users/ivrejchik/Desktop/art/data_enrich/data/silver.db")
GOLD_DB = Path("/Users/ivrejchik/Desktop/art/data_enrich/data/gold.db")


def load_29k_data():
    """Load 29K Sotheby's data directly from SQLite (no estimates)."""
    conn_s = sqlite3.connect(str(SILVER_DB))
    conn_g = sqlite3.connect(str(GOLD_DB))

    # Get gold features
    gold_rows = conn_g.execute("""
        SELECT lot_uuid, log_hammer_price, surface_area_cm2, artist_id, vital_status
        FROM gold_features
        WHERE log_hammer_price IS NOT NULL AND log_hammer_price > 0
    """).fetchall()
    gold_map = {r[0]: r for r in gold_rows}

    # Get silver extractions
    silver_rows = conn_s.execute("""
        SELECT lot_uuid, artist_name, artist_nationality, artist_birth_year, artist_death_year,
               height_cm, width_cm, depth_cm, medium, creation_year,
               hammer_price_usd, auction_uuid
        FROM silver_extractions
        WHERE hammer_price_usd IS NOT NULL AND hammer_price_usd >= 100
    """).fetchall()

    conn_s.close()
    conn_g.close()

    # Build records
    records = []
    for row in silver_rows:
        lot_uuid = row[0]
        gold = gold_map.get(lot_uuid)
        if gold is None:
            continue

        log_price = gold[1]
        surface_area = gold[2]
        artist_id = gold[3]
        vital_status = gold[4]

        artist_name = row[1]
        nationality = row[2]
        birth_year = row[3]
        death_year = row[4]
        height_cm = row[5]
        width_cm = row[6]
        depth_cm = row[7]
        medium = row[8]
        creation_year = row[9]
        hammer_usd = row[10]
        auction_uuid = row[11]

        records.append({
            "lot_uuid": lot_uuid,
            "log_price": log_price,
            "hammer_usd": hammer_usd,
            "surface_area": surface_area,
            "artist_id": artist_id,
            "vital_status": vital_status,
            "artist_name": artist_name,
            "nationality": nationality,
            "birth_year": birth_year,
            "death_year": death_year,
            "height_cm": height_cm,
            "width_cm": width_cm,
            "depth_cm": depth_cm,
            "medium": medium,
            "creation_year": creation_year,
            "auction_uuid": auction_uuid,
        })

    print(f"  Loaded {len(records)} records from SQLite")

    # Build features
    # Compute artist stats from training data
    from collections import defaultdict
    artist_prices = defaultdict(list)
    for r in records:
        if r["artist_id"] is not None:
            artist_prices[r["artist_id"]].append(r["log_price"])

    feature_rows = []
    y_values = []
    medium_labels = []
    price_bucket_labels = []

    for r in records:
        feats = {}
        # Physical
        feats["height_cm"] = r["height_cm"] if r["height_cm"] else np.nan
        feats["width_cm"] = r["width_cm"] if r["width_cm"] else np.nan
        feats["log_surface_area"] = np.log1p(r["surface_area"]) if r["surface_area"] else np.nan
        feats["aspect_ratio"] = (r["height_cm"] / r["width_cm"]) if r["height_cm"] and r["width_cm"] and r["width_cm"] > 0 else np.nan
        feats["has_depth"] = 1.0 if r["depth_cm"] and r["depth_cm"] > 0 else 0.0
        feats["log_depth_cm"] = np.log1p(r["depth_cm"]) if r["depth_cm"] and r["depth_cm"] > 0 else 0.0

        # Artist
        feats["is_living"] = 1.0 if r["vital_status"] == "living" else 0.0 if r["vital_status"] else np.nan
        feats["birth_year"] = float(r["birth_year"]) if r["birth_year"] else np.nan
        feats["nationality_known"] = 1.0 if r["nationality"] else 0.0

        # Creation
        feats["creation_year"] = float(r["creation_year"]) if r["creation_year"] else np.nan
        feats["years_since_creation"] = (2024 - float(r["creation_year"])) if r["creation_year"] else np.nan

        # Medium flags
        med_lower = (r["medium"] or "").lower()
        feats["is_painting"] = 1.0 if any(w in med_lower for w in ["oil", "acrylic", "canvas", "painting"]) else 0.0
        feats["is_sculpture"] = 1.0 if any(w in med_lower for w in ["bronze", "marble", "sculpture", "carved"]) else 0.0
        feats["is_work_on_paper"] = 1.0 if any(w in med_lower for w in ["paper", "print", "lithograph", "etching", "watercolor"]) else 0.0
        feats["medium_known"] = 1.0 if r["medium"] else 0.0

        # Artist history
        aid = r["artist_id"]
        if aid and aid in artist_prices and len(artist_prices[aid]) >= 2:
            prices = artist_prices[aid]
            feats["artist_avg_log_price"] = np.mean(prices)
            feats["artist_median_log_price"] = np.median(prices)
            feats["artist_price_std"] = np.std(prices) if len(prices) > 1 else 0.0
            feats["artist_prior_lots"] = float(len(prices))
            feats["artist_market_depth"] = np.log1p(len(prices))
            feats["is_rare_artist"] = 1.0 if len(prices) < 5 else 0.0
        else:
            feats["artist_avg_log_price"] = np.nan
            feats["artist_median_log_price"] = np.nan
            feats["artist_price_std"] = np.nan
            feats["artist_prior_lots"] = np.nan
            feats["artist_market_depth"] = np.nan
            feats["is_rare_artist"] = np.nan

        feature_rows.append(feats)
        y_values.append(r["log_price"])

        # Medium label for probes
        if feats["is_painting"]:
            medium_labels.append("painting")
        elif feats["is_sculpture"]:
            medium_labels.append("sculpture")
        elif feats["is_work_on_paper"]:
            medium_labels.append("work_on_paper")
        else:
            medium_labels.append("other")

    # Convert to arrays
    feature_names = sorted(feature_rows[0].keys())
    X = np.array([[row[f] for f in feature_names] for row in feature_rows], dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)

    # Price buckets (quartiles)
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    buckets = np.digitize(y, [q25, q50, q75])  # 0,1,2,3

    print(f"  Features: {len(feature_names)} -> {feature_names}")
    print(f"  Price range: ${np.exp(y.min()):.0f} - ${np.exp(y.max()):.0f}")
    print(f"  Quartiles: Q1=${np.exp(q25):.0f}, Q2=${np.exp(q50):.0f}, Q3=${np.exp(q75):.0f}")

    return X, y, feature_names, buckets, medium_labels


def impute_and_scale(X, X_train_ref=None, scaler=None):
    """Impute NaN with medians and standardize."""
    X = X.copy()
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            med = np.nanmedian(X[:, col])
            X[mask, col] = med if np.isfinite(med) else 0.0
    # Replace inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
    else:
        X_sc = scaler.transform(X)
    return X_sc, scaler


def regression_metrics(y_true, y_pred, label):
    """Compute regression metrics in log space and dollar terms."""
    r2 = r2_score(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log = mean_absolute_error(y_true, y_pred)

    # Convert to dollar terms
    # If y = log(price), then price = exp(y)
    # Error in log space = log(predicted/actual)
    # So exp(error) = predicted/actual ratio
    actual_prices = np.exp(y_true)
    predicted_prices = np.exp(y_pred)

    # Percentage errors
    pct_errors = np.abs(predicted_prices - actual_prices) / actual_prices
    mdape = float(np.median(pct_errors) * 100)
    mean_ape = float(np.mean(pct_errors) * 100)

    # Typical prediction range: for a $10K item
    # If RMSE in log space = 0.5, then prediction is within exp(0.5) = 1.65x
    mult_factor = np.exp(rmse_log)
    dollar_example = 10000
    low_bound = dollar_example / mult_factor
    high_bound = dollar_example * mult_factor

    return {
        "label": label,
        "r2": float(r2),
        "rmse_log": float(rmse_log),
        "mae_log": float(mae_log),
        "mdape": mdape,
        "mean_ape": mean_ape,
        "mult_factor": float(mult_factor),
        "example_10k_low": float(low_bound),
        "example_10k_high": float(high_bound),
    }


def main():
    t0 = time.perf_counter()

    print("=" * 80)
    print("  REGRESSION PRECISION ANALYSIS")
    print("  How narrow are the predictions?")
    print("=" * 80)

    output_dir = EXPERIMENT_DIR / "results" / "regression_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("\n--- Load 29K Sotheby's Data (no estimates) ---")
    X, y, feature_names, price_buckets, medium_labels = load_29k_data()
    n_samples, n_features = X.shape
    print(f"  Shape: {n_samples} x {n_features}")

    # Temporal split 70/15/15
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.85)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]
    buckets_test = price_buckets[n_val:]

    print(f"  Split: train={n_train}, val={n_val-n_train}, test={n_samples-n_val}")

    # Standardize
    X_train_sc, scaler = impute_and_scale(X_train)
    X_val_sc, _ = impute_and_scale(X_val, scaler=scaler)
    X_test_sc, _ = impute_and_scale(X_test, scaler=scaler)
    X_all_sc, _ = impute_and_scale(X, scaler=scaler)

    # Binary target for SVAE training
    y_median = np.median(y_train)
    y_binary = (y >= y_median).astype(np.float32)

    # ---- Train Orth-SVAE ----
    print("\n--- Train Orth-SVAE (dim=8, gamma=1.0) ---")
    data_dir = output_dir / "train_data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "features.npy", X_all_sc.astype(np.float32))
    np.save(data_dir / "labels.npy", y_binary)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump({"feature_names": feature_names, "n_features": n_features}, f)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_config = TrainConfig(
        model_type="supervised_vae",
        embedding_dim=8,
        hidden_dims=(256, 128),
        dropout=0.1,
        beta=1.0, alpha=1.0, gamma=1.0,
        epochs=300, batch_size=128,
        learning_rate=0.001, patience=30,
        data_dir=str(data_dir),
        output_dir=str(ckpt_dir),
        seed=42,
    )
    result = train_model(train_config)
    print(f"  Best epoch: {result['best_epoch']}, val loss: {result['best_val_loss']:.6f}")

    # Load model and encode
    model_config = AutoencoderConfig(
        input_dim=n_features, embedding_dim=8,
        hidden_dims=(256, 128), dropout=0.1,
        model_type="supervised_vae",
        beta=1.0, alpha=1.0, gamma=1.0,
    )
    model = MarketAutoencoder(model_config)
    model.load_state_dict(torch.load(ckpt_dir / "best_model_supervised_vae.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        Z_train = model.get_embedding(torch.tensor(X_train_sc, dtype=torch.float32))
        Z_test = model.get_embedding(torch.tensor(X_test_sc, dtype=torch.float32))
        Z_all = model.get_embedding(torch.tensor(X_all_sc, dtype=torch.float32))

    # PCA
    pca = PCA(n_components=8, random_state=42)
    Z_train_pca = pca.fit_transform(X_train_sc)
    Z_test_pca = pca.transform(X_test_sc)

    # ---- 5-Model Regression Comparison ----
    print("\n--- 5-Model Regression Comparison (OOS Test Set) ---")

    models = {}

    # 1. Raw+Ridge
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train_sc, y_train)
    models["Raw+Ridge"] = regression_metrics(y_test, ridge_raw.predict(X_test_sc), "Raw+Ridge")

    # 2. PCA+Ridge
    ridge_pca = Ridge(alpha=1.0)
    ridge_pca.fit(Z_train_pca, y_train)
    models["PCA+Ridge"] = regression_metrics(y_test, ridge_pca.predict(Z_test_pca), "PCA+Ridge")

    # 3. Raw+RF
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_sc, y_train)
    models["Raw+RF"] = regression_metrics(y_test, rf.predict(X_test_sc), "Raw+RF")

    # 4. Raw+LGBM
    if HAS_LGBM:
        gbm = lgb.LGBMRegressor(n_estimators=200, max_depth=-1, num_leaves=31,
                                  learning_rate=0.1, min_child_samples=20,
                                  subsample=0.8, colsample_bytree=0.8,
                                  verbose=-1, random_state=42)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                         learning_rate=0.1, min_samples_leaf=20,
                                         subsample=0.8, random_state=42)
    gbm.fit(X_train_sc, y_train)
    lgbm_label = "Raw+LGBM" if HAS_LGBM else "Raw+GBM"
    models[lgbm_label] = regression_metrics(y_test, gbm.predict(X_test_sc), lgbm_label)

    # 5. SVAE+Ridge
    ridge_svae = Ridge(alpha=1.0)
    ridge_svae.fit(Z_train, y_train)
    models["SVAE+Ridge"] = regression_metrics(y_test, ridge_svae.predict(Z_test), "SVAE+Ridge")

    # Print results table
    print(f"\n{'='*100}")
    print(f"  REGRESSION RESULTS (Out-of-Sample Test Set)")
    print(f"{'='*100}")
    header = f"  {'Model':<15s} {'R2':>8s} {'RMSE(log)':>10s} {'MAE(log)':>10s} {'MdAPE%':>8s} {'MeanAPE%':>9s} {'x-factor':>9s} {'$10K range':>20s}"
    print(header)
    print("  " + "-" * 95)

    for name in ["Raw+Ridge", "PCA+Ridge", "Raw+RF", lgbm_label, "SVAE+Ridge"]:
        m = models[name]
        print(f"  {m['label']:<15s} {m['r2']:>8.4f} {m['rmse_log']:>10.4f} {m['mae_log']:>10.4f} "
              f"{m['mdape']:>8.1f} {m['mean_ape']:>9.1f} {m['mult_factor']:>8.2f}x "
              f"${m['example_10k_low']:>7,.0f}-${m['example_10k_high']:>7,.0f}")

    # ---- Interpretation ----
    print(f"\n{'='*100}")
    print(f"  INTERPRETATION: How Narrow Are The Predictions?")
    print(f"{'='*100}")

    best = min(models.values(), key=lambda m: m["rmse_log"])
    svae = models["SVAE+Ridge"]

    print(f"\n  Best model: {best['label']}")
    print(f"  RMSE in log space: {best['rmse_log']:.4f}")
    print(f"  This means predictions are within a {best['mult_factor']:.2f}x factor of actual price:")
    print(f"    If actual = $1,000  -> prediction range: ${1000/best['mult_factor']:,.0f} - ${1000*best['mult_factor']:,.0f}")
    print(f"    If actual = $10,000 -> prediction range: ${10000/best['mult_factor']:,.0f} - ${10000*best['mult_factor']:,.0f}")
    print(f"    If actual = $100,000-> prediction range: ${100000/best['mult_factor']:,.0f} - ${100000*best['mult_factor']:,.0f}")
    print(f"  Median Absolute Percentage Error: {best['mdape']:.1f}%")
    print(f"    For a $10,000 item, half the predictions are within +/-${10000*best['mdape']/100:,.0f}")

    print(f"\n  SVAE+Ridge:")
    print(f"  RMSE in log space: {svae['rmse_log']:.4f}")
    print(f"  Prediction range: {svae['mult_factor']:.2f}x factor")
    print(f"    $10,000 item -> ${svae['example_10k_low']:,.0f} - ${svae['example_10k_high']:,.0f}")
    print(f"  MdAPE: {svae['mdape']:.1f}%")

    # ---- 4-Class Price Bucket Prediction ----
    print(f"\n{'='*100}")
    print(f"  4-CLASS PRICE BUCKET PREDICTION (Quartiles)")
    print(f"{'='*100}")

    q25, q50, q75 = np.percentile(y_train, [25, 50, 75])
    print(f"  Q1: <${np.exp(q25):,.0f}  Q2: ${np.exp(q25):,.0f}-${np.exp(q50):,.0f}  "
          f"Q3: ${np.exp(q50):,.0f}-${np.exp(q75):,.0f}  Q4: >${np.exp(q75):,.0f}")

    # Predict continuous, then bin
    bucket_results = {}
    for name, y_pred_test in [
        ("Raw+Ridge", ridge_raw.predict(X_test_sc)),
        ("PCA+Ridge", ridge_pca.predict(Z_test_pca)),
        ("Raw+RF", rf.predict(X_test_sc)),
        (lgbm_label, gbm.predict(X_test_sc)),
        ("SVAE+Ridge", ridge_svae.predict(Z_test)),
    ]:
        pred_buckets = np.digitize(y_pred_test, [q25, q50, q75])
        actual_buckets = np.digitize(y_test, [q25, q50, q75])
        acc = accuracy_score(actual_buckets, pred_buckets)
        # Within-1 accuracy (adjacent bucket is OK)
        within1 = np.mean(np.abs(pred_buckets.astype(int) - actual_buckets.astype(int)) <= 1)
        bucket_results[name] = {"accuracy": acc, "within_1": within1}
        print(f"  {name:<15s}  Exact: {acc:.1%}   Within-1-bucket: {within1:.1%}")

    print(f"\n  Random baseline: 25.0% exact, ~62.5% within-1")

    # ---- Detailed error distribution ----
    print(f"\n{'='*100}")
    print(f"  ERROR DISTRIBUTION (Best model: {best['label']})")
    print(f"{'='*100}")

    if best["label"] == "Raw+RF":
        y_pred_best = rf.predict(X_test_sc)
    elif best["label"] == lgbm_label:
        y_pred_best = gbm.predict(X_test_sc)
    elif best["label"] == "Raw+Ridge":
        y_pred_best = ridge_raw.predict(X_test_sc)
    elif best["label"] == "SVAE+Ridge":
        y_pred_best = ridge_svae.predict(Z_test)
    else:
        y_pred_best = ridge_pca.predict(Z_test_pca)

    errors_log = y_pred_best - y_test
    abs_pct_errors = np.abs(np.exp(errors_log) - 1) * 100

    percentiles = [10, 25, 50, 75, 90, 95]
    print(f"\n  Absolute percentage error distribution:")
    for p in percentiles:
        val = np.percentile(abs_pct_errors, p)
        print(f"    P{p:02d}: {val:>6.1f}%  (${10000 * val / 100:>8,.0f} error on a $10K item)")

    print(f"\n  Fraction within error thresholds:")
    for threshold in [10, 20, 30, 50, 100]:
        frac = np.mean(abs_pct_errors <= threshold)
        print(f"    Within +/-{threshold}%: {frac:.1%}")

    # ---- Save report ----
    report = {
        "experiment": "regression_precision_analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "Sotheby's 29K (no estimates)",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_train": n_train,
            "n_test": n_samples - n_val,
            "price_range_usd": [float(np.exp(y.min())), float(np.exp(y.max()))],
            "quartiles_usd": [float(np.exp(q25)), float(np.exp(q50)), float(np.exp(q75))],
        },
        "regression_results": models,
        "bucket_results": bucket_results,
        "error_distribution": {
            f"p{p}": float(np.percentile(abs_pct_errors, p)) for p in percentiles
        },
        "within_thresholds": {
            f"within_{t}pct": float(np.mean(abs_pct_errors <= t))
            for t in [10, 20, 30, 50, 100]
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    report_path = output_dir / "regression_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  Completed in {report['elapsed_seconds']:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
