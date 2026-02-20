"""Main extraction script: queries ClickHouse, computes features, saves .npz.

Extracts the 27-feature spec from RESEARCH.md for resolved binary Polymarket
contracts.  Does NOT use the market_prices table (broken -- empty condition_id).
Price features come from market_trades and markets table pre-computed fields.

Usage:
    python -m experiments.data.extract                     # defaults (Tier 2)
    python -m experiments.data.extract --min-trades 100    # Tier 3 (rich)
    python -m experiments.data.extract --min-trades 3      # relaxed threshold
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd
from clickhouse_connect.driver.client import Client

from data.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    MIN_TRADES,
    OUTPUT_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data.features import build_feature_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# ClickHouse connection
# ======================================================================

def get_client() -> Client:
    """Create a ClickHouse client using configured credentials."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
        compress="lz4",
        connect_timeout=30,
        send_receive_timeout=300,
    )


# ======================================================================
# Data queries
# ======================================================================

def fetch_resolved_markets(client: Client, min_trades: int) -> pd.DataFrame:
    """Fetch resolved binary markets with sufficient trade data.

    Filters to:
      - resolved = 1, winning_outcome != ''
      - Binary outcomes (Yes/No or Over/Under only)
      - At least min_trades trades in market_trades

    Returns a DataFrame with market metadata + trade count.
    """
    query = """
    SELECT
        m.condition_id,
        m.question,
        m.event_id,
        m.event_slug,
        m.category,
        m.outcomes,
        m.outcome_prices,
        m.token_ids,
        m.winning_outcome,
        m.neg_risk,
        m.volume_total,
        m.volume_24h,
        m.volume_1wk,
        m.volume_1mo,
        m.liquidity,
        m.one_day_price_change,
        m.one_week_price_change,
        m.start_date,
        m.end_date,
        m.created_at,
        m.updated_at,
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
    ORDER BY m.volume_total DESC
    """
    result = client.query(query, parameters={"min_trades": min_trades})
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info("Found %d resolved binary markets with >= %d trades", len(df), min_trades)
    return df


def fetch_trades(client: Client, condition_id: str) -> pd.DataFrame:
    """Fetch all trades for a market, ordered by timestamp."""
    query = """
    SELECT price, size, side, timestamp
    FROM market_trades
    WHERE condition_id = {cid:String}
    ORDER BY timestamp
    """
    result = client.query(query, parameters={"cid": condition_id})
    return pd.DataFrame(result.result_rows, columns=result.column_names)


def fetch_category_median_volumes(client: Client) -> dict[str, float]:
    """Compute median volume_total per category for all resolved markets.

    Returns a dict mapping category -> median volume.
    """
    query = """
    SELECT category, median(volume_total) AS med_vol
    FROM markets FINAL
    WHERE resolved = 1 AND winning_outcome != ''
    GROUP BY category
    """
    result = client.query(query)
    return {row[0]: float(row[1]) for row in result.result_rows}


def fetch_wallet_data_batch(
    client: Client,
    condition_ids: list[str],
) -> dict[str, dict]:
    """Fetch wallet/smart money features for a batch of markets.

    Queries wallet_activity and insider_scores to compute:
      - unique_wallet_count
      - whale_buy_ratio
      - top_wallet_concentration
      - avg_insider_score

    Returns a dict mapping condition_id -> wallet feature dict.
    Markets without wallet data will be absent from the result.
    """
    if not condition_ids:
        return {}

    # 1. Wallet activity aggregations per market
    wallet_query = """
    SELECT
        condition_id,
        uniqExact(proxy_wallet) AS unique_wallet_count,
        sumIf(usdc_size, side = 'BUY') / nullIf(sum(usdc_size), 0) AS whale_buy_ratio,
        max(wallet_vol) / nullIf(sum(usdc_size), 0) AS top_wallet_concentration
    FROM (
        SELECT
            condition_id,
            proxy_wallet,
            side,
            usdc_size,
            sum(usdc_size) OVER (PARTITION BY condition_id, proxy_wallet) AS wallet_vol
        FROM wallet_activity
        WHERE condition_id IN {cids:Array(String)}
          AND activity_type = 'TRADE'
    )
    GROUP BY condition_id
    """
    result = client.query(wallet_query, parameters={"cids": condition_ids})
    wallet_data: dict[str, dict] = {}
    for row in result.result_rows:
        cid = row[0]
        wallet_data[cid] = {
            "unique_wallet_count": row[1],
            "whale_buy_ratio": float(row[2]) if row[2] is not None else np.nan,
            "top_wallet_concentration": float(row[3]) if row[3] is not None else np.nan,
            "avg_insider_score": np.nan,  # filled below
        }

    # 2. Average insider score for wallets active in each market
    insider_query = """
    SELECT
        wa.condition_id,
        avg(ins.score) AS avg_insider_score
    FROM (
        SELECT DISTINCT condition_id, proxy_wallet
        FROM wallet_activity
        WHERE condition_id IN {cids:Array(String)}
          AND activity_type = 'TRADE'
    ) AS wa
    INNER JOIN (
        SELECT proxy_wallet, score
        FROM insider_scores FINAL
    ) AS ins ON wa.proxy_wallet = ins.proxy_wallet
    GROUP BY wa.condition_id
    """
    result = client.query(insider_query, parameters={"cids": condition_ids})
    for row in result.result_rows:
        cid = row[0]
        if cid in wallet_data:
            wallet_data[cid]["avg_insider_score"] = float(row[1]) if row[1] is not None else np.nan

    return wallet_data


# ======================================================================
# Outcome encoding
# ======================================================================

def encode_outcome(market_row: pd.Series) -> float:
    """Encode binary market outcome.

    Returns 1.0 if the first outcome won (Yes, Over),
    0.0 if the second outcome won (No, Under).

    Note on class imbalance: Yes/No markets skew 22/78 toward No.
    Over/Under markets are roughly balanced (49/51).
    """
    winning = str(market_row.get("winning_outcome", ""))
    outcomes = market_row.get("outcomes", [])

    if not outcomes or not winning:
        return np.nan

    # First outcome wins -> 1.0
    if winning == outcomes[0]:
        return 1.0
    # Second outcome wins -> 0.0
    if winning == outcomes[1]:
        return 0.0

    return np.nan


# ======================================================================
# Temporal split
# ======================================================================

def temporal_split(
    markets_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    outcomes: np.ndarray,
    market_ids: list[str],
) -> dict[str, np.ndarray]:
    """Split data temporally by market end_date (resolution proxy).

    Returns dict with X_train, y_train, X_val, y_val, X_test, y_test,
    plus market_ids_{train,val,test}.
    """
    dates = pd.to_datetime(markets_df["end_date"]).values
    sort_idx = np.argsort(dates)

    n = len(sort_idx)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_idx = sort_idx[:n_train]
    val_idx = sort_idx[n_train:n_val]
    test_idx = sort_idx[n_val:]

    ids_arr = np.array(market_ids)

    return {
        "X_train": feature_matrix[train_idx],
        "y_train": outcomes[train_idx],
        "X_val": feature_matrix[val_idx],
        "y_val": outcomes[val_idx],
        "X_test": feature_matrix[test_idx],
        "y_test": outcomes[test_idx],
        "market_ids_train": ids_arr[train_idx],
        "market_ids_val": ids_arr[val_idx],
        "market_ids_test": ids_arr[test_idx],
    }


# ======================================================================
# Main pipeline
# ======================================================================

def main(
    min_trades: int = MIN_TRADES,
    output_dir: Path | None = None,
) -> None:
    """Run the full extraction pipeline: query -> features -> split -> save."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ClickHouse at %s:%s", CLICKHOUSE_HOST, CLICKHOUSE_PORT)
    client = get_client()

    # 1. Fetch resolved binary markets with trade data
    markets_df = fetch_resolved_markets(client, min_trades)
    if markets_df.empty:
        logger.warning("No resolved markets found meeting thresholds. Exiting.")
        return

    # 2. Pre-compute category median volumes (for feature 23)
    cat_medians = fetch_category_median_volumes(client)
    logger.info("Category medians computed for %d categories", len(cat_medians))

    # 3. Batch-fetch wallet/smart money data (Category E)
    all_cids = markets_df["condition_id"].tolist()
    # Fetch in chunks of 500 to avoid query size limits
    wallet_data_all: dict[str, dict] = {}
    chunk_size = 500
    for i in range(0, len(all_cids), chunk_size):
        chunk = all_cids[i : i + chunk_size]
        wallet_data_all.update(fetch_wallet_data_batch(client, chunk))
    logger.info(
        "Wallet data available for %d / %d markets",
        len(wallet_data_all),
        len(all_cids),
    )

    # 4. Extract features for each market
    all_features: list[dict[str, float]] = []
    all_outcomes: list[float] = []
    valid_market_ids: list[str] = []
    valid_market_indices: list[int] = []

    for i, (idx, row) in enumerate(markets_df.iterrows()):
        cid = row["condition_id"]
        if (i + 1) % 200 == 0 or i == 0:
            logger.info(
                "Processing %d/%d: %s (trades=%d)",
                i + 1, len(markets_df),
                cid[:16] + "...",
                row.get("n_trades", 0),
            )

        # Encode outcome first (skip if not encodable)
        y = encode_outcome(row)
        if np.isnan(y):
            continue

        # Fetch trades
        trades = fetch_trades(client, cid)

        # Market metadata as dict
        market_meta = row.to_dict()

        # Category median volume
        cat = row.get("category", "")
        cat_median = cat_medians.get(cat, 0.0)

        # Wallet data (may be None)
        wallet_data = wallet_data_all.get(cid)

        # Build feature vector (27 features)
        fv = build_feature_vector(
            market=market_meta,
            trades_df=trades,
            category_median_volume=cat_median,
            wallet_data=wallet_data,
        )

        all_features.append(fv)
        all_outcomes.append(y)
        valid_market_ids.append(cid)
        valid_market_indices.append(idx)

    if not all_features:
        logger.warning("No markets produced valid features. Exiting.")
        return

    # 5. Build aligned feature matrix
    all_keys: set[str] = set()
    for fv in all_features:
        all_keys.update(fv.keys())
    feature_names = sorted(all_keys)

    X = np.array(
        [[fv.get(fn, np.nan) for fn in feature_names] for fv in all_features],
        dtype=np.float64,
    )
    y = np.array(all_outcomes, dtype=np.float64)

    logger.info(
        "Feature matrix: %s, outcome vector: %s", X.shape, y.shape,
    )
    logger.info(
        "Features: %d, NaN fraction: %.2f%%",
        len(feature_names), 100 * np.isnan(X).mean(),
    )

    # Class balance report
    n_pos = (y == 1.0).sum()
    n_neg = (y == 0.0).sum()
    logger.info(
        "Class balance: positive=%.1f%% (%d), negative=%.1f%% (%d)",
        100 * n_pos / len(y), n_pos,
        100 * n_neg / len(y), n_neg,
    )

    # 6. Temporal split
    valid_markets_df = markets_df.iloc[valid_market_indices].reset_index(drop=True)
    splits = temporal_split(valid_markets_df, X, y, valid_market_ids)

    # 7. Save .npz
    features_path = out / "features.npz"
    np.savez_compressed(
        features_path,
        feature_names=np.array(feature_names),
        **splits,
    )
    logger.info("Saved features to %s", features_path)

    # 8. Save metadata JSON
    metadata = {
        "n_markets": len(valid_market_ids),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_categories": {
            "A_market_structure": [
                "volume_total", "liquidity", "neg_risk",
                "market_duration_days", "num_outcomes",
            ],
            "B_price_signals": [
                "last_price", "one_day_price_change", "one_week_price_change",
                "price_range", "price_at_75pct_life", "final_price_velocity",
            ],
            "C_trade_microstructure": [
                "trade_count", "avg_trade_size", "max_trade_size",
                "buy_sell_ratio", "buy_volume_ratio", "trade_size_gini",
                "trades_per_day", "late_volume_ratio",
            ],
            "D_volume_dynamics": [
                "volume_24h", "volume_1wk",
                "volume_acceleration", "volume_vs_category_median",
            ],
            "E_wallet_smart_money": [
                "unique_wallet_count", "whale_buy_ratio",
                "top_wallet_concentration", "avg_insider_score",
            ],
        },
        "split_sizes": {
            "train": int(splits["X_train"].shape[0]),
            "val": int(splits["X_val"].shape[0]),
            "test": int(splits["X_test"].shape[0]),
        },
        "class_balance": {
            "positive_frac": float(n_pos / len(y)),
            "negative_frac": float(n_neg / len(y)),
            "note": "Yes/No markets skew 22/78 toward No; Over/Under ~balanced",
        },
        "thresholds": {
            "min_trades": min_trades,
        },
        "nan_fraction": float(np.isnan(X).mean()),
        "data_notes": {
            "market_prices_broken": "All 22.8M rows have empty condition_id. Not used.",
            "ohlcv_shallow": "Pipeline ~6 days old; median 3 hourly bars per resolved market.",
            "price_source": "Price features from market_trades + markets table Gamma API fields.",
            "wallet_coverage": f"{len(wallet_data_all)}/{len(all_cids)} markets have wallet data.",
        },
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = out / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    logger.info(
        "Extraction complete: %d markets, %d features, split %d/%d/%d",
        len(valid_market_ids),
        len(feature_names),
        splits["X_train"].shape[0],
        splits["X_val"].shape[0],
        splits["X_test"].shape[0],
    )


# ======================================================================
# CLI entry point
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from resolved Polymarket contracts",
    )
    parser.add_argument(
        "--min-trades", type=int, default=MIN_TRADES,
        help=f"Minimum trades per market (default: {MIN_TRADES})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        min_trades=args.min_trades,
        output_dir=args.output_dir,
    )
