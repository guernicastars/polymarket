#!/usr/bin/env python3
"""Initial calibration study for resolved binary Polymarket markets.

This script is the first empirical scaffold for Proposal A:
``Toward a Causal Microstructure of Prediction Market Calibration``.

Design goals:
- Use the current ClickHouse-backed repo directly.
- Avoid the broken ``market_prices`` table by reconstructing implied
  probabilities from ``market_trades``.
- Produce lightweight, paper-friendly artifacts: market-level CSV,
  slice-level CSVs, reliability bins, and a JSON summary.

Probability convention:
- For every binary market, the "positive" event is the first listed outcome.
- Trades on that outcome use ``price`` directly.
- Trades on the other outcome are mapped to ``1 - price``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from data.config import (  # noqa: E402
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)
from data.extract import fetch_resolved_markets  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


RESULTS_DIR = EXPERIMENT_DIR / "results" / "calibration_study"
DEFAULT_HORIZONS = {
    "final": None,
    "6h": 6,
    "24h": 24,
    "168h": 168,
}
LONGSHOT_BINS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.50]
LONGSHOT_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-40%", "40-50%"]


@dataclass
class MetricSummary:
    n: int
    positive_rate: float
    mean_pred: float
    brier: float
    log_loss: float
    ece_10: float


def get_client() -> clickhouse_connect.driver.client.Client:
    """Create a ClickHouse client using shared experiment credentials."""
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


def encode_positive_outcome(row: pd.Series) -> str:
    """Use the first binary outcome as the positive event."""
    outcomes = row.get("outcomes", [])
    if isinstance(outcomes, list) and outcomes:
        return str(outcomes[0])
    return ""


def encode_target(row: pd.Series) -> float:
    """Return 1.0 when the first outcome wins, else 0.0."""
    outcomes = row.get("outcomes", [])
    winning = str(row.get("winning_outcome", ""))
    if not outcomes or len(outcomes) < 2 or not winning:
        return np.nan
    return float(winning == outcomes[0])


def duration_days(row: pd.Series) -> float:
    """Compute market duration in days."""
    start = pd.Timestamp(row.get("start_date"))
    end = pd.Timestamp(row.get("end_date"))
    if pd.notna(start) and pd.notna(end) and end > start:
        return float((end - start).total_seconds() / 86400.0)
    return np.nan


def fetch_trades_batch(
    client: clickhouse_connect.driver.client.Client,
    condition_ids: list[str],
) -> pd.DataFrame:
    """Fetch trade rows for a chunk of condition_ids."""
    query = """
    SELECT
        condition_id,
        outcome,
        price,
        size,
        side,
        timestamp
    FROM market_trades
    WHERE condition_id IN {cids:Array(String)}
    ORDER BY condition_id, timestamp
    """
    result = client.query(query, parameters={"cids": condition_ids})
    return pd.DataFrame(result.result_rows, columns=result.column_names)


def probability_before_cutoff(
    times: np.ndarray,
    probs: np.ndarray,
    cutoff: np.datetime64,
) -> float:
    """Return the last implied probability observed on or before cutoff."""
    idx = np.searchsorted(times, cutoff, side="right") - 1
    if idx < 0:
        return np.nan
    return float(probs[idx])


def compute_market_probabilities(
    markets_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    horizons: dict[str, int | None],
) -> pd.DataFrame:
    """Compute implied probabilities per market at multiple horizons."""
    if trades_df.empty:
        return pd.DataFrame(columns=["condition_id", *[f"p_{h}" for h in horizons]])

    meta = markets_df[["condition_id", "positive_outcome", "end_date"]].copy()
    merged = trades_df.merge(meta, on="condition_id", how="inner")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=False)
    merged["end_date"] = pd.to_datetime(merged["end_date"], utc=False)
    merged["prob_positive"] = np.where(
        merged["outcome"].astype(str) == merged["positive_outcome"].astype(str),
        merged["price"].astype(float),
        1.0 - merged["price"].astype(float),
    )

    rows: list[dict[str, float | str]] = []
    for condition_id, group in merged.groupby("condition_id", sort=False):
        group = group.sort_values("timestamp")
        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        probs = group["prob_positive"].to_numpy(dtype=np.float64)
        end_time = np.datetime64(group["end_date"].iloc[0].to_datetime64())

        row: dict[str, float | str] = {"condition_id": condition_id}
        for horizon_name, hours in horizons.items():
            cutoff = end_time if hours is None else end_time - np.timedelta64(hours, "h")
            row[f"p_{horizon_name}"] = probability_before_cutoff(times, probs, cutoff)

        row["first_trade_time"] = group["timestamp"].iloc[0].isoformat()
        row["last_trade_time"] = group["timestamp"].iloc[-1].isoformat()
        rows.append(row)

    return pd.DataFrame(rows)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts."""
    return float(np.mean((y_prob - y_true) ** 2))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary log loss with clipping."""
    eps = 1e-6
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def row_log_loss(y_true: pd.Series, y_prob: pd.Series) -> pd.Series:
    """Row-wise binary log loss with clipping."""
    eps = 1e-6
    clipped = y_prob.astype(float).clip(eps, 1.0 - eps)
    targets = y_true.astype(float)
    return -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error on equally spaced bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for idx in range(n_bins):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return reliability-bin statistics for plotting or appendix tables."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_start": lo,
                    "bin_end": hi,
                    "count": 0,
                    "mean_pred": np.nan,
                    "observed_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "count": count,
                "mean_pred": float(np.mean(y_prob[mask])),
                "observed_rate": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame(rows)


def build_market_horizon_panel(
    analysis_df: pd.DataFrame,
    horizons: dict[str, int | None],
) -> pd.DataFrame:
    """Create one paper-facing row per market per horizon."""
    panel_frames: list[pd.DataFrame] = []
    base_cols = [
        "condition_id",
        "question",
        "event_id",
        "event_slug",
        "category",
        "winning_outcome",
        "positive_outcome",
        "target",
        "n_trades",
        "trade_bin",
        "volume_total",
        "volume_24h",
        "volume_1wk",
        "volume_1mo",
        "volume_quartile",
        "one_day_price_change",
        "one_week_price_change",
        "neg_risk",
        "duration_days",
        "start_date",
        "end_date",
        "first_trade_time",
        "last_trade_time",
    ]

    for horizon_name, hours in horizons.items():
        prob_col = f"p_{horizon_name}"
        sub = analysis_df[base_cols + [prob_col]].copy()
        sub = sub[sub[prob_col].notna()].copy()
        if sub.empty:
            continue

        sub["horizon"] = horizon_name
        sub["horizon_hours"] = np.nan if hours is None else float(hours)
        sub["resolution_time"] = pd.to_datetime(sub["end_date"], errors="coerce")
        if hours is None:
            sub["horizon_time"] = sub["resolution_time"]
        else:
            sub["horizon_time"] = sub["resolution_time"] - pd.to_timedelta(hours, unit="h")

        sub["implied_probability"] = sub[prob_col].astype(float)
        sub["brier"] = (sub["implied_probability"] - sub["target"].astype(float)) ** 2
        sub["log_loss"] = row_log_loss(sub["target"], sub["implied_probability"])
        sub["favorite_prob"] = np.maximum(sub["implied_probability"], 1.0 - sub["implied_probability"])
        sub["longshot_prob"] = 1.0 - sub["favorite_prob"]
        sub["favorite_win"] = np.where(
            sub["implied_probability"] >= 0.5,
            sub["target"].astype(float),
            1.0 - sub["target"].astype(float),
        )
        sub["longshot_win"] = 1.0 - sub["favorite_win"]
        sub["probability_bucket"] = pd.cut(
            sub["implied_probability"],
            bins=np.linspace(0.0, 1.0, 11),
            include_lowest=True,
            labels=[f"{int(lo*100):02d}-{int(hi*100):02d}%" for lo, hi in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10))],
        )
        sub["longshot_bin"] = pd.cut(
            sub["longshot_prob"],
            bins=LONGSHOT_BINS,
            labels=LONGSHOT_LABELS,
            include_lowest=True,
            right=False,
        )
        panel_frames.append(sub)

    if not panel_frames:
        return pd.DataFrame()

    panel = pd.concat(panel_frames, ignore_index=True)
    overpricing = (
        panel.dropna(subset=["longshot_bin"])
        .groupby(["horizon", "longshot_bin"], observed=True)
        .agg(
            bin_n=("condition_id", "count"),
            bin_mean_longshot_prob=("longshot_prob", "mean"),
            bin_observed_longshot_win=("longshot_win", "mean"),
        )
        .reset_index()
    )
    overpricing["bin_longshot_bias"] = (
        overpricing["bin_mean_longshot_prob"] - overpricing["bin_observed_longshot_win"]
    )
    denom = overpricing["bin_observed_longshot_win"].replace(0.0, np.nan)
    overpricing["bin_relative_longshot_overpricing"] = (
        overpricing["bin_mean_longshot_prob"] / denom
    )
    overpricing["longshot_overpricing_flag"] = (
        overpricing["bin_relative_longshot_overpricing"] >= 1.25
    ) & (overpricing["bin_longshot_bias"] > 0.0)

    panel = panel.merge(
        overpricing,
        on=["horizon", "longshot_bin"],
        how="left",
    )
    return panel


def summarize_predictions(df: pd.DataFrame, prob_col: str) -> MetricSummary | None:
    """Compute calibration metrics for one probability column."""
    valid = df[df[prob_col].notna()].copy()
    if valid.empty:
        return None
    y_true = valid["target"].to_numpy(dtype=np.float64)
    y_prob = valid[prob_col].to_numpy(dtype=np.float64)
    return MetricSummary(
        n=int(len(valid)),
        positive_rate=float(np.mean(y_true)),
        mean_pred=float(np.mean(y_prob)),
        brier=brier_score(y_true, y_prob),
        log_loss=binary_log_loss(y_true, y_prob),
        ece_10=expected_calibration_error(y_true, y_prob, n_bins=10),
    )


def add_trade_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add coarse trade-count bins for slice analysis."""
    out = df.copy()
    bins = [-math.inf, 10, 50, 100, 500, math.inf]
    labels = ["<10", "10-49", "50-99", "100-499", "500+"]
    out["trade_bin"] = pd.cut(
        out["n_trades"].astype(float),
        bins=bins,
        labels=labels,
        right=False,
    )
    return out


def add_volume_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume quartiles for descriptive heterogeneity checks."""
    out = df.copy()
    try:
        out["volume_quartile"] = pd.qcut(
            out["volume_total"].astype(float),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        out["volume_quartile"] = "Q?"
    return out


def slice_metrics(
    df: pd.DataFrame,
    slice_col: str,
    horizon_cols: list[str],
    min_count: int,
) -> pd.DataFrame:
    """Compute calibration metrics by slice and horizon."""
    rows = []
    for slice_value, group in df.groupby(slice_col, dropna=False):
        for prob_col in horizon_cols:
            summary = summarize_predictions(group, prob_col)
            if summary is None or summary.n < min_count:
                continue
            row = {"slice_col": slice_col, "slice_value": str(slice_value), "horizon": prob_col}
            row.update(asdict(summary))
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary_markdown(summary: dict, path: Path) -> None:
    """Write a short markdown note for the paper-writing loop."""
    lines = [
        "# Calibration Study Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Min trades filter: `{summary['config']['min_trades']}`",
        f"- Resolved binary markets in sample: `{summary['dataset']['n_markets']}`",
        f"- Unique categories: `{summary['dataset']['n_categories']}`",
        f"- Market-horizon rows: `{summary['dataset']['n_market_horizon_rows']}`",
        "",
        "## Overall",
        "",
    ]
    for horizon, metrics in summary["overall"].items():
        if metrics is None:
            continue
        lines.extend(
            [
                f"### {horizon}",
                "",
                f"- n = {metrics['n']}",
                f"- Brier = {metrics['brier']:.4f}",
                f"- Log loss = {metrics['log_loss']:.4f}",
                f"- ECE(10) = {metrics['ece_10']:.4f}",
                f"- Mean predicted = {metrics['mean_pred']:.4f}",
                f"- Positive rate = {metrics['positive_rate']:.4f}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--min-category-count", type=int, default=30)
    parser.add_argument("--min-slice-count", type=int, default=30)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client()
    logger.info("Fetching resolved binary markets with >= %d trades", args.min_trades)
    markets_df = fetch_resolved_markets(client, min_trades=args.min_trades).copy()

    markets_df["positive_outcome"] = markets_df.apply(encode_positive_outcome, axis=1)
    markets_df["target"] = markets_df.apply(encode_target, axis=1)
    markets_df["duration_days"] = markets_df.apply(duration_days, axis=1)
    markets_df["end_date"] = pd.to_datetime(markets_df["end_date"], utc=False)
    markets_df = markets_df[markets_df["target"].notna()].copy()

    logger.info(
        "Resolved binary sample after target encoding: %d markets across %d categories",
        len(markets_df),
        markets_df["category"].nunique(dropna=True),
    )

    probability_frames = []
    condition_ids = markets_df["condition_id"].tolist()
    for start in range(0, len(condition_ids), args.chunk_size):
        chunk_ids = condition_ids[start : start + args.chunk_size]
        logger.info(
            "Processing trade chunk %d-%d / %d",
            start + 1,
            min(start + len(chunk_ids), len(condition_ids)),
            len(condition_ids),
        )
        trades_df = fetch_trades_batch(client, chunk_ids)
        if trades_df.empty:
            continue
        chunk_markets = markets_df[markets_df["condition_id"].isin(chunk_ids)]
        prob_df = compute_market_probabilities(chunk_markets, trades_df, DEFAULT_HORIZONS)
        probability_frames.append(prob_df)

    if not probability_frames:
        raise RuntimeError("No trade-derived probabilities could be computed.")

    probabilities_df = pd.concat(probability_frames, ignore_index=True)
    analysis_df = markets_df.merge(probabilities_df, on="condition_id", how="left")
    analysis_df = add_trade_bins(analysis_df)
    analysis_df = add_volume_quartiles(analysis_df)
    market_horizon_panel = build_market_horizon_panel(analysis_df, DEFAULT_HORIZONS)

    horizon_cols = [f"p_{name}" for name in DEFAULT_HORIZONS]
    overall = {
        horizon: (
            None
            if (summary := summarize_predictions(analysis_df, f"p_{horizon}")) is None
            else asdict(summary)
        )
        for horizon in DEFAULT_HORIZONS
    }

    category_metrics = slice_metrics(
        analysis_df,
        slice_col="category",
        horizon_cols=horizon_cols,
        min_count=args.min_category_count,
    )
    trade_bin_metrics = slice_metrics(
        analysis_df,
        slice_col="trade_bin",
        horizon_cols=horizon_cols,
        min_count=args.min_slice_count,
    )
    volume_metrics = slice_metrics(
        analysis_df,
        slice_col="volume_quartile",
        horizon_cols=horizon_cols,
        min_count=args.min_slice_count,
    )

    reliability_outputs = {}
    for horizon in DEFAULT_HORIZONS:
        prob_col = f"p_{horizon}"
        valid = analysis_df[analysis_df[prob_col].notna()].copy()
        if valid.empty:
            continue
        bins_df = reliability_bins(
            valid["target"].to_numpy(dtype=np.float64),
            valid[prob_col].to_numpy(dtype=np.float64),
            n_bins=10,
        )
        bins_path = RESULTS_DIR / f"reliability_{horizon}.csv"
        bins_df.to_csv(bins_path, index=False)
        reliability_outputs[horizon] = str(bins_path)

    market_level_path = RESULTS_DIR / "market_level.csv"
    market_horizon_panel_path = RESULTS_DIR / "market_horizon_panel.csv"
    category_metrics_path = RESULTS_DIR / "category_metrics.csv"
    trade_bin_metrics_path = RESULTS_DIR / "trade_bin_metrics.csv"
    volume_metrics_path = RESULTS_DIR / "volume_quartile_metrics.csv"

    analysis_df.to_csv(market_level_path, index=False)
    market_horizon_panel.to_csv(market_horizon_panel_path, index=False)
    category_metrics.to_csv(category_metrics_path, index=False)
    trade_bin_metrics.to_csv(trade_bin_metrics_path, index=False)
    volume_metrics.to_csv(volume_metrics_path, index=False)

    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": {
            "min_trades": args.min_trades,
            "chunk_size": args.chunk_size,
            "horizons": DEFAULT_HORIZONS,
        },
        "dataset": {
            "n_markets": int(len(analysis_df)),
            "n_market_horizon_rows": int(len(market_horizon_panel)),
            "n_categories": int(analysis_df["category"].nunique(dropna=True)),
            "positive_rate": float(analysis_df["target"].mean()),
            "median_trades": float(analysis_df["n_trades"].median()),
            "median_volume_total": float(analysis_df["volume_total"].median()),
            "median_duration_days": float(analysis_df["duration_days"].median()),
            "coverage": {
                horizon: int(analysis_df[f"p_{horizon}"].notna().sum())
                for horizon in DEFAULT_HORIZONS
            },
        },
        "overall": overall,
        "artifacts": {
            "market_level_csv": str(market_level_path),
            "market_horizon_panel_csv": str(market_horizon_panel_path),
            "category_metrics_csv": str(category_metrics_path),
            "trade_bin_metrics_csv": str(trade_bin_metrics_path),
            "volume_quartile_metrics_csv": str(volume_metrics_path),
            "reliability_bins": reliability_outputs,
        },
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_summary_markdown(summary, RESULTS_DIR / "summary.md")

    logger.info("Wrote calibration study outputs to %s", RESULTS_DIR)
    logger.info("Overall summary: %s", json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
