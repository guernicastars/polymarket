#!/usr/bin/env python3
"""Longshot-bias study for resolved Polymarket markets.

This script turns trader feedback into testable diagnostics:

- Are longshots overpriced relative to realized frequency?
- Is that distortion stronger at tradable horizons than at the final print?
- Does the effect vary by category?
- Is there tentative evidence that the bias is shrinking over time?

The script reuses the market-level output from ``run_calibration_study.py`` and
produces paper-facing CSV/JSON/Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = EXPERIMENT_DIR / "results" / "calibration_study"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "longshot_bias"

LONGSHOT_BINS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.50]
LONGSHOT_LABELS = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-40%", "40-50%"]
MONTHLY_BANDS = {
    "0-2%": (0.0, 0.02),
    "0-10%": (0.0, 0.10),
    "10-20%": (0.10, 0.20),
}


def prepare_horizon(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """Convert an arbitrary positive-outcome probability into favorite/longshot view."""
    sub = df[df[prob_col].notna() & df["target"].notna()].copy()
    sub["p"] = sub[prob_col].astype(float)
    sub["target"] = sub["target"].astype(float)
    sub["favorite_prob"] = np.maximum(sub["p"], 1.0 - sub["p"])
    sub["longshot_prob"] = 1.0 - sub["favorite_prob"]
    sub["favorite_win"] = np.where(sub["p"] >= 0.5, sub["target"], 1.0 - sub["target"])
    sub["longshot_win"] = 1.0 - sub["favorite_win"]
    sub["month"] = sub["end_date"].dt.to_period("M").astype(str)
    return sub


def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise ratio with NaN when denominator is zero."""
    denom_adj = denom.replace(0.0, np.nan)
    return numer / denom_adj


def bin_table(sub: pd.DataFrame, horizon_name: str) -> pd.DataFrame:
    """Longshot calibration table by probability bin."""
    working = sub.copy()
    working["longshot_bin"] = pd.cut(
        working["longshot_prob"],
        bins=LONGSHOT_BINS,
        labels=LONGSHOT_LABELS,
        include_lowest=True,
        right=False,
    )
    grouped = (
        working.groupby("longshot_bin", observed=True)
        .agg(
            n=("condition_id", "count"),
            mean_longshot_prob=("longshot_prob", "mean"),
            observed_longshot_win=("longshot_win", "mean"),
            mean_favorite_prob=("favorite_prob", "mean"),
            observed_favorite_win=("favorite_win", "mean"),
        )
        .reset_index()
    )
    grouped["longshot_bias"] = grouped["mean_longshot_prob"] - grouped["observed_longshot_win"]
    grouped["favorite_bias"] = grouped["mean_favorite_prob"] - grouped["observed_favorite_win"]
    grouped["relative_longshot_overpricing"] = safe_ratio(
        grouped["mean_longshot_prob"],
        grouped["observed_longshot_win"],
    )
    grouped.insert(0, "horizon", horizon_name)
    return grouped


def monthly_table(
    sub: pd.DataFrame,
    horizon_name: str,
    band_name: str,
    lo: float,
    hi: float,
    min_month_count: int,
) -> pd.DataFrame:
    """Month-by-month longshot-bias table for a selected band."""
    band = sub[(sub["longshot_prob"] >= lo) & (sub["longshot_prob"] < hi)].copy()
    grouped = (
        band.groupby("month")
        .agg(
            n=("condition_id", "count"),
            mean_longshot_prob=("longshot_prob", "mean"),
            observed_longshot_win=("longshot_win", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_month_count].copy()
    grouped["longshot_bias"] = grouped["mean_longshot_prob"] - grouped["observed_longshot_win"]
    grouped["relative_longshot_overpricing"] = safe_ratio(
        grouped["mean_longshot_prob"],
        grouped["observed_longshot_win"],
    )
    grouped.insert(0, "band", band_name)
    grouped.insert(0, "horizon", horizon_name)
    return grouped


def category_table(
    sub: pd.DataFrame,
    horizon_name: str,
    lo: float,
    hi: float,
    min_count: int,
) -> pd.DataFrame:
    """Category heterogeneity within a chosen longshot band."""
    band = sub[(sub["longshot_prob"] >= lo) & (sub["longshot_prob"] < hi)].copy()
    grouped = (
        band.groupby("category")
        .agg(
            n=("condition_id", "count"),
            mean_longshot_prob=("longshot_prob", "mean"),
            observed_longshot_win=("longshot_win", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_count].copy()
    grouped["longshot_bias"] = grouped["mean_longshot_prob"] - grouped["observed_longshot_win"]
    grouped["relative_longshot_overpricing"] = safe_ratio(
        grouped["mean_longshot_prob"],
        grouped["observed_longshot_win"],
    )
    grouped = grouped.sort_values(["n", "relative_longshot_overpricing"], ascending=[False, False])
    grouped.insert(0, "band", "1-10%")
    grouped.insert(0, "horizon", horizon_name)
    return grouped


def write_summary_markdown(summary: dict, path: Path) -> None:
    """Write a concise markdown memo."""
    lines = [
        "# Longshot Bias Study",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Markets with final probabilities: `{summary['n_final']}`",
        f"- Markets with 24h probabilities: `{summary['n_24h']}`",
        "",
        "## Headline Results",
        "",
        (
            f"- Final horizon, `1-2%` longshots trade at `{summary['final_1_2_relative_overpricing']:.2f}x` "
            f"their realized frequency."
        ),
        (
            f"- 24h horizon, `0-2%` longshots trade at `{summary['h24_0_2_relative_overpricing']:.2f}x` "
            f"their realized frequency."
        ),
        (
            f"- 24h horizon, `0-10%` longshots trade at `{summary['h24_0_10_relative_overpricing']:.2f}x` "
            f"their realized frequency."
        ),
        (
            f"- 24h horizon, categories with the strongest `1-10%` longshot overpricing are "
            f"`{summary['top_24h_category']}` and `{summary['second_24h_category']}` in the filtered table."
        ),
        "",
        "## Interpretation",
        "",
        "These diagnostics are consistent with a longshot-bias story: very small",
        "tail probabilities are overpriced relative to realized frequencies, and",
        "the distortion is larger at tradable horizons than at the final print.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-level-csv",
        type=Path,
        default=CALIBRATION_DIR / "market_level.csv",
    )
    parser.add_argument("--min-month-count", type=int, default=50)
    parser.add_argument("--min-category-count", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.market_level_csv)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df[df["end_date"].notna() & (df["end_date"].dt.year < 2030)].copy()

    final_df = prepare_horizon(df, "p_final")
    h24_df = prepare_horizon(df, "p_24h")
    h168_df = prepare_horizon(df, "p_168h")

    bin_tables = pd.concat(
        [
            bin_table(final_df, "final"),
            bin_table(h24_df, "24h"),
            bin_table(h168_df, "168h"),
        ],
        ignore_index=True,
    )
    bin_tables.to_csv(RESULTS_DIR / "bin_metrics.csv", index=False)

    monthly_tables = []
    for horizon_name, sub in [("final", final_df), ("24h", h24_df)]:
        for band_name, (lo, hi) in MONTHLY_BANDS.items():
            monthly_tables.append(
                monthly_table(
                    sub,
                    horizon_name=horizon_name,
                    band_name=band_name,
                    lo=lo,
                    hi=hi,
                    min_month_count=args.min_month_count,
                )
            )
    monthly_df = pd.concat(monthly_tables, ignore_index=True)
    monthly_df.to_csv(RESULTS_DIR / "monthly_metrics.csv", index=False)

    category_tables = pd.concat(
        [
            category_table(
                final_df,
                horizon_name="final",
                lo=0.01,
                hi=0.10,
                min_count=args.min_category_count,
            ),
            category_table(
                h24_df,
                horizon_name="24h",
                lo=0.01,
                hi=0.10,
                min_count=args.min_category_count,
            ),
        ],
        ignore_index=True,
    )
    category_tables.to_csv(RESULTS_DIR / "category_metrics.csv", index=False)

    def fetch_metric(table: pd.DataFrame, horizon: str, label: str, column: str) -> float:
        row = table[(table["horizon"] == horizon) & (table["longshot_bin"] == label)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][column])

    def fetch_month_metric(table: pd.DataFrame, horizon: str, band: str, month: str, column: str) -> float | None:
        row = table[(table["horizon"] == horizon) & (table["band"] == band) & (table["month"] == month)]
        if row.empty:
            return None
        value = row.iloc[0][column]
        return None if pd.isna(value) else float(value)

    cat24 = category_tables[
        (category_tables["horizon"] == "24h")
        & category_tables["relative_longshot_overpricing"].notna()
        & (category_tables["longshot_bias"] > 0)
    ].copy()
    cat24 = cat24.sort_values(
        ["relative_longshot_overpricing", "longshot_bias", "n"],
        ascending=[False, False, False],
    )
    top_cat_names = cat24["category"].tolist()[:2]
    while len(top_cat_names) < 2:
        top_cat_names.append("N/A")

    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_final": int(len(final_df)),
        "n_24h": int(len(h24_df)),
        "n_168h": int(len(h168_df)),
        "final_1_2_relative_overpricing": fetch_metric(
            bin_tables,
            horizon="final",
            label="1-2%",
            column="relative_longshot_overpricing",
        ),
        "h24_0_2_relative_overpricing": float(
            monthly_df[
                (monthly_df["horizon"] == "24h")
                & (monthly_df["band"] == "0-2%")
                & monthly_df["relative_longshot_overpricing"].notna()
            ]["relative_longshot_overpricing"].mean()
        ),
        "h24_0_10_relative_overpricing": float(
            monthly_df[
                (monthly_df["horizon"] == "24h")
                & (monthly_df["band"] == "0-10%")
                & monthly_df["relative_longshot_overpricing"].notna()
            ]["relative_longshot_overpricing"].mean()
        ),
        "h24_0_10_bias_2026_02": fetch_month_metric(monthly_df, "24h", "0-10%", "2026-02", "longshot_bias"),
        "h24_0_10_bias_2026_03": fetch_month_metric(monthly_df, "24h", "0-10%", "2026-03", "longshot_bias"),
        "h24_0_10_bias_2026_04": fetch_month_metric(monthly_df, "24h", "0-10%", "2026-04", "longshot_bias"),
        "top_24h_category": top_cat_names[0],
        "second_24h_category": top_cat_names[1],
        "artifacts": {
            "bin_metrics_csv": str(RESULTS_DIR / "bin_metrics.csv"),
            "monthly_metrics_csv": str(RESULTS_DIR / "monthly_metrics.csv"),
            "category_metrics_csv": str(RESULTS_DIR / "category_metrics.csv"),
        },
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_markdown(summary, RESULTS_DIR / "summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
