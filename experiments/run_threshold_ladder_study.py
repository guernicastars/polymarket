#!/usr/bin/env python3
"""Threshold-ladder study for tail-aware forecasting diagnostics.

Motivation:
- A single binary market only gives one exceedance probability P(X > k).
- For fat-tailed or asymmetric outcomes, calibration at one threshold is not
  enough; we want more of the exceedance curve or distribution.
- Polymarket often lists multiple Over/Under thresholds on the same event.

This script reconstructs those threshold ladders from the market-level output of
``run_calibration_study.py`` and computes ladder-level diagnostics:
- monotonicity violations in the implied survival curve
- integrated Brier score across thresholds
- truncated first- and second-moment proxies from the exceedance curve
- variance proxies derived from those moments
- at-the-money threshold diagnostics to test whether local calibration tracks
  the full implied tail object
- simple tail-shape comparisons (exponential vs power-law fit on the right tail)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


EXPERIMENT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = EXPERIMENT_DIR / "results" / "calibration_study"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "threshold_ladders"

THRESHOLD_RE = re.compile(r"(?:O/U|Over/Under)\s*([0-9]+(?:\.[0-9]+)?)", re.I)
THRESHOLD_TEMPLATE_RE = re.compile(r"((?:O/U|Over/Under)\s*)([0-9]+(?:\.[0-9]+)?)", re.I)


@dataclass
class LadderSummary:
    ladder_key: str
    event_slug: str
    category: str
    n_markets: int
    n_thresholds: int
    min_threshold: float
    max_threshold: float
    curve_brier_final: float
    curve_brier_24h: float | None
    monotonicity_violations_final: int
    monotonicity_violations_24h: int | None
    monotonicity_violation_rate_final: float
    mean_proxy_final: float | None
    mean_proxy_24h: float | None
    second_moment_proxy_final: float | None
    second_moment_proxy_24h: float | None
    variance_proxy_final: float | None
    variance_proxy_24h: float | None
    realized_mean_proxy: float | None
    mean_proxy_error_final: float | None
    abs_mean_proxy_error_final: float | None
    atm_threshold_final: float | None
    atm_prob_final: float | None
    atm_brier_final: float | None
    isotonic_l1_gap_final: float | None
    isotonic_l2_gap_final: float | None
    adjacent_arbitrage_count_final: int
    adjacent_arbitrage_gross_edge_final: float
    max_adjacent_arbitrage_edge_final: float
    exp_r2_final: float | None
    power_r2_final: float | None
    tail_fit_winner_final: str


def parse_outcomes(value: str | list[str]) -> list[str]:
    """Parse serialized outcomes from CSV or pass lists through unchanged."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def parse_threshold(question: str) -> float | None:
    """Extract numeric O/U threshold from a question."""
    if not isinstance(question, str):
        return None
    match = THRESHOLD_RE.search(question)
    if not match:
        return None
    return float(match.group(1))


def normalize_ladder_key(question: str) -> str | None:
    """Normalize an O/U question into a ladder identifier.

    One event can contain multiple separate ladders, for example match games,
    set-specific games, or player props. Grouping only by event slug would
    incorrectly merge these into a fake ladder, so we replace the strike with a
    placeholder and group by the normalized question template.
    """
    if not isinstance(question, str):
        return None
    normalized = THRESHOLD_TEMPLATE_RE.sub(r"\1{threshold}", question)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized if "{threshold}" in normalized else None


def exceedance_probability(row: pd.Series, prob_col: str) -> float:
    """Return probability of the Over event for one threshold market."""
    prob = row.get(prob_col)
    if pd.isna(prob):
        return np.nan

    outcomes = parse_outcomes(row.get("outcomes"))
    positive = str(row.get("positive_outcome", ""))

    over_label = None
    for outcome in outcomes:
        if "over" in outcome.lower():
            over_label = outcome
            break

    if over_label is None:
        return np.nan

    return float(prob if positive == over_label else 1.0 - prob)


def exceedance_outcome(row: pd.Series) -> float:
    """Return 1 if the Over outcome won, else 0."""
    winning = str(row.get("winning_outcome", ""))
    outcomes = parse_outcomes(row.get("outcomes"))
    for outcome in outcomes:
        if "over" in outcome.lower():
            return float(winning == outcome)
    return np.nan


def truncated_mean_proxy(thresholds: np.ndarray, survival: np.ndarray) -> float | None:
    """Approximate a lower-truncated mean from a survival curve.

    For sorted thresholds k_1 < ... < k_n with survival S(k_i), use:
      k_1 + sum_i S(k_i) * (k_{i+1} - k_i)
    This is a truncated proxy, not a full mean, but it is already more
    informative than a single threshold probability.
    """
    if len(thresholds) < 2:
        return None
    delta = np.diff(thresholds)
    return float(thresholds[0] + np.sum(survival[:-1] * delta))


def truncated_second_moment_proxy(thresholds: np.ndarray, survival: np.ndarray) -> float | None:
    """Approximate a lower-truncated second moment from a survival curve.

    For nonnegative X, E[X^2] = 2 * integral x * P(X > x) dx. With the same
    left-Riemann approximation used for the mean proxy, we recover:
      k_1^2 + sum_i S(k_i) * (k_{i+1}^2 - k_i^2)
    """
    if len(thresholds) < 2:
        return None
    left = thresholds[:-1]
    right = thresholds[1:]
    return float(thresholds[0] ** 2 + np.sum(survival[:-1] * (right**2 - left**2)))


def variance_proxy(
    mean_proxy: float | None,
    second_moment_proxy: float | None,
) -> float | None:
    """Convert moment proxies into a truncated variance proxy."""
    if mean_proxy is None or second_moment_proxy is None:
        return None
    return float(max(second_moment_proxy - mean_proxy**2, 0.0))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute R^2 safely."""
    if len(y_true) < 2:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return None
    return 1.0 - ss_res / ss_tot


def fit_tail_shape(thresholds: np.ndarray, probs: np.ndarray) -> tuple[float | None, float | None, str]:
    """Compare exponential and power-law fits on the right tail."""
    mask = np.isfinite(thresholds) & np.isfinite(probs) & (probs > 1e-4) & (probs < 0.35)
    if np.sum(mask) < 3:
        return None, None, "insufficient"

    x = thresholds[mask]
    p = probs[mask]

    # Focus on the upper half of thresholds to approximate the right tail.
    median_thr = np.median(x)
    tail_mask = x >= median_thr
    if np.sum(tail_mask) >= 3:
        x = x[tail_mask]
        p = p[tail_mask]
    if len(x) < 3:
        return None, None, "insufficient"

    y = np.log(p)

    # Thin-tail proxy: exponential decay log p ~ a + b * k
    coef_exp = np.polyfit(x, y, deg=1)
    pred_exp = coef_exp[0] * x + coef_exp[1]
    exp_r2 = r2_score(y, pred_exp)

    # Fat-tail proxy: power-law decay log p ~ a + b * log k
    log_x = np.log(np.maximum(x, 1e-8))
    coef_power = np.polyfit(log_x, y, deg=1)
    pred_power = coef_power[0] * log_x + coef_power[1]
    power_r2 = r2_score(y, pred_power)

    if exp_r2 is None and power_r2 is None:
        winner = "insufficient"
    elif exp_r2 is None:
        winner = "power"
    elif power_r2 is None:
        winner = "exp"
    elif exp_r2 > power_r2 + 0.02:
        winner = "exp"
    elif power_r2 > exp_r2 + 0.02:
        winner = "power"
    else:
        winner = "tie"

    return exp_r2, power_r2, winner


def safe_correlation(x: pd.Series, y: pd.Series) -> float | None:
    """Compute a Pearson correlation when enough finite values exist."""
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def isotonic_gap(probs: np.ndarray) -> tuple[float, float]:
    """Distance from the nearest non-increasing exceedance curve."""
    if len(probs) < 2:
        return 0.0, 0.0
    x = np.arange(len(probs), dtype=np.float64)
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    fitted = ir.fit_transform(x, probs)
    diff = probs - fitted
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def adjacent_arbitrage_metrics(probs: np.ndarray, tol: float = 0.0) -> tuple[int, float, float]:
    """Compute adjacent-strike dominance violations and spread edges.

    For thresholds k_i < k_{i+1}, survival prices should satisfy
    P(X > k_i) >= P(X > k_{i+1}). If the reverse holds, buying the cheaper
    low-strike over and selling the richer high-strike over is a nonnegative
    payoff spread before fees.
    """
    if len(probs) < 2:
        return 0, 0.0, 0.0
    gaps = np.maximum(np.diff(probs) - tol, 0.0)
    return int(np.sum(gaps > 0.0)), float(np.sum(gaps)), float(np.max(gaps, initial=0.0))


def summarize_ladder(ladder_key: str, event_slug: str, group: pd.DataFrame) -> LadderSummary | None:
    """Aggregate one event-level threshold ladder."""
    agg = (
        group.groupby("threshold", as_index=False)
        .agg(
            category=("category", "first"),
            n_markets=("condition_id", "count"),
            p_final_over=("p_final_over", "mean"),
            p_24h_over=("p_24h_over", "mean"),
            y_over=("y_over", "mean"),
            total_volume=("volume_total", "sum"),
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )

    if len(agg) < 3:
        return None

    thresholds = agg["threshold"].to_numpy(dtype=np.float64)
    p_final = agg["p_final_over"].to_numpy(dtype=np.float64)
    p_24h = agg["p_24h_over"].to_numpy(dtype=np.float64)
    y = agg["y_over"].to_numpy(dtype=np.float64)

    valid_final = np.isfinite(p_final) & np.isfinite(y)
    if np.sum(valid_final) < 3:
        return None

    diff_final = np.diff(p_final[valid_final])
    violations_final = int(np.sum(diff_final > 0.02))

    valid_24h = np.isfinite(p_24h) & np.isfinite(y)
    if np.sum(valid_24h) >= 3:
        diff_24h = np.diff(p_24h[valid_24h])
        violations_24h = int(np.sum(diff_24h > 0.02))
        curve_brier_24h = float(np.mean((p_24h[valid_24h] - y[valid_24h]) ** 2))
        mean_proxy_24h = truncated_mean_proxy(thresholds[valid_24h], p_24h[valid_24h])
        second_moment_proxy_24h = truncated_second_moment_proxy(
            thresholds[valid_24h],
            p_24h[valid_24h],
        )
        variance_proxy_24h = variance_proxy(mean_proxy_24h, second_moment_proxy_24h)
    else:
        violations_24h = None
        curve_brier_24h = None
        mean_proxy_24h = None
        second_moment_proxy_24h = None
        variance_proxy_24h = None

    curve_brier_final = float(np.mean((p_final[valid_final] - y[valid_final]) ** 2))
    mean_proxy_final = truncated_mean_proxy(thresholds[valid_final], p_final[valid_final])
    second_moment_proxy_final = truncated_second_moment_proxy(
        thresholds[valid_final],
        p_final[valid_final],
    )
    variance_proxy_final = variance_proxy(mean_proxy_final, second_moment_proxy_final)
    realized_mean = truncated_mean_proxy(thresholds[valid_final], y[valid_final])
    exp_r2, power_r2, winner = fit_tail_shape(thresholds[valid_final], p_final[valid_final])
    atm_idx = int(np.argmin(np.abs(p_final[valid_final] - 0.5)))
    atm_threshold_final = float(thresholds[valid_final][atm_idx])
    atm_prob_final = float(p_final[valid_final][atm_idx])
    atm_brier_final = float((p_final[valid_final][atm_idx] - y[valid_final][atm_idx]) ** 2)
    isotonic_l1_gap_final, isotonic_l2_gap_final = isotonic_gap(p_final[valid_final])
    (
        adjacent_arbitrage_count_final,
        adjacent_arbitrage_gross_edge_final,
        max_adjacent_arbitrage_edge_final,
    ) = adjacent_arbitrage_metrics(p_final[valid_final], tol=0.0)
    mean_proxy_error_final = (
        None if mean_proxy_final is None or realized_mean is None
        else float(mean_proxy_final - realized_mean)
    )

    return LadderSummary(
        ladder_key=ladder_key,
        event_slug=event_slug,
        category=str(agg["category"].iloc[0]),
        n_markets=int(int(agg["n_markets"].sum())),
        n_thresholds=int(len(agg)),
        min_threshold=float(thresholds.min()),
        max_threshold=float(thresholds.max()),
        curve_brier_final=curve_brier_final,
        curve_brier_24h=curve_brier_24h,
        monotonicity_violations_final=violations_final,
        monotonicity_violations_24h=violations_24h,
        monotonicity_violation_rate_final=float(violations_final / max(len(thresholds) - 1, 1)),
        mean_proxy_final=mean_proxy_final,
        mean_proxy_24h=mean_proxy_24h,
        second_moment_proxy_final=second_moment_proxy_final,
        second_moment_proxy_24h=second_moment_proxy_24h,
        variance_proxy_final=variance_proxy_final,
        variance_proxy_24h=variance_proxy_24h,
        realized_mean_proxy=realized_mean,
        mean_proxy_error_final=mean_proxy_error_final,
        abs_mean_proxy_error_final=(
            None if mean_proxy_error_final is None else float(abs(mean_proxy_error_final))
        ),
        atm_threshold_final=atm_threshold_final,
        atm_prob_final=atm_prob_final,
        atm_brier_final=atm_brier_final,
        isotonic_l1_gap_final=isotonic_l1_gap_final,
        isotonic_l2_gap_final=isotonic_l2_gap_final,
        adjacent_arbitrage_count_final=adjacent_arbitrage_count_final,
        adjacent_arbitrage_gross_edge_final=adjacent_arbitrage_gross_edge_final,
        max_adjacent_arbitrage_edge_final=max_adjacent_arbitrage_edge_final,
        exp_r2_final=exp_r2,
        power_r2_final=power_r2,
        tail_fit_winner_final=winner,
    )


def write_summary_markdown(summary: dict, path: Path) -> None:
    """Write a short markdown summary."""
    lines = [
        "# Threshold Ladder Study",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- O/U markets analyzed: `{summary['n_markets_ou']}`",
        f"- Ladder events with >=3 thresholds: `{summary['n_ladders']}`",
        f"- Median thresholds per ladder: `{summary['median_thresholds_per_ladder']}`",
        "",
        "## Ladder-Level Averages",
        "",
        f"- Mean final integrated Brier: `{summary['mean_curve_brier_final']:.4f}`",
        f"- Mean final ATM Brier: `{summary['mean_atm_brier_final']:.4f}`",
        f"- Mean final monotonicity violation rate: `{summary['mean_monotonicity_violation_rate_final']:.4f}`",
        f"- Mean abs. truncated-mean error: `{summary['mean_abs_mean_proxy_error_final']:.4f}`",
        f"- Share with adjacent arbitrage: `{summary['share_with_any_adjacent_arbitrage_final']:.4f}`",
        f"- Share with material (>2c) adjacent arbitrage: `{summary['share_with_material_adjacent_arbitrage_final']:.4f}`",
        f"- Mean adjacent arbitrage gross edge: `{summary['mean_adjacent_arbitrage_gross_edge_final']:.4f}`",
        f"- Mean isotonic L1 repair gap: `{summary['mean_isotonic_l1_gap_final']:.4f}`",
        f"- Tail-fit winner counts: `{summary['tail_fit_winner_counts']}`",
        "",
        "## Decoupling Diagnostics",
        "",
        f"- Corr(ATM Brier, curve Brier): `{summary['corr_atm_brier_vs_curve_brier']:.4f}`",
        f"- Corr(ATM Brier, abs. mean-proxy error): `{summary['corr_atm_brier_vs_abs_mean_proxy_error']:.4f}`",
        f"- Corr(ATM Brier, adjacent arbitrage gross edge): `{summary['corr_atm_brier_vs_adjacent_arbitrage_gross_edge']:.4f}`",
        f"- Share of low-ATM-Brier ladders with any monotonicity violation: `{summary['share_low_atm_brier_with_any_violation']:.4f}`",
        f"- Share of low-ATM-Brier ladders with high abs. mean error: `{summary['share_low_atm_brier_with_high_abs_mean_error']:.4f}`",
        f"- Share of low-ATM-Brier ladders with any adjacent arbitrage: `{summary['share_low_atm_brier_with_any_adjacent_arbitrage']:.4f}`",
        f"- Share of low-ATM-Brier ladders with material (>2c) adjacent arbitrage: `{summary['share_low_atm_brier_with_material_adjacent_arbitrage']:.4f}`",
        "",
        "## Interpretation",
        "",
        "These ladders are a first step toward distributional forecasting in",
        "prediction markets. A single binary contract only gives one tail",
        "probability, but a ladder of thresholds approximates an exceedance",
        "curve, which is much closer to the object needed for expected-value",
        "and tail-risk analysis.",
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
    parser.add_argument("--min-thresholds", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.market_level_csv)
    df = df[df["question"].astype(str).str.contains(r"O/U|Over/Under", case=False, regex=True)].copy()
    df["threshold"] = df["question"].map(parse_threshold)
    df["ladder_template"] = df["question"].map(normalize_ladder_key)
    df = df[df["threshold"].notna()].copy()
    df["y_over"] = df.apply(exceedance_outcome, axis=1)
    df["p_final_over"] = df.apply(lambda row: exceedance_probability(row, "p_final"), axis=1)
    df["p_24h_over"] = df.apply(lambda row: exceedance_probability(row, "p_24h"), axis=1)
    df = df[
        df["event_slug"].notna()
        & (df["event_slug"].astype(str) != "")
        & df["ladder_template"].notna()
    ]

    ladder_rows = []
    for (event_slug, ladder_template), group in df.groupby(["event_slug", "ladder_template"], sort=False):
        if group["threshold"].nunique() < args.min_thresholds:
            continue
        summary = summarize_ladder(
            ladder_key=f"{event_slug}::{ladder_template}",
            event_slug=str(event_slug),
            group=group,
        )
        if summary is not None:
            ladder_rows.append(asdict(summary))

    ladders_df = pd.DataFrame(ladder_rows)
    if ladders_df.empty:
        raise RuntimeError("No threshold ladders found.")

    ladders_df = ladders_df.sort_values(
        ["n_thresholds", "n_markets", "curve_brier_final"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    ladder_path = RESULTS_DIR / "ladder_metrics.csv"
    ladders_df.to_csv(ladder_path, index=False)

    category_metrics = (
        ladders_df.groupby("category", as_index=False)
        .agg(
            n_ladders=("ladder_key", "count"),
            mean_curve_brier_final=("curve_brier_final", "mean"),
            mean_atm_brier_final=("atm_brier_final", "mean"),
            mean_abs_mean_proxy_error_final=("abs_mean_proxy_error_final", "mean"),
            mean_variance_proxy_final=("variance_proxy_final", "mean"),
            mean_isotonic_l1_gap_final=("isotonic_l1_gap_final", "mean"),
            mean_adjacent_arbitrage_gross_edge_final=("adjacent_arbitrage_gross_edge_final", "mean"),
            share_with_any_monotonicity_violation_final=(
                "monotonicity_violations_final",
                lambda s: float(np.mean(np.asarray(s) > 0)),
            ),
            share_with_any_adjacent_arbitrage_final=(
                "adjacent_arbitrage_count_final",
                lambda s: float(np.mean(np.asarray(s) > 0)),
            ),
        )
        .sort_values(["n_ladders", "mean_curve_brier_final"], ascending=[False, True])
        .reset_index(drop=True)
    )
    category_metrics_path = RESULTS_DIR / "category_metrics.csv"
    category_metrics.to_csv(category_metrics_path, index=False)

    top_examples = ladders_df.head(20)[
        [
            "ladder_key",
            "event_slug",
            "category",
            "n_markets",
            "n_thresholds",
            "curve_brier_final",
            "atm_brier_final",
            "abs_mean_proxy_error_final",
            "monotonicity_violation_rate_final",
            "adjacent_arbitrage_gross_edge_final",
            "max_adjacent_arbitrage_edge_final",
            "tail_fit_winner_final",
        ]
    ]
    top_examples_path = RESULTS_DIR / "top_ladders.csv"
    top_examples.to_csv(top_examples_path, index=False)

    low_atm_cutoff = float(ladders_df["atm_brier_final"].quantile(0.25))
    high_abs_mean_error_cutoff = float(ladders_df["abs_mean_proxy_error_final"].quantile(0.75))
    decoupling_examples = ladders_df[
        (ladders_df["atm_brier_final"] <= low_atm_cutoff)
        & (
            (ladders_df["monotonicity_violations_final"] > 0)
            | (ladders_df["abs_mean_proxy_error_final"] >= high_abs_mean_error_cutoff)
        )
    ][
        [
            "ladder_key",
            "event_slug",
            "category",
            "n_thresholds",
            "atm_threshold_final",
            "atm_brier_final",
            "curve_brier_final",
            "abs_mean_proxy_error_final",
            "monotonicity_violation_rate_final",
            "adjacent_arbitrage_gross_edge_final",
            "max_adjacent_arbitrage_edge_final",
        ]
    ].sort_values(
        ["atm_brier_final", "adjacent_arbitrage_gross_edge_final", "abs_mean_proxy_error_final", "n_thresholds"],
        ascending=[True, False, False, False],
    )
    decoupling_examples_path = RESULTS_DIR / "decoupling_examples.csv"
    decoupling_examples.to_csv(decoupling_examples_path, index=False)

    tail_counts = ladders_df["tail_fit_winner_final"].value_counts(dropna=False).to_dict()
    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_markets_ou": int(len(df)),
        "n_ladders": int(len(ladders_df)),
        "median_thresholds_per_ladder": float(ladders_df["n_thresholds"].median()),
        "mean_thresholds_per_ladder": float(ladders_df["n_thresholds"].mean()),
        "mean_curve_brier_final": float(ladders_df["curve_brier_final"].mean()),
        "mean_atm_brier_final": float(ladders_df["atm_brier_final"].mean()),
        "median_curve_brier_final": float(ladders_df["curve_brier_final"].median()),
        "mean_monotonicity_violation_rate_final": float(
            ladders_df["monotonicity_violation_rate_final"].mean()
        ),
        "share_with_any_monotonicity_violation_final": float(
            np.mean(ladders_df["monotonicity_violations_final"] > 0)
        ),
        "mean_abs_mean_proxy_error_final": float(ladders_df["abs_mean_proxy_error_final"].mean()),
        "median_abs_mean_proxy_error_final": float(ladders_df["abs_mean_proxy_error_final"].median()),
        "mean_variance_proxy_final": float(ladders_df["variance_proxy_final"].dropna().mean()),
        "mean_isotonic_l1_gap_final": float(ladders_df["isotonic_l1_gap_final"].mean()),
        "mean_isotonic_l2_gap_final": float(ladders_df["isotonic_l2_gap_final"].mean()),
        "share_with_any_adjacent_arbitrage_final": float(
            np.mean(ladders_df["adjacent_arbitrage_count_final"] > 0)
        ),
        "share_with_material_adjacent_arbitrage_final": float(
            np.mean(ladders_df["max_adjacent_arbitrage_edge_final"] > 0.02)
        ),
        "mean_adjacent_arbitrage_gross_edge_final": float(
            ladders_df["adjacent_arbitrage_gross_edge_final"].mean()
        ),
        "mean_max_adjacent_arbitrage_edge_final": float(
            ladders_df["max_adjacent_arbitrage_edge_final"].mean()
        ),
        "corr_atm_brier_vs_curve_brier": safe_correlation(
            ladders_df["atm_brier_final"],
            ladders_df["curve_brier_final"],
        ),
        "corr_atm_brier_vs_abs_mean_proxy_error": safe_correlation(
            ladders_df["atm_brier_final"],
            ladders_df["abs_mean_proxy_error_final"],
        ),
        "corr_atm_brier_vs_adjacent_arbitrage_gross_edge": safe_correlation(
            ladders_df["atm_brier_final"],
            ladders_df["adjacent_arbitrage_gross_edge_final"],
        ),
        "corr_curve_brier_vs_abs_mean_proxy_error": safe_correlation(
            ladders_df["curve_brier_final"],
            ladders_df["abs_mean_proxy_error_final"],
        ),
        "low_atm_brier_cutoff_q25": low_atm_cutoff,
        "high_abs_mean_error_cutoff_q75": high_abs_mean_error_cutoff,
        "share_low_atm_brier_with_any_violation": float(
            np.mean(
                ladders_df.loc[
                    ladders_df["atm_brier_final"] <= low_atm_cutoff,
                    "monotonicity_violations_final",
                ] > 0
            )
        ),
        "share_low_atm_brier_with_high_abs_mean_error": float(
            np.mean(
                ladders_df.loc[
                    ladders_df["atm_brier_final"] <= low_atm_cutoff,
                    "abs_mean_proxy_error_final",
                ] >= high_abs_mean_error_cutoff
            )
        ),
        "share_low_atm_brier_with_any_adjacent_arbitrage": float(
            np.mean(
                ladders_df.loc[
                    ladders_df["atm_brier_final"] <= low_atm_cutoff,
                    "adjacent_arbitrage_count_final",
                ] > 0
            )
        ),
        "share_low_atm_brier_with_material_adjacent_arbitrage": float(
            np.mean(
                ladders_df.loc[
                    ladders_df["atm_brier_final"] <= low_atm_cutoff,
                    "max_adjacent_arbitrage_edge_final",
                ] > 0.02
            )
        ),
        "tail_fit_winner_counts": tail_counts,
        "category_counts_top10": {
            str(k): int(v)
            for k, v in ladders_df["category"].value_counts().head(10).to_dict().items()
        },
        "artifacts": {
            "ladder_metrics_csv": str(ladder_path),
            "category_metrics_csv": str(category_metrics_path),
            "top_ladders_csv": str(top_examples_path),
            "decoupling_examples_csv": str(decoupling_examples_path),
        },
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_summary_markdown(summary, RESULTS_DIR / "summary.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
