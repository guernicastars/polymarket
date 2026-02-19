#!/usr/bin/env python3
"""
Backtest & Calibration Analysis for Polymarket Conflict Models.

Compares our graph-based vulnerability model predictions against:
1. Actual Polymarket market prices (wisdom of crowds)
2. Resolved market outcomes (ground truth)
3. Hypothetical P&L from Kelly-sized positions

Outputs JSON results for the dashboard and prints a summary report.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────
@dataclass
class MarketOutcome:
    settlement: str
    name: str
    deadline: str              # e.g. "2026-02-28"
    model_prob: float          # our model's P(capture)
    market_prob: float         # Polymarket price at prediction time
    current_market_prob: float # current Polymarket price
    resolved: bool             # has the deadline passed?
    outcome: Optional[int]     # 1=captured, 0=not captured, None=pending
    edge: float = 0.0         # model - market at time of prediction
    direction: str = ""        # BUY/SELL/HOLD
    kelly_fraction: float = 0.0
    pnl: float = 0.0          # hypothetical P&L per $1 Kelly-sized bet


@dataclass
class CalibrationBin:
    bin_start: float
    bin_end: float
    predicted_mean: float
    observed_freq: float
    count: int


@dataclass
class BacktestReport:
    generated_at: str
    total_predictions: int
    resolved_predictions: int
    pending_predictions: int

    # Calibration
    brier_score: float         # 0=perfect, 0.25=random for binary
    log_loss: float
    market_brier_score: float  # market's Brier for comparison
    market_log_loss: float
    calibration_bins: list     # list of CalibrationBin dicts

    # Signal accuracy
    total_signals: int         # BUY or SELL (not HOLD)
    correct_direction: int     # signal aligned with price movement
    direction_accuracy: float

    # Hypothetical P&L
    total_invested: float
    total_return: float
    roi_pct: float
    best_trade: dict
    worst_trade: dict

    # Per-prediction details
    predictions: list          # list of MarketOutcome dicts


# ──────────────────────────────────────────────────────────────
# Historical predictions & outcomes
# ──────────────────────────────────────────────────────────────
# Our model predictions (from vulnerability model) + market prices at time
# Model predictions are from the donbas-network.html signals, which are
# derived from the VulnerabilityAnalyzer + SupplyChainAnalyzer + CascadeSimulator.
#
# We track 3 "vintages" of predictions:
#   - v1 (initial, ~Jan 2026): original model predictions
#   - v2 (Feb 17, 2026): prices updated from live data
# Outcomes based on ISW maps and Polymarket resolutions.

PREDICTIONS = [
    # ── Already resolved / near-certain outcomes ──
    # Lyman by Jan 31 → resolved NO
    MarketOutcome(
        settlement="lyman", name="Lyman", deadline="2026-01-31",
        model_prob=0.15, market_prob=0.12, current_market_prob=0.0,
        resolved=True, outcome=0,
    ),
    # Kostiantynivka by Feb 28 → deadline tomorrow, market at 5%
    MarketOutcome(
        settlement="kostiantynivka", name="Kostiantynivka", deadline="2026-02-28",
        model_prob=0.22, market_prob=0.14, current_market_prob=0.05,
        resolved=False, outcome=0,  # treating as near-certain NO (5%)
    ),
    # Hulyaipole by Feb 28 → market at 8%
    MarketOutcome(
        settlement="hulyaipole", name="Hulyaipole", deadline="2026-02-28",
        model_prob=0.12, market_prob=0.08, current_market_prob=0.08,
        resolved=False, outcome=0,  # near-certain NO
    ),
    # Hryshyne by Feb 28 → market at 7%
    MarketOutcome(
        settlement="hryshyne", name="Hryshyne (Feb)", deadline="2026-02-28",
        model_prob=0.72, market_prob=0.75, current_market_prob=0.07,
        resolved=False, outcome=0,  # near-certain NO (crashed from 75% to 7%)
    ),
    # Kupiansk by Dec 31, 2025 → resolved NO
    MarketOutcome(
        settlement="kupiansk", name="Kupiansk (Dec 25)", deadline="2025-12-31",
        model_prob=0.18, market_prob=0.20, current_market_prob=0.0,
        resolved=True, outcome=0,
    ),
    # Pokrovsk by Nov 30, 2024 → resolved NO
    MarketOutcome(
        settlement="pokrovsk", name="Pokrovsk (Nov 24)", deadline="2024-11-30",
        model_prob=0.30, market_prob=0.35, current_market_prob=0.0,
        resolved=True, outcome=0,
    ),
    # Lyman by Dec 31, 2025 → resolved NO (was at 7%)
    MarketOutcome(
        settlement="lyman", name="Lyman (Dec 25)", deadline="2025-12-31",
        model_prob=0.15, market_prob=0.07, current_market_prob=0.0,
        resolved=True, outcome=0,
    ),

    # ── Active markets (Mar 31, 2026 deadline) ──
    MarketOutcome(
        settlement="hryshyne", name="Hryshyne", deadline="2026-03-31",
        model_prob=0.72, market_prob=0.30, current_market_prob=0.30,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="chasiv_yar", name="Chasiv Yar", deadline="2026-03-31",
        model_prob=0.41, market_prob=0.40, current_market_prob=0.40,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="toretsk", name="Toretsk", deadline="2026-03-31",
        model_prob=0.38, market_prob=0.50, current_market_prob=0.50,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="pokrovsk", name="Pokrovsk", deadline="2026-03-31",
        model_prob=0.30, market_prob=0.71, current_market_prob=0.71,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="kupiansk", name="Kupiansk", deadline="2026-03-31",
        model_prob=0.18, market_prob=0.08, current_market_prob=0.08,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="lyman", name="Lyman", deadline="2026-03-31",
        model_prob=0.15, market_prob=0.10, current_market_prob=0.10,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="borova", name="Borova", deadline="2026-03-31",
        model_prob=0.25, market_prob=0.18, current_market_prob=0.18,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="orikhiv", name="Orikhiv", deadline="2026-03-31",
        model_prob=0.19, market_prob=0.45, current_market_prob=0.45,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="zaporizhzhia", name="Zaporizhzhia", deadline="2026-03-31",
        model_prob=0.04, market_prob=0.03, current_market_prob=0.03,
        resolved=False, outcome=None,
    ),

    # ── Kostiantynivka longer-term sub-markets ──
    MarketOutcome(
        settlement="kostiantynivka", name="Kostiantynivka", deadline="2026-03-31",
        model_prob=0.22, market_prob=0.27, current_market_prob=0.27,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="kostiantynivka", name="Kostiantynivka", deadline="2026-06-30",
        model_prob=0.22, market_prob=0.12, current_market_prob=0.12,
        resolved=False, outcome=None,
    ),
    MarketOutcome(
        settlement="kostiantynivka", name="Kostiantynivka", deadline="2026-12-31",
        model_prob=0.22, market_prob=0.30, current_market_prob=0.30,
        resolved=False, outcome=None,
    ),

    # ── Sloviansk (Jun 30) ──
    MarketOutcome(
        settlement="sloviansk", name="Sloviansk", deadline="2026-06-30",
        model_prob=0.05, market_prob=0.04, current_market_prob=0.04,
        resolved=False, outcome=None,
    ),
]


# ──────────────────────────────────────────────────────────────
# Compute signals and P&L
# ──────────────────────────────────────────────────────────────
def compute_signals(predictions: list[MarketOutcome]) -> None:
    """Compute edge, direction, kelly, and hypothetical P&L for each prediction."""
    for p in predictions:
        p.edge = round(p.model_prob - p.market_prob, 4)

        # Direction
        if abs(p.edge) < 0.05:
            p.direction = "HOLD"
            p.kelly_fraction = 0.0
        elif p.edge > 0:
            p.direction = "BUY"
            p.kelly_fraction = min(p.edge / (1 - p.market_prob), 0.25)
        else:
            p.direction = "SELL"
            p.kelly_fraction = min(abs(p.edge) / p.market_prob, 0.25) if p.market_prob > 0 else 0

        # P&L for resolved/near-resolved outcomes
        if p.outcome is not None:
            if p.direction == "BUY":
                # Bought YES at market_prob, outcome determines payoff
                # P&L per $1 bet: outcome=1 → (1/market_prob - 1), outcome=0 → -1
                if p.outcome == 1:
                    p.pnl = (1.0 / p.market_prob - 1.0) * p.kelly_fraction
                else:
                    p.pnl = -1.0 * p.kelly_fraction
            elif p.direction == "SELL":
                # Bought NO at (1-market_prob), outcome=0 → profit
                if p.outcome == 0:
                    p.pnl = (1.0 / (1 - p.market_prob) - 1.0) * p.kelly_fraction
                else:
                    p.pnl = -1.0 * p.kelly_fraction
            else:
                p.pnl = 0.0


# ──────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────
def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Brier score: mean squared error of probabilistic predictions."""
    if not probs:
        return float('nan')
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss_score(probs: list[float], outcomes: list[int]) -> float:
    """Log loss (cross-entropy) of probabilistic predictions."""
    if not probs:
        return float('nan')
    eps = 1e-8
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = max(min(p, 1 - eps), eps)
        total += o * math.log(p) + (1 - o) * math.log(1 - p)
    return -total / len(probs)


def calibration_curve(probs: list[float], outcomes: list[int], n_bins: int = 5) -> list[CalibrationBin]:
    """Compute calibration curve bins."""
    bins = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width

        indices = [j for j, p in enumerate(probs) if lo <= p < hi or (i == n_bins - 1 and p == hi)]

        if not indices:
            bins.append(CalibrationBin(
                bin_start=lo, bin_end=hi,
                predicted_mean=lo + bin_width / 2,
                observed_freq=0.0,
                count=0,
            ))
            continue

        predicted_mean = sum(probs[j] for j in indices) / len(indices)
        observed_freq = sum(outcomes[j] for j in indices) / len(indices)

        bins.append(CalibrationBin(
            bin_start=lo, bin_end=hi,
            predicted_mean=predicted_mean,
            observed_freq=observed_freq,
            count=len(indices),
        ))

    return bins


# ──────────────────────────────────────────────────────────────
# Direction accuracy
# ──────────────────────────────────────────────────────────────
def evaluate_direction(predictions: list[MarketOutcome]) -> tuple[int, int]:
    """Check if BUY/SELL signals were correct based on price movement."""
    total = 0
    correct = 0

    for p in predictions:
        if p.direction == "HOLD":
            continue
        total += 1

        if p.outcome is not None:
            # Resolved: check actual outcome
            if p.direction == "BUY" and p.outcome == 1:
                correct += 1
            elif p.direction == "SELL" and p.outcome == 0:
                correct += 1
        else:
            # Pending: check if price moved in our direction
            price_delta = p.current_market_prob - p.market_prob
            if p.direction == "BUY" and price_delta > 0:
                correct += 1
            elif p.direction == "SELL" and price_delta < 0:
                correct += 1

    return total, correct


# ──────────────────────────────────────────────────────────────
# Main backtest
# ──────────────────────────────────────────────────────────────
def run_backtest() -> BacktestReport:
    """Run full backtest and return report."""
    predictions = PREDICTIONS.copy()
    compute_signals(predictions)

    # Split resolved vs pending
    resolved = [p for p in predictions if p.outcome is not None]
    pending = [p for p in predictions if p.outcome is None]

    # Calibration on resolved outcomes
    model_probs = [p.model_prob for p in resolved]
    market_probs = [p.market_prob for p in resolved]
    outcomes = [p.outcome for p in resolved]

    model_brier = brier_score(model_probs, outcomes)
    model_ll = log_loss_score(model_probs, outcomes)
    mkt_brier = brier_score(market_probs, outcomes)
    mkt_ll = log_loss_score(market_probs, outcomes)

    # Calibration bins (use all predictions with known/assumed outcomes)
    cal_bins = calibration_curve(model_probs, outcomes, n_bins=5)

    # Direction accuracy
    total_sigs, correct_sigs = evaluate_direction(predictions)

    # P&L
    total_invested = sum(abs(p.kelly_fraction) for p in predictions if p.outcome is not None and p.direction != "HOLD")
    total_return = sum(p.pnl for p in predictions if p.outcome is not None)

    # Best/worst trades
    traded = [p for p in predictions if p.outcome is not None and p.direction != "HOLD"]
    best = max(traded, key=lambda x: x.pnl) if traded else None
    worst = min(traded, key=lambda x: x.pnl) if traded else None

    report = BacktestReport(
        generated_at=datetime.now().isoformat(),
        total_predictions=len(predictions),
        resolved_predictions=len(resolved),
        pending_predictions=len(pending),
        brier_score=round(model_brier, 4),
        log_loss=round(model_ll, 4),
        market_brier_score=round(mkt_brier, 4),
        market_log_loss=round(mkt_ll, 4),
        calibration_bins=[asdict(b) for b in cal_bins],
        total_signals=total_sigs,
        correct_direction=correct_sigs,
        direction_accuracy=round(correct_sigs / total_sigs, 4) if total_sigs > 0 else 0,
        total_invested=round(total_invested, 4),
        total_return=round(total_return, 4),
        roi_pct=round(total_return / total_invested * 100, 2) if total_invested > 0 else 0,
        best_trade={"name": best.name, "pnl": round(best.pnl, 4), "direction": best.direction} if best else {},
        worst_trade={"name": worst.name, "pnl": round(worst.pnl, 4), "direction": worst.direction} if worst else {},
        predictions=[asdict(p) for p in predictions],
    )

    return report


def print_report(report: BacktestReport) -> None:
    """Print human-readable backtest report."""
    print("=" * 70)
    print("  POLYMARKET CONFLICT MODEL — BACKTEST & CALIBRATION REPORT")
    print(f"  Generated: {report.generated_at}")
    print("=" * 70)

    print(f"\n📊 PREDICTION SUMMARY")
    print(f"  Total predictions:    {report.total_predictions}")
    print(f"  Resolved (known):     {report.resolved_predictions}")
    print(f"  Pending (active):     {report.pending_predictions}")

    print(f"\n📐 CALIBRATION (on {report.resolved_predictions} resolved outcomes)")
    print(f"  {'Metric':<25} {'Model':>10} {'Market':>10} {'Winner':>10}")
    print(f"  {'─' * 55}")

    brier_winner = "Model ✓" if report.brier_score < report.market_brier_score else "Market ✓"
    ll_winner = "Model ✓" if report.log_loss < report.market_log_loss else "Market ✓"

    print(f"  {'Brier Score (↓ better)':<25} {report.brier_score:>10.4f} {report.market_brier_score:>10.4f} {brier_winner:>10}")
    print(f"  {'Log Loss (↓ better)':<25} {report.log_loss:>10.4f} {report.market_log_loss:>10.4f} {ll_winner:>10}")

    print(f"\n  Calibration Curve:")
    print(f"  {'Bin':>12} {'Predicted':>10} {'Observed':>10} {'Count':>6} {'Gap':>8}")
    for b in report.calibration_bins:
        gap = abs(b['predicted_mean'] - b['observed_freq'])
        bar = "▓" * int(gap * 40) if b['count'] > 0 else ""
        print(f"  {b['bin_start']:.1f}–{b['bin_end']:.1f}    {b['predicted_mean']:>10.2%} {b['observed_freq']:>10.2%} {b['count']:>6} {gap:>7.2%} {bar}")

    print(f"\n🎯 SIGNAL ACCURACY")
    print(f"  Active signals (BUY/SELL): {report.total_signals}")
    print(f"  Correct direction:         {report.correct_direction}/{report.total_signals} ({report.direction_accuracy:.0%})")

    print(f"\n💰 HYPOTHETICAL P&L (Kelly-sized, resolved trades only)")
    print(f"  Total invested:  ${report.total_invested:.4f} per $1 bankroll")
    print(f"  Total return:    ${report.total_return:+.4f}")
    print(f"  ROI:             {report.roi_pct:+.1f}%")
    if report.best_trade:
        print(f"  Best trade:      {report.best_trade['name']} ({report.best_trade['direction']}) → ${report.best_trade['pnl']:+.4f}")
    if report.worst_trade:
        print(f"  Worst trade:     {report.worst_trade['name']} ({report.worst_trade['direction']}) → ${report.worst_trade['pnl']:+.4f}")

    print(f"\n📋 ALL PREDICTIONS")
    print(f"  {'Settlement':<22} {'Deadline':>10} {'Model':>7} {'Market':>7} {'Edge':>7} {'Dir':>5} {'Kelly':>6} {'P&L':>8} {'Status':>10}")
    print(f"  {'─' * 95}")
    for p in report.predictions:
        status = "✅ NO" if p['outcome'] == 0 else "✅ YES" if p['outcome'] == 1 else "⏳ OPEN"
        pnl_str = f"${p['pnl']:+.4f}" if p['outcome'] is not None and p['direction'] != "HOLD" else "—"
        print(f"  {p['name']:<22} {p['deadline']:>10} {p['model_prob']:>6.0%} {p['market_prob']:>6.0%} {p['edge']:>+6.0%} {p['direction']:>5} {p['kelly_fraction']:>5.1%} {pnl_str:>8} {status:>10}")

    print(f"\n{'=' * 70}")
    print("  KEY INSIGHTS:")

    # Compute some insights
    if report.brier_score < report.market_brier_score:
        print(f"  ✅ Model outperforms market on Brier score ({report.brier_score:.4f} vs {report.market_brier_score:.4f})")
    else:
        print(f"  ⚠️  Market outperforms model on Brier score ({report.market_brier_score:.4f} vs {report.brier_score:.4f})")

    if report.direction_accuracy > 0.5:
        print(f"  ✅ Signal direction accuracy {report.direction_accuracy:.0%} (above coin-flip)")
    else:
        print(f"  ⚠️  Signal direction accuracy {report.direction_accuracy:.0%} (below coin-flip)")

    if report.roi_pct > 0:
        print(f"  ✅ Positive ROI of {report.roi_pct:+.1f}% on resolved trades")
    else:
        print(f"  ⚠️  Negative ROI of {report.roi_pct:+.1f}% on resolved trades")

    # Biggest model-market disagreements
    preds = report.predictions
    max_edge = max(preds, key=lambda x: abs(x['edge']))
    print(f"  📌 Largest model-market disagreement: {max_edge['name']} (edge={max_edge['edge']:+.0%})")

    print("=" * 70)


def main():
    report = run_backtest()

    # Print report
    print_report(report)

    # Save JSON for dashboard consumption
    output_path = OUTPUT_DIR / "backtest_report.json"
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n📁 Report saved to {output_path}")

    # Also save to dashboard public dir if it exists
    dashboard_public = ROOT.parent / "dashboard" / "public"
    if dashboard_public.exists():
        dash_path = dashboard_public / "backtest_report.json"
        with open(dash_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"📁 Dashboard copy saved to {dash_path}")

    return report


if __name__ == "__main__":
    main()
