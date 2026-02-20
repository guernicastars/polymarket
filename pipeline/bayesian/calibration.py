"""Calibration tracking: rolling Brier score, reliability diagrams.

Tracks whether the Bayesian combiner adds value vs market price
by comparing prediction accuracy over time.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Stores a single prediction for calibration tracking."""

    condition_id: str
    source: str                    # 'bayesian', 'gnn', 'composite'
    predicted_prob: float
    market_price: float
    actual_outcome: Optional[float]  # None until resolved
    timestamp: datetime


class CalibrationTracker:
    """Tracks calibration quality of the Bayesian combiner.

    Accumulates (predicted_prob, market_price, actual_outcome) tuples
    and computes Brier scores, calibration curves, and reliability
    adjustments per source.
    """

    def __init__(self, buffer_size: int = 2000) -> None:
        self._buffer: deque[PredictionRecord] = deque(maxlen=buffer_size)
        self._source_adjustments: dict[str, float] = {}

    def record(
        self,
        condition_id: str,
        source: str,
        predicted_prob: float,
        market_price: float,
        actual_outcome: Optional[float] = None,
    ) -> None:
        """Record a prediction for later calibration assessment."""
        self._buffer.append(PredictionRecord(
            condition_id=condition_id,
            source=source,
            predicted_prob=predicted_prob,
            market_price=market_price,
            actual_outcome=actual_outcome,
            timestamp=datetime.now(timezone.utc),
        ))

    def update_outcome(self, condition_id: str, outcome: float) -> int:
        """Update actual outcome for a resolved market. Returns count updated."""
        count = 0
        for rec in self._buffer:
            if rec.condition_id == condition_id and rec.actual_outcome is None:
                rec.actual_outcome = outcome
                count += 1
        return count

    def brier_score(self, source: Optional[str] = None) -> Optional[float]:
        """Compute Brier score for resolved predictions. Lower is better."""
        resolved = [
            r for r in self._buffer
            if r.actual_outcome is not None
            and (source is None or r.source == source)
        ]
        if not resolved:
            return None
        return sum(
            (r.predicted_prob - r.actual_outcome) ** 2 for r in resolved
        ) / len(resolved)

    def brier_score_vs_market(self) -> Optional[tuple[float, float]]:
        """Compare Brier score: model vs market. Returns (model_brier, market_brier)."""
        resolved = [
            r for r in self._buffer
            if r.actual_outcome is not None and r.source == "bayesian"
        ]
        if not resolved:
            return None

        model_bs = sum(
            (r.predicted_prob - r.actual_outcome) ** 2 for r in resolved
        ) / len(resolved)
        market_bs = sum(
            (r.market_price - r.actual_outcome) ** 2 for r in resolved
        ) / len(resolved)

        return (model_bs, market_bs)

    def calibration_curve(
        self, source: str = "bayesian", n_bins: int = 10
    ) -> dict[str, list]:
        """Binned calibration curve: predicted vs actual by decile."""
        resolved = [
            r for r in self._buffer
            if r.actual_outcome is not None and r.source == source
        ]
        if not resolved:
            return {"bin_centers": [], "mean_predicted": [], "mean_actual": [], "counts": []}

        bins: dict[int, list[tuple[float, float]]] = {
            i: [] for i in range(n_bins)
        }
        for r in resolved:
            bin_idx = min(int(r.predicted_prob * n_bins), n_bins - 1)
            bins[bin_idx].append((r.predicted_prob, r.actual_outcome))

        centers, predicted, actual, counts = [], [], [], []
        for i in range(n_bins):
            if bins[i]:
                centers.append((i + 0.5) / n_bins)
                predicted.append(sum(p for p, _ in bins[i]) / len(bins[i]))
                actual.append(sum(a for _, a in bins[i]) / len(bins[i]))
                counts.append(len(bins[i]))

        return {
            "bin_centers": centers,
            "mean_predicted": predicted,
            "mean_actual": actual,
            "counts": counts,
        }

    def reliability_adjustment(self, source: str) -> float:
        """Compute reliability adjustment for a source.

        Returns a multiplicative factor (0.5 - 1.5):
          < 1.0: source is overconfident → shrink its Bayes factors
          > 1.0: source is underconfident → amplify its Bayes factors
          = 1.0: well calibrated
        """
        if source in self._source_adjustments:
            return self._source_adjustments[source]
        return 1.0

    def recompute_adjustments(self) -> dict[str, float]:
        """Recompute reliability adjustments for all sources."""
        for source in {"bayesian", "gnn", "composite"}:
            bs = self.brier_score(source)
            if bs is None:
                continue
            # Compare to market Brier score
            market_bs = self.brier_score("market")
            if market_bs is None or market_bs == 0:
                self._source_adjustments[source] = 1.0
                continue

            # Ratio: < 1 means source is better than market
            ratio = bs / market_bs
            # Map to adjustment: better → amplify, worse → shrink
            adj = max(0.5, min(1.5, 1.0 / ratio))
            self._source_adjustments[source] = adj

        return dict(self._source_adjustments)

    async def write_to_clickhouse(self, writer) -> None:
        """Persist calibration metrics to calibration_history table."""
        now = datetime.now(timezone.utc)
        rows = []

        for source in {"bayesian", "gnn", "composite", "market"}:
            bs = self.brier_score(source)
            resolved = [
                r for r in self._buffer
                if r.actual_outcome is not None
                and (r.source == source or (source == "market"))
            ]
            n_total = sum(1 for r in self._buffer if r.source == source or source == "market")
            n_resolved = len(resolved)

            if source == "market":
                # Market Brier score uses market_price as prediction
                if resolved:
                    bs = sum(
                        (r.market_price - r.actual_outcome) ** 2 for r in resolved
                    ) / len(resolved)

            cal_curve = self.calibration_curve(source)
            adj = self._source_adjustments.get(source, 1.0)

            rows.append([
                source,
                bs or 0.0,
                n_total,
                n_resolved,
                json.dumps(cal_curve),
                adj,
                now,
            ])

        if rows:
            try:
                await writer.write("calibration_history", rows)
                await writer.flush_all()
                logger.info("Wrote calibration metrics for %d sources", len(rows))
            except Exception as e:
                logger.error("Failed to write calibration: %s", e)
