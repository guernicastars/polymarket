"""Signal ensemble — quality-weighted combination of weak predictive signals.

Inspired by OpenForage's core insight: millions of individually weak signals,
when ensembled with quality weighting and orthogonality constraints, produce
predictions far stronger than any single signal. The "blessing of dimensionality"
in practice — high-dimensional signal spaces make signal independence the norm,
allowing massive parallel ensembles without catastrophic correlation collapse.

This module implements the ensemble layer that sits between the raw signal
sources (composite signals, GNN predictions, causal analysis outputs) and
the final trading/prediction output. Each signal is evaluated on:

1. IN-SAMPLE quality: how well does the signal predict within the training period?
   (OpenForage's "found" criterion — the signal looks good locally)

2. OUT-OF-SAMPLE quality: how well does the signal predict on held-out data?
   (OpenForage's "useful" criterion — the signal generalizes beyond local fit)

3. UNIQUENESS: how orthogonal is this signal to existing ensemble members?
   (OpenForage's correlation constraint — redundant signals dilute rather
   than strengthen the ensemble)

4. MARGINAL IMPROVEMENT: does adding this signal actually improve the ensemble?
   (The ensemble only accepts signals that increase out-of-sample performance)

The evaluation pipeline:
    Signal submitted → in-sample test → out-of-sample test → uniqueness check
    → marginal improvement test → accepted/rejected → ensemble weight assigned

References:
    OpenForage: "The Design of a Pure Alpha Yield" (2025)
    Breiman (1996): Bagging predictors (ensemble theory foundations)
    DeMiguel et al. (2009): 1/N portfolio vs optimal — why simple
        weighting often beats optimization (guards against overfitting)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalStatus(str, Enum):
    """Lifecycle status of a signal in the ensemble evaluation pipeline.

    Maps to OpenForage's found → useful distinction:
    - SUBMITTED: signal is proposed, awaiting in-sample evaluation
    - IN_SAMPLE_PASSED: signal shows in-sample predictive power (found)
    - VALIDATED: signal passes out-of-sample testing (useful)
    - REJECTED: signal failed evaluation at some stage
    - ACTIVE: signal is in the live ensemble
    - RETIRED: signal was active but degraded and was removed
    """

    SUBMITTED = "submitted"
    IN_SAMPLE_PASSED = "in_sample_passed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ACTIVE = "active"
    RETIRED = "retired"


class SignalSource(str, Enum):
    """Source category for a signal.

    Tracks which pipeline component generated the signal, enabling
    analysis of which sources produce the most useful predictions.
    """

    COMPOSITE = "composite"          # From signal_compositor (8-factor)
    GNN_TCN = "gnn_tcn"             # From GNN-TCN model
    GRANGER = "granger"             # From Granger causality analysis
    INFORMATION_FLOW = "info_flow"  # From transfer entropy analysis
    CROSS_MARKET = "cross_market"   # From PC algorithm causal discovery
    MANIPULATION = "manipulation"   # From manipulation detection (contrarian)
    EVENT_IMPACT = "event_impact"   # From causal impact analysis
    CUSTOM = "custom"               # User-defined signal


# ---------------------------------------------------------------------------
# Signal record
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """A single predictive signal in the ensemble.

    Each signal is a time series of predictions (directional scores from
    -100 to +100) for a specific market or set of markets. The signal
    carries quality metrics from the evaluation pipeline and, if accepted,
    an ensemble weight.

    Attributes:
        id: Unique signal identifier.
        name: Human-readable signal name.
        source: Which pipeline component generated this signal.
        market_ids: Which markets this signal applies to.
        status: Current evaluation status.
        in_sample_score: Predictive quality on training data (0-1).
        out_of_sample_score: Predictive quality on held-out data (0-1).
        uniqueness_score: Orthogonality to existing ensemble members (0-1).
        marginal_improvement: How much this signal improves ensemble performance.
        ensemble_weight: Weight assigned in the active ensemble (0-1).
        submitted_at: When the signal was first submitted.
        validated_at: When the signal passed out-of-sample testing.
        metadata: Additional signal-specific metadata.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique signal identifier.",
    )
    name: str = Field(
        ...,
        description="Human-readable signal name.",
    )
    source: SignalSource = Field(
        ...,
        description="Which pipeline component generated this signal.",
    )
    market_ids: list[str] = Field(
        default_factory=list,
        description="Condition IDs of markets this signal applies to.",
    )
    status: SignalStatus = Field(
        default=SignalStatus.SUBMITTED,
        description="Current evaluation status.",
    )

    # Quality metrics from evaluation pipeline
    in_sample_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Predictive quality on training data (0 = no power, 1 = perfect).",
    )
    out_of_sample_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Predictive quality on held-out data.",
    )
    uniqueness_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Orthogonality to existing ensemble members (0 = redundant, 1 = fully independent).",
    )
    marginal_improvement: Optional[float] = Field(
        default=None,
        description="Change in ensemble performance when this signal is added.",
    )

    # Ensemble participation
    ensemble_weight: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weight in the active ensemble.",
    )

    # Timestamps
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    validated_at: Optional[datetime] = None

    # Flexible metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Signal-specific metadata (parameters, thresholds, etc.).",
    )


# ---------------------------------------------------------------------------
# Ensemble evaluation configuration
# ---------------------------------------------------------------------------


class EnsembleConfig(BaseModel):
    """Configuration for the signal ensemble evaluation pipeline.

    These thresholds determine which signals are admitted to the ensemble
    and how they are weighted. The defaults are conservative — they require
    signals to demonstrate genuine out-of-sample predictive power before
    they influence the ensemble output.

    Inspired by OpenForage's era-specific evaluation parameters.
    """

    # Minimum scores for each evaluation stage
    min_in_sample_score: float = Field(
        default=0.52, ge=0.5, le=1.0,
        description="Minimum in-sample score to advance (barely above chance).",
    )
    min_out_of_sample_score: float = Field(
        default=0.51, ge=0.5, le=1.0,
        description="Minimum out-of-sample score for validation.",
    )
    min_uniqueness_score: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum orthogonality to existing ensemble.",
    )
    min_marginal_improvement: float = Field(
        default=0.0,
        description="Minimum improvement to ensemble performance (0 = must not hurt).",
    )

    # Ensemble construction
    max_correlation_between_signals: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Maximum pairwise correlation between active signals.",
    )
    max_ensemble_size: int = Field(
        default=50, ge=1,
        description="Maximum number of active signals in the ensemble.",
    )
    weight_method: str = Field(
        default="quality_weighted",
        description="How to assign weights: 'equal', 'quality_weighted', or 'optimal'.",
    )

    # Rebalancing
    rebalance_interval_hours: float = Field(
        default=24.0, ge=1.0,
        description="Hours between ensemble rebalancing.",
    )
    retirement_threshold: float = Field(
        default=0.48, ge=0.0, le=1.0,
        description="Out-of-sample score below which an active signal is retired.",
    )


# ---------------------------------------------------------------------------
# Signal Ensemble
# ---------------------------------------------------------------------------


class SignalEnsemble:
    """Quality-weighted signal ensemble for prediction market analysis.

    Manages the lifecycle of predictive signals: submission, evaluation,
    admission to the ensemble, weight assignment, and retirement. Produces
    a combined prediction that is stronger than any individual signal
    through the blessing of dimensionality.

    The ensemble implements OpenForage's core architecture pattern:
    - Many weak signals (individual predictions barely above chance)
    - Quality-gated admission (in-sample → out-of-sample → uniqueness)
    - Orthogonality weighting (unique signals get more weight)
    - Continuous monitoring (degraded signals are retired)

    Attributes:
        config: Ensemble evaluation configuration.
        signals: All signals, keyed by ID.
        active_signal_ids: IDs of signals currently in the ensemble.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()
        self.signals: dict[str, Signal] = {}
        self.active_signal_ids: list[str] = []
        self._prediction_cache: dict[str, np.ndarray] = {}

    @property
    def active_signals(self) -> list[Signal]:
        """Return the currently active ensemble members."""
        return [
            self.signals[sid] for sid in self.active_signal_ids
            if sid in self.signals
        ]

    @property
    def ensemble_size(self) -> int:
        """Number of active signals in the ensemble."""
        return len(self.active_signal_ids)

    # ------------------------------------------------------------------
    # Signal submission and evaluation
    # ------------------------------------------------------------------

    def submit_signal(self, signal: Signal) -> Signal:
        """Submit a new signal for evaluation.

        The signal enters the pipeline in SUBMITTED status. Call
        evaluate_signal() to run it through the quality gates.

        Args:
            signal: The signal to submit.

        Returns:
            The stored signal.
        """
        self.signals[signal.id] = signal
        logger.info(
            "Signal submitted: '%s' (source=%s, markets=%d)",
            signal.name,
            signal.source.value,
            len(signal.market_ids),
        )
        return signal

    def evaluate_signal(
        self,
        signal_id: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        train_mask: np.ndarray,
    ) -> Signal:
        """Evaluate a submitted signal through the full quality pipeline.

        Runs the signal through four gates:
        1. In-sample: predictive power on training data
        2. Out-of-sample: predictive power on held-out data
        3. Uniqueness: orthogonality to existing ensemble
        4. Marginal improvement: does the ensemble get better?

        Args:
            signal_id: ID of the signal to evaluate.
            predictions: Signal predictions array (same length as actuals).
            actuals: Actual outcomes (1 = event occurred, 0 = not).
            train_mask: Boolean mask (True = in-sample, False = out-of-sample).

        Returns:
            The evaluated signal with updated scores and status.

        Raises:
            KeyError: If signal_id is not found.
        """
        signal = self._get(signal_id)

        # Gate 1: In-sample evaluation
        in_sample_score = self._evaluate_accuracy(
            predictions[train_mask], actuals[train_mask]
        )
        signal.in_sample_score = in_sample_score

        if in_sample_score < self.config.min_in_sample_score:
            signal.status = SignalStatus.REJECTED
            logger.info(
                "Signal '%s' REJECTED: in-sample score %.4f < threshold %.4f",
                signal.name, in_sample_score, self.config.min_in_sample_score,
            )
            return signal

        signal.status = SignalStatus.IN_SAMPLE_PASSED

        # Gate 2: Out-of-sample evaluation (the real test)
        oos_mask = ~train_mask
        oos_score = self._evaluate_accuracy(
            predictions[oos_mask], actuals[oos_mask]
        )
        signal.out_of_sample_score = oos_score

        if oos_score < self.config.min_out_of_sample_score:
            signal.status = SignalStatus.REJECTED
            logger.info(
                "Signal '%s' REJECTED: out-of-sample score %.4f < threshold %.4f",
                signal.name, oos_score, self.config.min_out_of_sample_score,
            )
            return signal

        # Gate 3: Uniqueness check
        uniqueness = self._compute_uniqueness(predictions)
        signal.uniqueness_score = uniqueness

        if uniqueness < self.config.min_uniqueness_score:
            signal.status = SignalStatus.REJECTED
            logger.info(
                "Signal '%s' REJECTED: uniqueness %.4f < threshold %.4f "
                "(too correlated with existing ensemble)",
                signal.name, uniqueness, self.config.min_uniqueness_score,
            )
            return signal

        # Gate 4: Marginal improvement
        marginal = self._compute_marginal_improvement(
            predictions, actuals, oos_mask
        )
        signal.marginal_improvement = marginal

        if marginal < self.config.min_marginal_improvement:
            signal.status = SignalStatus.REJECTED
            logger.info(
                "Signal '%s' REJECTED: marginal improvement %.6f < threshold %.6f",
                signal.name, marginal, self.config.min_marginal_improvement,
            )
            return signal

        # All gates passed — signal is validated
        signal.status = SignalStatus.VALIDATED
        signal.validated_at = datetime.now(timezone.utc)
        self._prediction_cache[signal_id] = predictions

        logger.info(
            "Signal '%s' VALIDATED: in_sample=%.4f, oos=%.4f, "
            "uniqueness=%.4f, marginal=%.6f",
            signal.name,
            in_sample_score,
            oos_score,
            uniqueness,
            marginal,
        )
        return signal

    def activate_signal(self, signal_id: str) -> Signal:
        """Activate a validated signal into the live ensemble.

        Assigns an ensemble weight and adds the signal to the active set.

        Args:
            signal_id: ID of the validated signal.

        Returns:
            The activated signal with its ensemble weight.

        Raises:
            ValueError: If signal is not VALIDATED status.
        """
        signal = self._get(signal_id)
        if signal.status != SignalStatus.VALIDATED:
            raise ValueError(
                f"Signal '{signal_id}' is {signal.status.value}, expected VALIDATED."
            )

        if self.ensemble_size >= self.config.max_ensemble_size:
            # Replace the weakest active signal if this one is better
            weakest = min(
                self.active_signals,
                key=lambda s: (s.out_of_sample_score or 0) * (s.uniqueness_score or 0),
            )
            if (
                (signal.out_of_sample_score or 0) * (signal.uniqueness_score or 0)
                > (weakest.out_of_sample_score or 0) * (weakest.uniqueness_score or 0)
            ):
                self._retire_signal(weakest.id)
            else:
                logger.info(
                    "Signal '%s' validated but ensemble full and not better than weakest.",
                    signal.name,
                )
                return signal

        signal.status = SignalStatus.ACTIVE
        self.active_signal_ids.append(signal_id)
        self._recompute_weights()

        logger.info(
            "Signal '%s' ACTIVATED with weight %.4f (ensemble size: %d)",
            signal.name,
            signal.ensemble_weight,
            self.ensemble_size,
        )
        return signal

    def retire_signal(self, signal_id: str) -> Signal:
        """Retire an active signal from the ensemble.

        Args:
            signal_id: ID of the signal to retire.

        Returns:
            The retired signal.
        """
        return self._retire_signal(signal_id)

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        signal_predictions: dict[str, float],
    ) -> float:
        """Produce an ensemble prediction from active signal outputs.

        Combines individual signal predictions using quality-weighted
        averaging. This is the ensemble's primary output.

        Args:
            signal_predictions: Mapping of signal_id -> prediction score
                for the current timestep. Only active signals are used.

        Returns:
            Ensemble prediction score (-100 to +100).
        """
        if not self.active_signal_ids:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for sid in self.active_signal_ids:
            if sid not in signal_predictions:
                continue
            signal = self.signals.get(sid)
            if signal is None:
                continue

            pred = signal_predictions[sid]
            weight = signal.ensemble_weight

            weighted_sum += pred * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return max(-100.0, min(100.0, weighted_sum / weight_sum))

    # ------------------------------------------------------------------
    # Ensemble monitoring and rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        recent_predictions: dict[str, np.ndarray],
        recent_actuals: np.ndarray,
    ) -> dict[str, float]:
        """Rebalance the ensemble: retire degraded signals, recompute weights.

        Should be called periodically (per rebalance_interval_hours config).

        Args:
            recent_predictions: signal_id -> recent prediction arrays.
            recent_actuals: Recent actual outcomes.

        Returns:
            Updated weights mapping signal_id -> new weight.
        """
        retired = []

        for sid in list(self.active_signal_ids):
            if sid not in recent_predictions:
                continue

            preds = recent_predictions[sid]
            score = self._evaluate_accuracy(preds, recent_actuals)

            signal = self.signals.get(sid)
            if signal is None:
                continue

            # Update running out-of-sample score (exponential moving average)
            if signal.out_of_sample_score is not None:
                alpha = 0.3  # Weight for recent performance
                signal.out_of_sample_score = (
                    alpha * score + (1 - alpha) * signal.out_of_sample_score
                )
            else:
                signal.out_of_sample_score = score

            # Retire if degraded below threshold
            if signal.out_of_sample_score < self.config.retirement_threshold:
                self._retire_signal(sid)
                retired.append(signal.name)

        if retired:
            logger.info(
                "Rebalance retired %d signals: %s",
                len(retired), ", ".join(retired),
            )

        self._recompute_weights()

        return {
            sid: self.signals[sid].ensemble_weight
            for sid in self.active_signal_ids
            if sid in self.signals
        }

    def metrics(self) -> dict[str, Any]:
        """Return ensemble performance metrics.

        Returns:
            Dictionary with ensemble size, average quality, source
            distribution, and weight statistics.
        """
        active = self.active_signals
        if not active:
            return {
                "ensemble_size": 0,
                "avg_oos_score": None,
                "avg_uniqueness": None,
                "source_distribution": {},
                "total_signals_submitted": len(self.signals),
            }

        oos_scores = [s.out_of_sample_score for s in active if s.out_of_sample_score]
        uniq_scores = [s.uniqueness_score for s in active if s.uniqueness_score]

        source_dist: dict[str, int] = {}
        for s in active:
            source_dist[s.source.value] = source_dist.get(s.source.value, 0) + 1

        return {
            "ensemble_size": len(active),
            "avg_oos_score": np.mean(oos_scores) if oos_scores else None,
            "avg_uniqueness": np.mean(uniq_scores) if uniq_scores else None,
            "source_distribution": source_dist,
            "total_signals_submitted": len(self.signals),
            "total_rejected": sum(
                1 for s in self.signals.values()
                if s.status == SignalStatus.REJECTED
            ),
            "total_retired": sum(
                1 for s in self.signals.values()
                if s.status == SignalStatus.RETIRED
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, signal_id: str) -> Signal:
        """Retrieve a signal by ID."""
        if signal_id not in self.signals:
            raise KeyError(f"Signal '{signal_id}' not found.")
        return self.signals[signal_id]

    def _retire_signal(self, signal_id: str) -> Signal:
        """Internal retirement of a signal."""
        signal = self._get(signal_id)
        signal.status = SignalStatus.RETIRED
        signal.ensemble_weight = 0.0
        if signal_id in self.active_signal_ids:
            self.active_signal_ids.remove(signal_id)
        self._prediction_cache.pop(signal_id, None)
        logger.info("Signal '%s' RETIRED.", signal.name)
        return signal

    def _evaluate_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Evaluate directional accuracy of predictions.

        Computes the fraction of predictions that correctly predict the
        direction of the actual outcome. Predictions > 0 predict the event
        occurring; actuals of 1 mean it occurred.

        Args:
            predictions: Signal predictions (negative = no, positive = yes).
            actuals: Binary outcomes (0 or 1).

        Returns:
            Accuracy score from 0.0 to 1.0 (0.5 = random).
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.5

        # Convert signal scores to binary predictions
        pred_binary = (predictions > 0).astype(float)
        correct = (pred_binary == actuals).mean()
        return float(correct)

    def _compute_uniqueness(self, predictions: np.ndarray) -> float:
        """Compute how orthogonal a signal is to the existing ensemble.

        Uses average absolute correlation with active signals. A score
        of 1.0 means completely uncorrelated (maximally unique). A score
        of 0.0 means perfectly correlated with existing signals.

        Args:
            predictions: The candidate signal's prediction array.

        Returns:
            Uniqueness score from 0.0 to 1.0.
        """
        if not self.active_signal_ids or not self._prediction_cache:
            return 1.0  # No existing signals → maximally unique

        correlations = []
        for sid in self.active_signal_ids:
            if sid not in self._prediction_cache:
                continue
            existing = self._prediction_cache[sid]
            # Align lengths
            min_len = min(len(predictions), len(existing))
            if min_len < 10:
                continue
            corr = np.corrcoef(predictions[:min_len], existing[:min_len])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        if not correlations:
            return 1.0

        avg_correlation = np.mean(correlations)
        return float(1.0 - avg_correlation)

    def _compute_marginal_improvement(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        oos_mask: np.ndarray,
    ) -> float:
        """Compute how much adding this signal improves ensemble OOS performance.

        Compares the ensemble's out-of-sample accuracy with and without
        the candidate signal.

        Args:
            predictions: Candidate signal predictions.
            actuals: Actual outcomes.
            oos_mask: Boolean mask for out-of-sample period.

        Returns:
            Marginal improvement (positive = better, negative = worse).
        """
        if not self.active_signal_ids:
            return 0.01  # First signal always provides marginal improvement

        oos_actuals = actuals[oos_mask]
        oos_predictions = predictions[oos_mask]

        # Current ensemble prediction (equal weight for simplicity)
        ensemble_preds = np.zeros(oos_mask.sum())
        count = 0
        for sid in self.active_signal_ids:
            if sid not in self._prediction_cache:
                continue
            cached = self._prediction_cache[sid]
            min_len = min(len(cached[oos_mask]), len(ensemble_preds))
            ensemble_preds[:min_len] += cached[oos_mask][:min_len]
            count += 1

        if count > 0:
            ensemble_preds /= count

        current_accuracy = self._evaluate_accuracy(ensemble_preds, oos_actuals)

        # Ensemble with candidate added
        new_preds = (ensemble_preds * count + oos_predictions[:len(ensemble_preds)]) / (count + 1)
        new_accuracy = self._evaluate_accuracy(new_preds, oos_actuals)

        return float(new_accuracy - current_accuracy)

    def _recompute_weights(self) -> None:
        """Recompute ensemble weights for all active signals.

        Weight method options:
        - 'equal': 1/N weighting (robust baseline per DeMiguel et al.)
        - 'quality_weighted': proportional to OOS score * uniqueness
        - 'optimal': optimization-based (future implementation)
        """
        if not self.active_signal_ids:
            return

        if self.config.weight_method == "equal":
            w = 1.0 / len(self.active_signal_ids)
            for sid in self.active_signal_ids:
                self.signals[sid].ensemble_weight = w

        elif self.config.weight_method == "quality_weighted":
            # Weight proportional to (OOS score - 0.5) * uniqueness
            # This gives more weight to signals that are both accurate
            # and independent — the OpenForage insight
            raw_weights = {}
            for sid in self.active_signal_ids:
                signal = self.signals[sid]
                oos = signal.out_of_sample_score or 0.5
                uniq = signal.uniqueness_score or 0.5
                # Edge above chance * uniqueness
                raw_weights[sid] = max(0.001, (oos - 0.5) * uniq)

            total = sum(raw_weights.values())
            for sid, raw in raw_weights.items():
                self.signals[sid].ensemble_weight = raw / total

        else:
            # Default to equal weighting
            w = 1.0 / len(self.active_signal_ids)
            for sid in self.active_signal_ids:
                self.signals[sid].ensemble_weight = w
