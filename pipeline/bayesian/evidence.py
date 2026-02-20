"""Evidence adapters: convert raw signals to likelihood ratios (Bayes factors).

Each adapter takes a signal output and produces a likelihood ratio K:
  K = P(signal | event=YES) / P(signal | event=NO)

K > 1 favors YES, K < 1 favors NO, K = 1 is uninformative.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvidenceUpdate:
    """A single piece of evidence for the Bayesian combiner."""

    source: str                      # 'gnn', 'composite', 'granger', 'info_flow'
    likelihood_ratio: float          # Bayes factor K
    weight: float = 1.0              # Attenuation (0-1), K_eff = K^weight
    confidence: float = 0.0          # Source self-reported confidence
    timestamp: Optional[datetime] = None


class GNNEvidenceAdapter:
    """Convert GNN calibrated probability to likelihood ratio.

    K = (p_gnn / (1 - p_gnn)) * shrinkage_factor

    The shrinkage prevents the GNN from dominating the posterior.
    Cold-start predictions produce K=1 (uninformative).
    """

    def __init__(
        self,
        reliability: float = 0.6,
        calibration_shrinkage: float = 0.5,
    ) -> None:
        self.reliability = reliability
        self.calibration_shrinkage = calibration_shrinkage

    def to_evidence(
        self,
        calibrated_prob: float,
        model_uncertainty: float = 0.0,
        is_cold_start: bool = False,
    ) -> EvidenceUpdate:
        """Convert GNN prediction to evidence update."""
        if is_cold_start:
            return EvidenceUpdate(
                source="gnn",
                likelihood_ratio=1.0,
                weight=0.0,
                confidence=0.0,
            )

        # Clip to avoid infinity
        p = max(0.02, min(0.98, calibrated_prob))

        # Raw odds ratio
        odds = p / (1.0 - p)

        # Apply shrinkage: reduces the Bayes factor toward 1.0
        shrinkage = self.calibration_shrinkage * self.reliability
        k = odds ** shrinkage  # K closer to 1 than raw odds

        # Reduce weight further if model is uncertain
        weight = self.reliability * max(0.1, 1.0 - model_uncertainty)

        return EvidenceUpdate(
            source="gnn",
            likelihood_ratio=k,
            weight=weight,
            confidence=1.0 - model_uncertainty,
            timestamp=datetime.now(timezone.utc),
        )


class CompositeSignalAdapter:
    """Convert composite signal score (-100 to +100) to likelihood ratio.

    K = exp(score * reliability / temperature)

    Temperature controls sensitivity. Higher temperature = less responsive.
    """

    def __init__(
        self,
        reliability: float = 0.4,
        temperature: float = 50.0,
    ) -> None:
        self.reliability = reliability
        self.temperature = temperature

    def to_evidence(
        self,
        score: float,
        signal_confidence: float = 0.0,
    ) -> EvidenceUpdate:
        """Convert composite signal to evidence update."""
        if abs(score) < 5.0:
            # Very weak signal — treat as uninformative
            return EvidenceUpdate(
                source="composite",
                likelihood_ratio=1.0,
                weight=0.0,
                confidence=0.0,
            )

        exponent = score * self.reliability / self.temperature
        # Cap to prevent extreme K values
        exponent = max(-2.0, min(2.0, exponent))
        k = math.exp(exponent)

        # Higher confidence in stronger signals
        weight = self.reliability * min(1.0, abs(score) / 50.0)

        return EvidenceUpdate(
            source="composite",
            likelihood_ratio=k,
            weight=weight,
            confidence=signal_confidence,
            timestamp=datetime.now(timezone.utc),
        )


class GrangerEvidenceAdapter:
    """Convert Granger causality leading-market signal to likelihood ratio.

    If a market that Granger-causes this one has recently moved,
    that provides directional evidence about this market's future.
    """

    def __init__(self, reliability: float = 0.3) -> None:
        self.reliability = reliability

    def to_evidence(
        self,
        leading_market_move: float,
        granger_pvalue: float,
        lag_steps: int = 1,
    ) -> EvidenceUpdate:
        """Convert Granger signal to evidence update."""
        if granger_pvalue > 0.05:
            return EvidenceUpdate(
                source="granger",
                likelihood_ratio=1.0,
                weight=0.0,
                confidence=0.0,
            )

        significance = 1.0 - granger_pvalue
        exponent = leading_market_move * significance * self.reliability
        exponent = max(-1.5, min(1.5, exponent))
        k = math.exp(exponent)

        # Decay weight with lag
        weight = self.reliability * (0.8 ** lag_steps) * significance

        return EvidenceUpdate(
            source="granger",
            likelihood_ratio=k,
            weight=weight,
            confidence=significance,
            timestamp=datetime.now(timezone.utc),
        )


class InformationFlowAdapter:
    """Convert transfer entropy information flow to likelihood ratio."""

    def __init__(self, reliability: float = 0.25) -> None:
        self.reliability = reliability

    def to_evidence(
        self,
        source_market_move: float,
        transfer_entropy: float,
        min_te: float = 0.01,
    ) -> EvidenceUpdate:
        """Convert information flow signal to evidence update."""
        if transfer_entropy < min_te:
            return EvidenceUpdate(
                source="info_flow",
                likelihood_ratio=1.0,
                weight=0.0,
                confidence=0.0,
            )

        # Scale move by transfer entropy strength
        exponent = source_market_move * transfer_entropy * self.reliability * 10.0
        exponent = max(-1.0, min(1.0, exponent))
        k = math.exp(exponent)

        weight = self.reliability * min(1.0, transfer_entropy * 10.0)

        return EvidenceUpdate(
            source="info_flow",
            likelihood_ratio=k,
            weight=weight,
            confidence=min(1.0, transfer_entropy * 10.0),
            timestamp=datetime.now(timezone.utc),
        )
