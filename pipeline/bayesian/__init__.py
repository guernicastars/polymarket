"""Bayesian prediction combiner — Layer 2 of the two-layer architecture.

Uses market price as an informative Beta prior, then updates with
evidence from GNN-TCN, composite signals, and causal analysis.
"""

from .combiner import BayesianCombiner, BayesianPrediction, BetaPosterior
from .evidence import (
    GNNEvidenceAdapter,
    CompositeSignalAdapter,
    EvidenceUpdate,
)
from .state import PosteriorStateStore
from .calibration import CalibrationTracker

__all__ = [
    "BayesianCombiner",
    "BayesianPrediction",
    "BetaPosterior",
    "GNNEvidenceAdapter",
    "CompositeSignalAdapter",
    "EvidenceUpdate",
    "PosteriorStateStore",
    "CalibrationTracker",
]
