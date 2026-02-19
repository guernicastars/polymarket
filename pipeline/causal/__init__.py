"""Causal inference layer for the Polymarket pipeline.

This package provides tools for causal analysis of prediction market data,
going beyond correlational observation to identify what CAUSES market
movements. Implements cross-market causal discovery, event impact analysis,
information flow measurement, manipulation detection, counterfactual
reasoning, and quality-weighted signal ensembles.

Modules:
    granger          -- Granger causality testing between market price series
    event_impact     -- Causal impact analysis of real-world events on markets
    information_flow -- Transfer entropy and information flow analysis
    cross_market     -- PC-algorithm causal discovery on market panels
    manipulation     -- Market manipulation detection (wash trading, spoofing)
    counterfactual   -- Synthetic control and "what if" counterfactual analysis
    signal_ensemble  -- Quality-weighted signal ensemble (OpenForage-inspired)
"""

from pipeline.causal.granger import GrangerCausalityAnalyzer
from pipeline.causal.event_impact import EventImpactAnalyzer
from pipeline.causal.information_flow import InformationFlowAnalyzer
from pipeline.causal.cross_market import CrossMarketAnalyzer
from pipeline.causal.manipulation import ManipulationDetector
from pipeline.causal.counterfactual import CounterfactualAnalyzer
from pipeline.causal.signal_ensemble import (
    EnsembleConfig,
    Signal,
    SignalEnsemble,
    SignalSource,
    SignalStatus,
)

__all__ = [
    "GrangerCausalityAnalyzer",
    "EventImpactAnalyzer",
    "InformationFlowAnalyzer",
    "CrossMarketAnalyzer",
    "ManipulationDetector",
    "CounterfactualAnalyzer",
    "SignalEnsemble",
    "Signal",
    "SignalSource",
    "SignalStatus",
    "EnsembleConfig",
]
