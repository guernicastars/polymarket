"""Market signal generation — converts model probabilities to trading signals."""

from __future__ import annotations

import json
import math
import pathlib
from typing import TYPE_CHECKING, Optional

from ..core.types import ControlStatus, MarketSignal

if TYPE_CHECKING:
    from ..core.graph import DonbasGraph
    from .vulnerability import VulnerabilityAnalyzer
    from .supply_chain import SupplyChainAnalyzer
    from .cascade import CascadeSimulator


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class MarketSignalGenerator:
    """Generate buy/sell signals by comparing model P(fall) to Polymarket prices."""

    # Minimum edge (model - market) to generate a signal
    MIN_EDGE = 0.05

    def __init__(
        self,
        graph: "DonbasGraph",
        vulnerability: "VulnerabilityAnalyzer",
        supply: "SupplyChainAnalyzer",
        cascade: "CascadeSimulator",
    ) -> None:
        self.dg = graph
        self.vuln = vulnerability
        self.supply = supply
        self.cascade = cascade
        self.market_prices = self._load_market_prices()

    def _load_market_prices(self) -> dict[str, float]:
        """Load mock market prices (replace with live ClickHouse query)."""
        path = DATA_DIR / "polymarket_mapping.json"
        if not path.exists():
            return {}
        with open(path) as f:
            mapping = json.load(f)
        # Return mock prices for now — will be replaced by ClickHouse bridge
        mock_prices = {
            "pokrovsk": 0.35,
            "chasiv_yar": 0.55,
            "toretsk": 0.50,
            "kupiansk": 0.20,
            "zaporizhzhia": 0.03,
            "orikhiv": 0.15,
            "hryshyne": 0.75,
        }
        return mock_prices

    def model_probability(self, settlement_id: str) -> float:
        """Compute P(fall) from vulnerability, supply, and cascade analysis.

        Blends:
        - 40% vulnerability score (direct assault risk)
        - 30% supply risk (can it be sustained?)
        - 20% cascade effect (does losing neighbors make it worse?)
        - 10% base rate from control status
        """
        # Vulnerability component
        vuln_scores = self.vuln.score_all()
        vuln = vuln_scores.get(settlement_id)
        vuln_p = vuln.composite if vuln else 0.0

        # Supply component
        supply_info = self.supply.supply_risk_score(settlement_id)
        supply_p = supply_info["supply_risk"]

        # Cascade component: check if neighboring falls would isolate this node
        settlement = self.dg.settlements.get(settlement_id)
        cascade_p = 0.0
        if settlement:
            neighbors = list(self.dg.G.neighbors(settlement_id))
            ua_neighbors = [
                n for n in neighbors
                if self.dg.get_effective_control(n) in (ControlStatus.UA, ControlStatus.CONTESTED)
            ]
            for neighbor in ua_neighbors[:3]:  # check top 3 neighbors
                cascade_result = self.cascade.simulate_fall(neighbor)
                if settlement_id in cascade_result.isolated_nodes:
                    cascade_p = max(cascade_p, 0.5)

        # Base rate
        control = self.dg.get_effective_control(settlement_id)
        base_rate = {
            ControlStatus.UA: 0.1,
            ControlStatus.CONTESTED: 0.5,
            ControlStatus.RU: 0.95,
        }.get(control, 0.3)

        # Blend
        p = (
            0.40 * vuln_p
            + 0.30 * supply_p
            + 0.20 * cascade_p
            + 0.10 * base_rate
        )
        return min(max(p, 0.01), 0.99)

    def kelly_criterion(self, p_model: float, p_market: float) -> float:
        """Kelly criterion for position sizing.

        f* = (p * b - q) / b
        where b = odds, p = model prob, q = 1-p
        """
        if p_market <= 0 or p_market >= 1:
            return 0.0

        # Betting YES (we think it's underpriced)
        if p_model > p_market:
            b = (1.0 / p_market) - 1.0  # implied odds
            q = 1.0 - p_model
            f = (p_model * b - q) / b if b > 0 else 0.0
            return max(f, 0.0)

        # Betting NO (we think it's overpriced)
        p_no_model = 1.0 - p_model
        p_no_market = 1.0 - p_market
        if p_no_market <= 0:
            return 0.0
        b = (1.0 / p_no_market) - 1.0
        q = 1.0 - p_no_model
        f = (p_no_model * b - q) / b if b > 0 else 0.0
        return max(f, 0.0)

    def generate_signal(self, settlement_id: str) -> Optional[MarketSignal]:
        """Generate a trading signal for a single settlement."""
        market_price = self.market_prices.get(settlement_id)
        if market_price is None:
            return None

        p_model = self.model_probability(settlement_id)
        edge = p_model - market_price

        signal = MarketSignal(
            settlement_id=settlement_id,
            market_slug=self.dg.settlements[settlement_id].polymarket_slug or "",
            model_probability=round(p_model, 4),
            market_probability=market_price,
            edge=round(edge, 4),
        )

        if abs(edge) < self.MIN_EDGE:
            signal.direction = "HOLD"
            signal.kelly_fraction = 0.0
            signal.confidence = abs(edge) / self.MIN_EDGE
        elif edge > 0:
            signal.direction = "BUY"  # market underprices risk
            signal.kelly_fraction = round(self.kelly_criterion(p_model, market_price), 4)
            signal.confidence = min(abs(edge) / 0.3, 1.0)
        else:
            signal.direction = "SELL"  # market overprices risk
            signal.kelly_fraction = round(self.kelly_criterion(p_model, market_price), 4)
            signal.confidence = min(abs(edge) / 0.3, 1.0)

        return signal

    def generate_all_signals(self) -> list[MarketSignal]:
        """Generate signals for all Polymarket target settlements."""
        targets = self.dg.get_polymarket_targets()
        signals = []
        for t in targets:
            if self.dg.get_effective_control(t.id) == ControlStatus.RU:
                continue
            sig = self.generate_signal(t.id)
            if sig:
                signals.append(sig)
        return sorted(signals, key=lambda s: abs(s.edge), reverse=True)
