"""Data classes and enums for the Donbas network model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ControlStatus(str, Enum):
    """Who controls a settlement."""
    UA = "UA"
    RU = "RU"
    CONTESTED = "CONTESTED"


class EdgeType(str, Enum):
    """Type of connection between settlements."""
    HIGHWAY = "HIGHWAY"
    RAIL = "RAIL"
    SECONDARY_ROAD = "SECONDARY_ROAD"
    SUPPLY_ROUTE = "SUPPLY_ROUTE"
    FRONTLINE = "FRONTLINE"


@dataclass
class Settlement:
    """A node in the Donbas multigraph."""
    id: str
    name: str
    lat: float
    lng: float
    control: ControlStatus
    population: int = 0
    is_polymarket_target: bool = False
    polymarket_slug: Optional[str] = None
    oblast: str = ""
    fortification_level: float = 0.5       # 0-1, higher = more fortified
    terrain_difficulty: float = 0.5         # 0-1, higher = harder to attack
    garrison_estimate: int = 0             # estimated troop count
    tags: list[str] = field(default_factory=list)


@dataclass
class Edge:
    """An edge in the Donbas multigraph."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0                    # travel difficulty / strategic cost
    capacity: float = 1.0                  # supply throughput (normalized)
    distance_km: float = 0.0
    is_active: bool = True                 # can still be used?
    is_frontline: bool = False
    description: str = ""


@dataclass
class DynamicState:
    """Overlay for real-time OSINT updates — one file to rule them all."""
    settlement_id: str
    control_override: Optional[ControlStatus] = None
    assault_intensity: float = 0.0         # 0-1, current attack pressure
    shelling_intensity: float = 0.0        # 0-1, artillery/drone activity
    supply_disruption: float = 0.0         # 0-1, how cut off
    frontline_distance_km: float = 50.0    # distance to nearest frontline
    last_updated: str = ""                 # ISO timestamp


@dataclass
class VulnerabilityScore:
    """Composite vulnerability assessment for a settlement."""
    settlement_id: str
    connectivity_score: float = 0.0       # from graph centrality
    supply_score: float = 0.0             # from supply chain analysis
    force_balance_score: float = 0.0      # attacker/defender ratio proxy
    terrain_score: float = 0.0            # terrain difficulty inverted
    fortification_score: float = 0.0      # fortification inverted
    assault_intensity_score: float = 0.0  # from dynamic state
    frontline_score: float = 0.0          # proximity to front
    composite: float = 0.0               # weighted sum

    # weights
    WEIGHTS: dict = field(default_factory=lambda: {
        "connectivity": 0.20,
        "supply": 0.20,
        "force_balance": 0.15,
        "terrain": 0.10,
        "fortification": 0.10,
        "assault_intensity": 0.15,
        "frontline": 0.10,
    })

    def compute(self) -> float:
        w = self.WEIGHTS
        self.composite = (
            w["connectivity"] * self.connectivity_score
            + w["supply"] * self.supply_score
            + w["force_balance"] * self.force_balance_score
            + w["terrain"] * (1.0 - self.terrain_score)
            + w["fortification"] * (1.0 - self.fortification_score)
            + w["assault_intensity"] * self.assault_intensity_score
            + w["frontline"] * max(0, 1.0 - self.frontline_score / 50.0)
        )
        return self.composite


@dataclass
class MarketSignal:
    """Trading signal for a Polymarket market."""
    settlement_id: str
    market_slug: str
    model_probability: float       # our model's p(fall)
    market_probability: float      # Polymarket's current price
    edge: float                    # model - market
    direction: str = ""            # BUY / SELL / HOLD
    kelly_fraction: float = 0.0    # Kelly criterion sizing
    confidence: float = 0.0        # 0-1
