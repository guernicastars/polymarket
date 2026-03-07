"""Seismological risk scoring: fault proximity, historical quakes, liquefaction, PGA."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.types import District, RiskScore

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_risk_zones() -> dict:
    return json.loads((DATA_DIR / "risk_zones.json").read_text())


def score_fault_proximity(district: District) -> float:
    """Score based on distance to known fault systems.

    Bali sits on the Sunda Arc with:
    - Sunda Megathrust ~250km south (M8.5 capable)
    - Flores Back-Arc Thrust ~100km north (M7.5 capable)
    - Local Bali Fault Zone traversing the island
    """
    # All of Bali is within seismic zone — base risk is significant
    base = 45  # Island-wide baseline

    # Latitude proxy for Flores/Sunda position
    # Northern districts closer to Flores thrust
    lat = district.center.lat
    if lat > -8.2:
        base += 10  # Closer to Flores Back-Arc
    elif lat < -8.7:
        base += 8  # Closer to subduction zone

    # Volcanic areas have additional local seismicity
    if district.volcanic_proximity_km < 15:
        base += 15
    elif district.volcanic_proximity_km < 30:
        base += 8

    return min(100, base)


def score_historical_frequency(district: District) -> float:
    """Score based on historical earthquake frequency near the district.

    Major historical events affecting Bali:
    - 1815 eruption-related seismicity
    - 1917 M~6.5 southern Bali (1,500 deaths)
    - 1976 M6.5 Bali Sea
    - 2004 M6.1 south of Bali
    - 2018 M6.9 Lombok (strongly felt in Bali)
    """
    # All of Bali has similar historical exposure due to small island size
    base = 50

    # East Bali slightly higher (closer to Lombok seismic gap)
    lng = district.center.lng
    if lng > 115.4:
        base += 10
    elif lng < 114.8:
        base -= 5  # West Bali slightly lower

    return min(100, base)


def score_liquefaction(district: District, risk_zones: dict) -> float:
    """Score liquefaction susceptibility."""
    base = 10  # Default low

    # Low-lying coastal + sandy soil = high liquefaction risk
    if district.coastal and district.elevation_m < 20:
        base = 45

    # Alluvial river valleys
    if district.elevation_m < 50 and not district.coastal:
        base = 25

    # Check explicit zones
    for zone in risk_zones.get("liquefaction_zones", []):
        if district.id in zone.get("affected_districts", []):
            severity_map = {"high": 40, "medium": 25, "low": 10}
            base = max(base, severity_map.get(zone["severity"], 0))

    # Rocky highland = low risk
    if district.elevation_m > 500:
        base = max(5, base - 20)

    return min(100, base)


def score_ground_acceleration(district: District) -> float:
    """Estimate Peak Ground Acceleration (PGA) risk.

    Based on Indonesian seismic hazard maps (SNI 1726:2019).
    Bali is in Zone 4 (0.3-0.4g design PGA).
    """
    # Base PGA risk for Bali (Zone 4)
    base = 55

    # Soft soil amplification (low-lying coastal areas)
    if district.elevation_m < 20 and district.coastal:
        base += 15  # Site amplification
    elif district.elevation_m < 50:
        base += 8

    # Hard rock (highland) has less amplification
    if district.elevation_m > 600:
        base -= 10

    return min(100, max(0, base))


def compute_seismological_risk(district: District) -> RiskScore:
    """Compute composite seismological risk for a district."""
    risk_zones = _load_risk_zones()

    fault = score_fault_proximity(district)
    historical = score_historical_frequency(district)
    liquefaction = score_liquefaction(district, risk_zones)
    pga = score_ground_acceleration(district)

    weights = {
        "fault_proximity": 0.25,
        "historical_frequency": 0.25,
        "liquefaction": 0.25,
        "ground_acceleration": 0.25,
    }

    factors = {
        "fault_proximity": fault,
        "historical_frequency": historical,
        "liquefaction": liquefaction,
        "ground_acceleration": pga,
    }

    composite = sum(factors[k] * weights[k] for k in weights)

    return RiskScore(
        category="seismological",
        score=round(composite, 1),
        confidence=0.75,
        factors={k: round(v, 1) for k, v in factors.items()},
    )
