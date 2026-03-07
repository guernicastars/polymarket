"""Environmental risk scoring: floods, volcanic, landslide, tsunami, coastal erosion."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.types import District, RiskScore

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_risk_zones() -> dict:
    return json.loads((DATA_DIR / "risk_zones.json").read_text())


def score_volcanic_risk(district: District, risk_zones: dict) -> float:
    """Score volcanic risk 0-100 based on proximity and hazard zones."""
    # Base proximity score (inverse distance, capped)
    proximity = district.volcanic_proximity_km
    if proximity <= 4:
        base = 100
    elif proximity <= 8:
        base = 85
    elif proximity <= 15:
        base = 65
    elif proximity <= 30:
        base = 40
    elif proximity <= 50:
        base = 20
    else:
        base = 5

    # Check explicit danger zones
    zone_bonus = 0
    for volcano in risk_zones.get("volcanic_zones", {}).values():
        for zone in volcano.get("danger_zones", []):
            if district.id in zone.get("affected_districts", []):
                severity_map = {"extreme": 40, "high": 25, "medium": 15, "low": 5}
                zone_bonus = max(zone_bonus, severity_map.get(zone["severity"], 0))

    return min(100, base + zone_bonus * 0.3)


def score_flood_risk(district: District, risk_zones: dict) -> float:
    """Score flood risk 0-100."""
    base = 0

    # Elevation factor (lower = riskier)
    if district.elevation_m < 10:
        base = 60
    elif district.elevation_m < 30:
        base = 40
    elif district.elevation_m < 100:
        base = 20
    elif district.elevation_m < 300:
        base = 10
    else:
        base = 3

    # Coastal bonus
    if district.coastal and district.elevation_m < 20:
        base += 15

    # Check flood zones
    for zone in risk_zones.get("flood_zones", []):
        if district.id in zone.get("affected_districts", []):
            severity_map = {"high": 30, "medium": 20, "low": 10}
            base += severity_map.get(zone["severity"], 0)

    return min(100, base)


def score_tsunami_risk(district: District, risk_zones: dict) -> float:
    """Score tsunami risk 0-100."""
    if not district.coastal:
        return 2  # Minimal but non-zero (lahar rivers can carry)

    base = 20  # All coastal areas have baseline risk

    # Check tsunami zones
    for zone in risk_zones.get("tsunami_zones", []):
        if district.id in zone.get("affected_districts", []):
            severity_map = {"high": 55, "medium": 35, "low": 15}
            base = max(base, severity_map.get(zone["severity"], 0))

    # Low elevation amplifier
    if district.elevation_m < 10:
        base = min(100, base * 1.3)
    elif district.elevation_m < 30:
        base = min(100, base * 1.1)

    return min(100, base)


def score_landslide_risk(district: District, risk_zones: dict) -> float:
    """Score landslide risk 0-100."""
    # Elevation + slope proxy
    if district.elevation_m > 800:
        base = 50
    elif district.elevation_m > 400:
        base = 35
    elif district.elevation_m > 200:
        base = 20
    elif district.elevation_m > 100:
        base = 10
    else:
        base = 3

    # Check landslide zones
    for zone in risk_zones.get("landslide_zones", []):
        if district.id in zone.get("affected_districts", []):
            severity_map = {"high": 35, "medium": 20, "low": 10}
            base += severity_map.get(zone["severity"], 0)

    return min(100, base)


def score_coastal_erosion(district: District) -> float:
    """Score coastal erosion risk 0-100."""
    if not district.coastal:
        return 0

    base = 25  # All coastal areas
    if district.elevation_m < 10:
        base += 25
    if district.tourism_intensity > 0.7:
        base += 15  # Over-development accelerates erosion
    if "cliff" in district.tags:
        base += 10

    return min(100, base)


def compute_environmental_risk(district: District) -> RiskScore:
    """Compute composite environmental risk for a district."""
    risk_zones = _load_risk_zones()

    volcanic = score_volcanic_risk(district, risk_zones)
    flood = score_flood_risk(district, risk_zones)
    tsunami = score_tsunami_risk(district, risk_zones)
    landslide = score_landslide_risk(district, risk_zones)
    erosion = score_coastal_erosion(district)

    # Weighted combination
    weights = {
        "volcanic": 0.35,
        "flood": 0.25,
        "tsunami": 0.20,
        "landslide": 0.15,
        "coastal_erosion": 0.05,
    }

    factors = {
        "volcanic": volcanic,
        "flood": flood,
        "tsunami": tsunami,
        "landslide": landslide,
        "coastal_erosion": erosion,
    }

    composite = sum(factors[k] * weights[k] for k in weights)

    # Confidence based on data completeness
    confidence = 0.8  # Good baseline from BNPB data

    return RiskScore(
        category="environmental",
        score=round(composite, 1),
        confidence=confidence,
        factors={k: round(v, 1) for k, v in factors.items()},
    )
