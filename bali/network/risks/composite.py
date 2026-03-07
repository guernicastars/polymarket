"""Composite risk aggregation across all 4 categories."""

from __future__ import annotations

from datetime import datetime, timezone

from ..core.types import CompositeRisk, District
from .administrative import compute_administrative_risk
from .environmental import compute_environmental_risk
from .legal import compute_legal_risk
from .seismological import compute_seismological_risk

# Category weights for composite score
WEIGHTS = {
    "environmental": 0.30,
    "seismological": 0.25,
    "legal": 0.25,
    "administrative": 0.20,
}


def investment_grade(score: float) -> str:
    """Convert composite risk score to investment grade.

    A (0-25): Excellent — low risk, strong fundamentals
    B (25-40): Good — manageable risks, solid investment
    C (40-55): Moderate — significant risks, careful due diligence needed
    D (55-70): High risk — substantial concerns, expert guidance essential
    F (70+): Extreme — avoid or requires exceptional risk tolerance
    """
    if score < 25:
        return "A"
    elif score < 40:
        return "B"
    elif score < 55:
        return "C"
    elif score < 70:
        return "D"
    else:
        return "F"


def compute_composite_risk(district: District) -> CompositeRisk:
    """Compute full risk assessment for a district."""
    env = compute_environmental_risk(district)
    seis = compute_seismological_risk(district)
    legal = compute_legal_risk(district)
    admin = compute_administrative_risk(district)

    composite = (
        env.score * WEIGHTS["environmental"]
        + seis.score * WEIGHTS["seismological"]
        + legal.score * WEIGHTS["legal"]
        + admin.score * WEIGHTS["administrative"]
    )
    composite = round(composite, 1)

    return CompositeRisk(
        district_id=district.id,
        environmental=env,
        seismological=seis,
        legal=legal,
        administrative=admin,
        composite_score=composite,
        investment_grade=investment_grade(composite),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def compute_all_risks(districts: dict[str, District]) -> dict[str, CompositeRisk]:
    """Compute risk scores for all districts."""
    results = {}
    for district_id, district in districts.items():
        results[district_id] = compute_composite_risk(district)
    return results


def rank_districts_by_risk(risks: dict[str, CompositeRisk]) -> list[tuple[str, CompositeRisk]]:
    """Rank districts from safest to riskiest."""
    return sorted(risks.items(), key=lambda x: x[1].composite_score)


def rank_by_investment_value(
    districts: dict[str, District],
    risks: dict[str, CompositeRisk],
) -> list[dict]:
    """Rank districts by investment value (low risk + reasonable price).

    Returns a sorted list combining risk score with price data.
    Lower score = better investment opportunity.
    """
    results = []
    for district_id, district in districts.items():
        risk = risks.get(district_id)
        if not risk:
            continue

        # Value score: risk-adjusted price
        # Normalize price to 0-100 scale (max ~$7000/m2 in Bali)
        price_score = min(100, district.avg_land_price_usd_m2 / 70)

        # Combine: 60% risk, 40% price affordability
        value_score = risk.composite_score * 0.6 + price_score * 0.4

        results.append({
            "district_id": district_id,
            "name": district.name,
            "regency": district.regency,
            "composite_risk": risk.composite_score,
            "investment_grade": risk.investment_grade,
            "avg_price_usd_m2": district.avg_land_price_usd_m2,
            "value_score": round(value_score, 1),
            "tourism_intensity": district.tourism_intensity,
            "infrastructure_index": district.infrastructure_index,
        })

    return sorted(results, key=lambda x: x["value_score"])
