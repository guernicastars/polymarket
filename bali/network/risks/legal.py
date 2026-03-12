"""Legal risk scoring: ownership pathways, title types, zoning, dispute history."""

from __future__ import annotations

from ..core.types import District, RiskScore


def score_ownership_pathway(district: District) -> float:
    """Score risk of foreign ownership complications.

    Indonesian law prohibits foreign freehold ownership (Hak Milik/SHM).
    Legal pathways:
    - Hak Pakai (25+20+20 yr): safest, limited property types
    - PT PMA (foreign company): complex setup, must demonstrate business purpose
    - Nominee (illegal): extremely common, zero legal protection
    - Leasehold: safest for villas but no ownership rights
    """
    base = 40  # Baseline: foreign ownership is inherently complex in Indonesia

    # High foreign investor density = more nominee arrangements
    if district.foreign_investor_density > 0.7:
        base += 20  # Nominee structures very common
    elif district.foreign_investor_density > 0.4:
        base += 12
    elif district.foreign_investor_density > 0.2:
        base += 5

    # Tourism zones have more developed legal infrastructure for foreigners
    if district.dominant_zone == "tourism":
        base -= 10  # Notaries and lawyers experienced with foreign deals
    elif district.dominant_zone == "agricultural":
        base += 15  # Agricultural land conversion very restricted (PP 16/2004)

    # Rural areas = more girik (unregistered) land
    if district.dominant_title == "girik":
        base += 25
    elif district.dominant_title == "shm" and district.foreign_investor_density > 0.5:
        base += 10  # SHM areas with foreign interest = nominee risk

    return min(100, max(0, base))


def score_title_risk(district: District) -> float:
    """Score land title security risk.

    SHM (Hak Milik): strongest but Indonesian-only
    HGB: strong, foreigners via PMA, but has expiry
    Hak Pakai: foreigners direct, shorter term
    Girik: high risk, needs conversion
    """
    title_base = {
        "shm": 20,       # Low risk if legitimate (but foreigners can't hold)
        "hgb": 30,       # Moderate: needs renewal, corporate overhead
        "hak_pakai": 35, # Moderate: term limits, renewal uncertainty
        "girik": 70,     # High: unregistered, boundary disputes common
        "strata": 25,    # Low-moderate: clear but building-dependent
    }
    base = title_base.get(district.dominant_title, 50)

    # Rural areas: more boundary disputes due to informal land records
    if district.infrastructure_index < 0.4:
        base += 15
    elif district.infrastructure_index < 0.6:
        base += 8

    # High-value areas: better documented but more contentious
    if district.avg_land_price_usd_m2 > 3000:
        base -= 5  # Better records
    elif district.avg_land_price_usd_m2 > 1000:
        base -= 3

    return min(100, max(0, base))


def score_zoning_compliance(district: District) -> float:
    """Score risk from zoning issues.

    Bali has complex spatial planning (RTRW):
    - Green belt: no development allowed
    - Sacred zones: temple buffer (pura) restrict construction
    - Coastal setback (sempadan pantai): building restrictions within 100m
    - Agricultural: difficult conversion to commercial/tourism
    """
    base = 25  # Default moderate

    zone_risk = {
        "commercial": 15,    # Clear zoning, well-regulated
        "residential": 20,
        "tourism": 25,       # Tourism zones under increasing regulation
        "agricultural": 55,  # Conversion complex, moratorium risk
        "green_belt": 85,    # Extreme: basically unbuildable
        "sacred": 75,        # Near-impossible: temple radius rules
        "coastal": 45,       # Coastal setback + environmental rules
    }
    base = zone_risk.get(district.dominant_zone, 35)

    # Sacred site density (approximated by tags)
    if "sacred" in district.tags:
        base += 15
    if "temple" in district.tags:
        base += 10

    # Coastal setback for coastal districts
    if district.coastal:
        base += 10

    return min(100, max(0, base))


def score_dispute_density(district: District) -> float:
    """Score historical land dispute risk.

    Higher in areas with:
    - Mixed title types (formal vs customary)
    - Rapid price appreciation (incentivizes disputes)
    - High foreign involvement (expat-local disputes)
    - Low infrastructure (weak land administration)
    """
    base = 20

    # Price appreciation zones have more disputes
    if district.avg_land_price_usd_m2 > 3000:
        base += 20
    elif district.avg_land_price_usd_m2 > 1000:
        base += 10

    # High foreign density = more cross-cultural disputes
    if district.foreign_investor_density > 0.7:
        base += 18
    elif district.foreign_investor_density > 0.4:
        base += 10

    # Low infrastructure = weak land administration
    if district.infrastructure_index < 0.4:
        base += 15

    return min(100, max(0, base))


def compute_legal_risk(district: District) -> RiskScore:
    """Compute composite legal risk for a district."""
    ownership = score_ownership_pathway(district)
    title = score_title_risk(district)
    zoning = score_zoning_compliance(district)
    disputes = score_dispute_density(district)

    weights = {
        "ownership_pathway": 0.30,
        "title_security": 0.25,
        "zoning_compliance": 0.25,
        "dispute_density": 0.20,
    }

    factors = {
        "ownership_pathway": ownership,
        "title_security": title,
        "zoning_compliance": zoning,
        "dispute_density": disputes,
    }

    composite = sum(factors[k] * weights[k] for k in weights)

    return RiskScore(
        category="legal",
        score=round(composite, 1),
        confidence=0.70,  # Legal data is less precise than natural hazard data
        factors={k: round(v, 1) for k, v in factors.items()},
    )
