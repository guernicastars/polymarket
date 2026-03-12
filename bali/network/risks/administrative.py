"""Administrative risk scoring: permits, infrastructure, bureaucracy, utilities."""

from __future__ import annotations

from ..core.types import District, RiskScore


def score_permit_complexity(district: District) -> float:
    """Score difficulty of obtaining building permits (IMB/PBG).

    Post-2021 Indonesia uses PBG (Persetujuan Bangunan Gedung) via OSS system.
    Complexity varies by:
    - Local government capacity
    - Zoning compliance requirements
    - Environmental impact assessment (AMDAL) needs
    - Heritage/cultural site proximity
    """
    base = 35  # Default: Indonesian permits are moderately complex

    # Urban areas: better systems but more requirements
    if district.infrastructure_index > 0.8:
        base -= 10  # Digital systems, experienced officials
    elif district.infrastructure_index < 0.4:
        base += 20  # Under-resourced local government, long delays

    # Tourism zones: well-trodden permit path
    if district.dominant_zone == "tourism":
        base -= 5
    elif district.dominant_zone == "agricultural":
        base += 25  # Land use conversion requires AMDAL + multi-ministry approval

    # Sacred sites require additional cultural impact assessment
    if "sacred" in district.tags or "temple" in district.tags:
        base += 15

    # Nusa Penida: island logistics add complexity
    if "island" in district.tags:
        base += 10

    return min(100, max(0, base))


def score_infrastructure_quality(district: District) -> float:
    """Score infrastructure risk (inverse of quality).

    Factors: road quality, water supply, electricity reliability,
    internet connectivity, waste management.
    """
    # Direct inverse of infrastructure index
    base = (1 - district.infrastructure_index) * 80

    # Remote areas: power outages, no piped water
    if district.infrastructure_index < 0.3:
        base += 15

    # Island logistics
    if "island" in district.tags:
        base += 20  # Limited water, electricity from undersea cable

    # Highland access issues
    if district.elevation_m > 600 and district.infrastructure_index < 0.5:
        base += 10

    return min(100, max(0, base))


def score_bureaucratic_complexity(district: District) -> float:
    """Score bureaucratic overhead for real estate transactions.

    Includes: notary availability, BPN (land office) capacity,
    tax office efficiency, local government responsiveness.
    """
    base = 40  # Indonesia baseline bureaucratic complexity

    # Denpasar: most efficient bureaucracy in Bali
    if district.regency == "denpasar":
        base -= 15
    # Badung: second most developed
    elif district.regency == "badung":
        base -= 10
    # Remote regencies
    elif district.regency in ("jembrana", "karangasem"):
        base += 15
    elif district.regency == "bangli":
        base += 10

    # High foreign activity = more notaries experienced with foreigners
    if district.foreign_investor_density > 0.5:
        base -= 8
    elif district.foreign_investor_density < 0.1:
        base += 12  # Few professionals experienced with foreign transactions

    return min(100, max(0, base))


def score_utility_access(district: District) -> float:
    """Score risk from utility access issues.

    Critical for real estate: PDAM water, PLN electricity,
    internet (fiber availability), sewage/septic.
    """
    # Proxy from infrastructure index + specific factors
    base = (1 - district.infrastructure_index) * 70

    # Coastal areas: saltwater intrusion in wells
    if district.coastal and district.infrastructure_index < 0.5:
        base += 10

    # Highland: water access better (springs) but power less reliable
    if district.elevation_m > 500:
        base += 5  # Power reliability

    # Urban areas: full utility coverage
    if district.infrastructure_index > 0.85:
        base = max(5, base - 10)

    return min(100, max(0, base))


def compute_administrative_risk(district: District) -> RiskScore:
    """Compute composite administrative risk for a district."""
    permits = score_permit_complexity(district)
    infrastructure = score_infrastructure_quality(district)
    bureaucracy = score_bureaucratic_complexity(district)
    utilities = score_utility_access(district)

    weights = {
        "permit_complexity": 0.30,
        "infrastructure_quality": 0.25,
        "bureaucratic_complexity": 0.25,
        "utility_access": 0.20,
    }

    factors = {
        "permit_complexity": permits,
        "infrastructure_quality": infrastructure,
        "bureaucratic_complexity": bureaucracy,
        "utility_access": utilities,
    }

    composite = sum(factors[k] * weights[k] for k in weights)

    return RiskScore(
        category="administrative",
        score=round(composite, 1),
        confidence=0.65,
        factors={k: round(v, 1) for k, v in factors.items()},
    )
