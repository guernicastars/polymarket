"""Data classes for Bali real estate risk network model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Regency(Enum):
    """Bali's 9 kabupaten + 1 kota."""
    JEMBRANA = "jembrana"
    TABANAN = "tabanan"
    BADUNG = "badung"
    DENPASAR = "denpasar"
    GIANYAR = "gianyar"
    KLUNGKUNG = "klungkung"
    BANGLI = "bangli"
    KARANGASEM = "karangasem"
    BULELENG = "buleleng"


class EdgeType(Enum):
    """Types of connections between districts."""
    ROAD = "road"
    PROXIMITY = "proximity"
    SHARED_RISK_ZONE = "shared_risk_zone"
    ECONOMIC_TIE = "economic_tie"
    INFRASTRUCTURE = "infrastructure"
    WATERSHED = "watershed"


class RiskCategory(Enum):
    ENVIRONMENTAL = "environmental"
    SEISMOLOGICAL = "seismological"
    LEGAL = "legal"
    ADMINISTRATIVE = "administrative"


class LandTitleType(Enum):
    """Indonesian land title types."""
    SHM = "shm"           # Sertifikat Hak Milik - full ownership (Indonesian only)
    HGB = "hgb"           # Hak Guna Bangunan - right to build (foreigners via PMA)
    HAK_PAKAI = "hak_pakai"  # Right to use (foreigners directly, 25+20+20 yr)
    GIRIK = "girik"       # Traditional/customary (unregistered, high risk)
    STRATA = "strata"     # Strata title for apartments/condos


class OwnershipPathway(Enum):
    """Foreign ownership structures in Bali."""
    HAK_PAKAI_DIRECT = "hak_pakai_direct"   # Safest for foreigners
    PMA_COMPANY = "pma_company"              # PT PMA - foreign-owned company
    NOMINEE = "nominee"                       # Indonesian nominee (illegal but common)
    LEASEHOLD = "leasehold"                   # Long-term lease (Hak Sewa)


class ZoneType(Enum):
    """Zoning classifications."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    TOURISM = "tourism"
    AGRICULTURAL = "agricultural"
    GREEN_BELT = "green_belt"
    SACRED = "sacred"       # Balinese temple/holy site buffer zone
    COASTAL = "coastal"     # Sempadan pantai (coastal setback)


@dataclass
class GeoCoord:
    lat: float
    lng: float


@dataclass
class District:
    """A kecamatan (district) node in the Bali network."""
    id: str
    name: str
    regency: str
    center: GeoCoord
    area_km2: float
    population: int
    elevation_m: float              # Average elevation
    coastal: bool
    volcanic_proximity_km: float    # Distance to nearest volcano (Agung/Batur)
    dominant_zone: str
    dominant_title: str
    infrastructure_index: float     # 0-1 scale
    tourism_intensity: float        # 0-1 scale
    foreign_investor_density: float # 0-1 scale
    avg_land_price_usd_m2: float
    tags: list[str] = field(default_factory=list)


@dataclass
class RiskScore:
    """Risk score for a single category."""
    category: str
    score: float                # 0-100 (higher = riskier)
    confidence: float           # 0-1
    factors: dict[str, float]   # Component factor scores


@dataclass
class CompositeRisk:
    """Aggregated multi-category risk assessment."""
    district_id: str
    environmental: RiskScore
    seismological: RiskScore
    legal: RiskScore
    administrative: RiskScore
    composite_score: float      # Weighted aggregate 0-100
    investment_grade: str       # A/B/C/D/F
    computed_at: str


@dataclass
class Edge:
    """Connection between two districts."""
    source: str
    target: str
    edge_type: str
    weight: float               # 0-1 (strength of connection)
    properties: dict = field(default_factory=dict)


@dataclass
class SeismicEvent:
    """An earthquake event from BMKG."""
    event_id: str
    timestamp: str
    lat: float
    lng: float
    depth_km: float
    magnitude: float
    region: str
    felt_districts: list[str] = field(default_factory=list)


@dataclass
class PropertyListing:
    """A scraped property listing."""
    listing_id: str
    source: str                 # rumah123, lamudi, etc.
    title: str
    district_id: str
    price_idr: float
    price_usd: float
    area_m2: float
    land_area_m2: float
    property_type: str          # villa, land, apartment, house
    title_type: str
    listing_date: str
    url: str
    lat: Optional[float] = None
    lng: Optional[float] = None


@dataclass
class RiskZone:
    """A geographic risk zone polygon."""
    zone_id: str
    zone_type: str              # flood, tsunami, landslide, volcanic, liquefaction
    severity: str               # low, medium, high, extreme
    affected_districts: list[str]
    description: str
