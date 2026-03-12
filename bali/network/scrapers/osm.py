"""OpenStreetMap infrastructure data extractor via Overpass API.

Extracts infrastructure metrics per district:
- Road density and quality
- Hospital/clinic proximity
- School density
- Utility coverage (water, power)
- Commercial/retail density
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bali bounding box for Overpass queries
BALI_BBOX = "-8.85,114.40,-8.05,115.75"


@dataclass
class InfrastructureMetrics:
    """Infrastructure metrics for a geographic area."""
    district_id: str
    road_count: int = 0
    road_km: float = 0.0
    hospital_count: int = 0
    clinic_count: int = 0
    school_count: int = 0
    atm_count: int = 0
    restaurant_count: int = 0
    hotel_count: int = 0
    fuel_station_count: int = 0
    supermarket_count: int = 0
    police_count: int = 0
    fire_station_count: int = 0


class OSMScraper:
    """Extracts infrastructure data from OpenStreetMap via Overpass API."""

    def __init__(self, timeout: int = 60):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.metrics: dict[str, InfrastructureMetrics] = {}

    async def _query_overpass(self, query: str) -> dict:
        """Execute an Overpass API query."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(OVERPASS_URL, data={"data": query}) as resp:
                    if resp.status != 200:
                        logger.error("Overpass API: HTTP %d", resp.status)
                        return {"elements": []}
                    return await resp.json(content_type=None)
            except Exception as e:
                logger.error("Overpass query failed: %s", e)
                return {"elements": []}

    async def count_amenities(
        self,
        amenity_type: str,
        bbox: str = BALI_BBOX,
    ) -> list[dict]:
        """Count amenities of a given type within a bounding box."""
        query = f"""
        [out:json][timeout:30];
        (
          node["amenity"="{amenity_type}"]({bbox});
          way["amenity"="{amenity_type}"]({bbox});
        );
        out center count;
        """
        data = await self._query_overpass(query)
        return data.get("elements", [])

    async def get_hospitals(self, bbox: str = BALI_BBOX) -> list[dict]:
        """Get hospitals and clinics in Bali."""
        query = f"""
        [out:json][timeout:30];
        (
          node["amenity"="hospital"]({bbox});
          way["amenity"="hospital"]({bbox});
          node["amenity"="clinic"]({bbox});
          way["amenity"="clinic"]({bbox});
          node["amenity"="doctors"]({bbox});
        );
        out center;
        """
        data = await self._query_overpass(query)
        elements = data.get("elements", [])
        logger.info("OSM: found %d hospitals/clinics in Bali", len(elements))
        return elements

    async def get_roads(self, bbox: str = BALI_BBOX) -> list[dict]:
        """Get road network summary."""
        query = f"""
        [out:json][timeout:45];
        way["highway"~"primary|secondary|tertiary|trunk|motorway"]({bbox});
        out count;
        """
        data = await self._query_overpass(query)
        return data.get("elements", [])

    async def get_infrastructure_summary(self, bbox: str = BALI_BBOX) -> dict:
        """Get a comprehensive infrastructure summary for Bali."""
        amenity_types = [
            "hospital", "clinic", "school", "bank",
            "restaurant", "hotel", "fuel", "supermarket",
            "police", "fire_station",
        ]

        results = {}
        for amenity in amenity_types:
            elements = await self.count_amenities(amenity, bbox)
            results[amenity] = len(elements)
            await asyncio.sleep(1)  # Rate limit Overpass API

        logger.info("OSM infrastructure summary: %s", results)
        return results

    async def get_poi_near_point(
        self,
        lat: float,
        lng: float,
        radius_m: int = 5000,
        amenity_types: list[str] | None = None,
    ) -> dict[str, int]:
        """Count points of interest near a lat/lng point."""
        if amenity_types is None:
            amenity_types = ["hospital", "school", "restaurant", "bank", "fuel"]

        results = {}
        for amenity in amenity_types:
            query = f"""
            [out:json][timeout:15];
            (
              node["amenity"="{amenity}"](around:{radius_m},{lat},{lng});
              way["amenity"="{amenity}"](around:{radius_m},{lat},{lng});
            );
            out count;
            """
            data = await self._query_overpass(query)
            elements = data.get("elements", [])
            # Overpass count queries return a single element with tags.total
            if elements and "tags" in elements[0]:
                results[amenity] = int(elements[0]["tags"].get("total", 0))
            else:
                results[amenity] = len(elements)
            await asyncio.sleep(1)

        return results

    async def compute_district_metrics(
        self,
        districts: dict[str, dict],
    ) -> dict[str, InfrastructureMetrics]:
        """Compute infrastructure metrics for each district.

        Args:
            districts: dict of district_id -> {lat, lng, ...}
        """
        for district_id, info in districts.items():
            lat = info.get("lat", 0)
            lng = info.get("lng", 0)
            if lat == 0 or lng == 0:
                continue

            logger.info("Computing OSM metrics for %s", district_id)
            poi_counts = await self.get_poi_near_point(lat, lng, radius_m=5000)

            self.metrics[district_id] = InfrastructureMetrics(
                district_id=district_id,
                hospital_count=poi_counts.get("hospital", 0),
                clinic_count=poi_counts.get("clinic", 0),
                school_count=poi_counts.get("school", 0),
                atm_count=poi_counts.get("bank", 0),
                restaurant_count=poi_counts.get("restaurant", 0),
                fuel_station_count=poi_counts.get("fuel", 0),
            )

            await asyncio.sleep(2)  # Be nice to Overpass

        return self.metrics
