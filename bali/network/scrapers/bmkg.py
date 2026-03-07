"""BMKG (Badan Meteorologi, Klimatologi, dan Geofisika) earthquake feed scraper.

Data source: https://data.bmkg.go.id/DataMKG/TEWS/
Real-time earthquake data for Indonesia.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree

import aiohttp

from ..core.types import GeoCoord, SeismicEvent

logger = logging.getLogger(__name__)

# BMKG public earthquake data endpoints
BMKG_REALTIME_URL = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
BMKG_RECENT_URL = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
BMKG_FELT_URL = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"

# Bali bounding box (approx)
BALI_BOUNDS = {
    "lat_min": -9.0,
    "lat_max": -8.0,
    "lng_min": 114.4,
    "lng_max": 115.8,
}

# Extended region for felt earthquakes (Lombok, Java east, Flores)
BALI_REGION_BOUNDS = {
    "lat_min": -10.0,
    "lat_max": -7.5,
    "lng_min": 113.5,
    "lng_max": 117.0,
}


def _in_bali_region(lat: float, lng: float) -> bool:
    """Check if coordinates are within Bali's earthquake relevance zone."""
    b = BALI_REGION_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lng_min"] <= lng <= b["lng_max"]


def _parse_bmkg_coordinates(coord_str: str) -> tuple[float, float]:
    """Parse BMKG coordinate format (e.g., '8.35 LS', '115.50 BT')."""
    parts = coord_str.strip().split()
    if len(parts) != 2:
        return 0.0, 0.0

    value = float(parts[0])
    direction = parts[1].upper()

    if direction in ("LS", "S"):
        return -abs(value), 0  # South latitude
    elif direction in ("LU", "N"):
        return abs(value), 0
    elif direction in ("BT", "E"):
        return 0, abs(value)
    elif direction in ("BB", "W"):
        return 0, -abs(value)

    return value, 0


def _parse_gempa(data: dict) -> Optional[SeismicEvent]:
    """Parse a single BMKG earthquake record."""
    try:
        # Parse coordinates
        coords = data.get("Coordinates", "").split(",")
        if len(coords) == 2:
            lat = float(coords[0])
            lng = float(coords[1])
        else:
            lat_str = data.get("Lintang", "0 LS")
            lng_str = data.get("Bujur", "0 BT")
            lat, _ = _parse_bmkg_coordinates(lat_str)
            _, lng = _parse_bmkg_coordinates(lng_str)

        if not _in_bali_region(lat, lng):
            return None

        magnitude = float(data.get("Magnitude", "0"))
        depth_str = data.get("Kedalaman", "0 km").replace(" km", "").replace(" Km", "")
        depth = float(depth_str)

        # Parse timestamp
        date_str = data.get("Tanggal", "")
        time_str = data.get("Jam", "")
        timestamp = f"{date_str} {time_str}".strip()

        region = data.get("Wilayah", data.get("Area", "Unknown"))

        event_id = f"bmkg_{date_str}_{time_str}_{magnitude}".replace(" ", "_").replace(":", "")

        return SeismicEvent(
            event_id=event_id,
            timestamp=timestamp,
            lat=lat,
            lng=lng,
            depth_km=depth,
            magnitude=magnitude,
            region=region,
        )
    except (ValueError, KeyError) as e:
        logger.warning("Failed to parse BMKG record: %s — %s", e, data)
        return None


class BMKGScraper:
    """Scrapes earthquake data from BMKG API."""

    def __init__(self, timeout: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.events: list[SeismicEvent] = []

    async def fetch_latest(self) -> Optional[SeismicEvent]:
        """Fetch the most recent earthquake."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(BMKG_REALTIME_URL) as resp:
                    if resp.status != 200:
                        logger.error("BMKG realtime: HTTP %d", resp.status)
                        return None
                    data = await resp.json(content_type=None)
                    gempa = data.get("Infogempa", {}).get("gempa", {})
                    return _parse_gempa(gempa)
            except Exception as e:
                logger.error("BMKG realtime fetch failed: %s", e)
                return None

    async def fetch_recent(self, felt_only: bool = False) -> list[SeismicEvent]:
        """Fetch recent earthquakes (M5+ or felt)."""
        url = BMKG_FELT_URL if felt_only else BMKG_RECENT_URL
        events = []

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.error("BMKG recent: HTTP %d", resp.status)
                        return events
                    data = await resp.json(content_type=None)
                    gempa_list = data.get("Infogempa", {}).get("gempa", [])

                    for gempa in gempa_list:
                        event = _parse_gempa(gempa)
                        if event:
                            events.append(event)

            except Exception as e:
                logger.error("BMKG recent fetch failed: %s", e)

        self.events.extend(events)
        logger.info("BMKG: fetched %d events relevant to Bali region", len(events))
        return events

    async def fetch_all(self) -> list[SeismicEvent]:
        """Fetch both recent and felt earthquakes."""
        recent, felt = await asyncio.gather(
            self.fetch_recent(felt_only=False),
            self.fetch_recent(felt_only=True),
        )

        # Deduplicate by event_id
        seen = set()
        all_events = []
        for event in recent + felt:
            if event.event_id not in seen:
                seen.add(event.event_id)
                all_events.append(event)

        self.events = all_events
        return all_events

    def get_events_near_bali(self, max_distance_km: float = 300) -> list[SeismicEvent]:
        """Filter events within specified distance of Bali center."""
        from ..core.graph import haversine_km
        bali_center = GeoCoord(-8.4095, 115.1889)

        return [
            e for e in self.events
            if haversine_km(bali_center, GeoCoord(e.lat, e.lng)) <= max_distance_km
        ]

    def to_json(self) -> list[dict]:
        """Export events as JSON-serializable dicts."""
        return [asdict(e) for e in self.events]
