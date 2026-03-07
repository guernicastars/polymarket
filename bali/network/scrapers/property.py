"""Property listing scraper for Bali real estate market data.

Scrapes from:
- Rumah123.com — Indonesia's largest property portal
- Lamudi.co.id — International property platform with Indonesia focus
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

from ..core.types import PropertyListing

logger = logging.getLogger(__name__)

# IDR to USD approximate rate
IDR_TO_USD = 1 / 15800

# Bali district name -> district_id mapping (partial, for matching)
DISTRICT_KEYWORDS = {
    "kuta selatan": "kuta_selatan",
    "kuta utara": "kuta_utara",
    "south kuta": "kuta_selatan",
    "north kuta": "kuta_utara",
    "nusa dua": "kuta_selatan",
    "jimbaran": "kuta_selatan",
    "uluwatu": "kuta_selatan",
    "canggu": "kuta_utara",
    "berawa": "kuta_utara",
    "seminyak": "kuta",
    "legian": "kuta",
    "kerobokan": "kuta_utara",
    "sanur": "denpasar_selatan",
    "denpasar": "denpasar_selatan",
    "ubud": "ubud",
    "tegallalang": "tegallalang",
    "gianyar": "gianyar",
    "tabanan": "tabanan",
    "tanah lot": "kediri_tabanan",
    "singaraja": "buleleng",
    "lovina": "banjar",
    "amed": "karangasem",
    "candidasa": "manggis",
    "sidemen": "sidemen",
    "nusa penida": "nusa_penida",
    "pererenan": "kuta_utara",
    "pecatu": "kuta_selatan",
    "ungasan": "kuta_selatan",
    "kedonganan": "kuta",
    "tuban": "kuta",
    "mengwi": "mengwi",
    "medewi": "pekutatan",
    "pemuteran": "gerokgak",
    "munduk": "banjar",
}


def _parse_price_idr(text: str) -> Optional[float]:
    """Parse Indonesian price format (e.g., 'Rp 2,5 Miliar', 'Rp 500 Juta')."""
    text = text.lower().replace("rp", "").replace(".", "").replace(",", ".").strip()

    miliar_match = re.search(r"([\d.]+)\s*miliar", text)
    if miliar_match:
        return float(miliar_match.group(1)) * 1_000_000_000

    juta_match = re.search(r"([\d.]+)\s*juta", text)
    if juta_match:
        return float(juta_match.group(1)) * 1_000_000

    # Try plain number
    num_match = re.search(r"[\d.]+", text)
    if num_match:
        return float(num_match.group())

    return None


def _parse_area(text: str) -> Optional[float]:
    """Parse area string (e.g., '200 m²', '2 are')."""
    text = text.lower().replace(",", ".").strip()

    m2_match = re.search(r"([\d.]+)\s*m", text)
    if m2_match:
        return float(m2_match.group(1))

    are_match = re.search(r"([\d.]+)\s*are", text)
    if are_match:
        return float(are_match.group(1)) * 100  # 1 are = 100 m²

    hectare_match = re.search(r"([\d.]+)\s*h[ae]", text)
    if hectare_match:
        return float(hectare_match.group(1)) * 10000

    return None


def _match_district(location_text: str) -> Optional[str]:
    """Match a location string to a district_id."""
    text = location_text.lower()
    for keyword, district_id in DISTRICT_KEYWORDS.items():
        if keyword in text:
            return district_id
    return None


class PropertyScraper:
    """Scrapes property listings from Indonesian real estate portals."""

    def __init__(self, timeout: int = 30, max_pages: int = 5):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_pages = max_pages
        self.listings: list[PropertyListing] = []
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        }

    async def scrape_rumah123(
        self,
        property_type: str = "dijual",
        location: str = "bali",
    ) -> list[PropertyListing]:
        """Scrape listings from Rumah123.com.

        Args:
            property_type: 'dijual' (for sale) or 'disewa' (for rent)
            location: search location
        """
        listings = []
        base_url = f"https://www.rumah123.com/{property_type}/{location}/tanah/"

        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            for page in range(1, self.max_pages + 1):
                url = f"{base_url}?page={page}" if page > 1 else base_url
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            logger.warning("Rumah123 page %d: HTTP %d", page, resp.status)
                            break

                        html = await resp.text()
                        soup = BeautifulSoup(html, "html.parser")

                        cards = soup.select('[class*="card-featured"]') or soup.select('[class*="listing-card"]')
                        if not cards:
                            logger.info("Rumah123: no more listings at page %d", page)
                            break

                        for card in cards:
                            listing = self._parse_rumah123_card(card)
                            if listing:
                                listings.append(listing)

                        logger.info("Rumah123 page %d: %d listings", page, len(cards))
                        await asyncio.sleep(2)  # Rate limit

                except Exception as e:
                    logger.error("Rumah123 page %d failed: %s", page, e)
                    break

        self.listings.extend(listings)
        return listings

    def _parse_rumah123_card(self, card) -> Optional[PropertyListing]:
        """Parse a single Rumah123 listing card."""
        try:
            title_el = card.select_one("h2, [class*='title']")
            title = title_el.get_text(strip=True) if title_el else "Unknown"

            price_el = card.select_one("[class*='price']")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price_idr = _parse_price_idr(price_text) or 0

            location_el = card.select_one("[class*='location'], [class*='address']")
            location_text = location_el.get_text(strip=True) if location_el else ""

            link_el = card.select_one("a[href]")
            url = link_el.get("href", "") if link_el else ""
            if url and not url.startswith("http"):
                url = f"https://www.rumah123.com{url}"

            area_el = card.select_one("[class*='attribute']")
            area_text = area_el.get_text(strip=True) if area_el else ""
            land_area = _parse_area(area_text) or 0

            listing_id = hashlib.md5(f"r123_{url}".encode()).hexdigest()[:12]
            district_id = _match_district(location_text) or _match_district(title)

            if not district_id:
                return None

            return PropertyListing(
                listing_id=listing_id,
                source="rumah123",
                title=title,
                district_id=district_id,
                price_idr=price_idr,
                price_usd=round(price_idr * IDR_TO_USD, 2),
                area_m2=0,
                land_area_m2=land_area,
                property_type="land",
                title_type="unknown",
                listing_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                url=url,
            )
        except Exception as e:
            logger.debug("Failed to parse Rumah123 card: %s", e)
            return None

    async def scrape_lamudi(
        self,
        property_type: str = "for-sale",
        location: str = "bali",
    ) -> list[PropertyListing]:
        """Scrape listings from Lamudi.co.id."""
        listings = []
        base_url = f"https://www.lamudi.co.id/{location}/{property_type}/"

        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            for page in range(1, self.max_pages + 1):
                url = f"{base_url}?page={page}" if page > 1 else base_url
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            logger.warning("Lamudi page %d: HTTP %d", page, resp.status)
                            break

                        html = await resp.text()
                        soup = BeautifulSoup(html, "html.parser")

                        cards = soup.select('[class*="ListingCell"]') or soup.select('[data-listing-id]')
                        if not cards:
                            break

                        for card in cards:
                            listing = self._parse_lamudi_card(card)
                            if listing:
                                listings.append(listing)

                        logger.info("Lamudi page %d: %d listings", page, len(cards))
                        await asyncio.sleep(2)

                except Exception as e:
                    logger.error("Lamudi page %d failed: %s", page, e)
                    break

        self.listings.extend(listings)
        return listings

    def _parse_lamudi_card(self, card) -> Optional[PropertyListing]:
        """Parse a single Lamudi listing card."""
        try:
            title_el = card.select_one("[class*='Title'], h2, h3")
            title = title_el.get_text(strip=True) if title_el else "Unknown"

            price_el = card.select_one("[class*='Price']")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price_idr = _parse_price_idr(price_text) or 0

            location_el = card.select_one("[class*='Location']")
            location_text = location_el.get_text(strip=True) if location_el else title

            link_el = card.select_one("a[href]")
            url = link_el.get("href", "") if link_el else ""
            if url and not url.startswith("http"):
                url = f"https://www.lamudi.co.id{url}"

            listing_id = hashlib.md5(f"lamudi_{url}".encode()).hexdigest()[:12]
            district_id = _match_district(location_text) or _match_district(title)

            if not district_id:
                return None

            return PropertyListing(
                listing_id=listing_id,
                source="lamudi",
                title=title,
                district_id=district_id,
                price_idr=price_idr,
                price_usd=round(price_idr * IDR_TO_USD, 2),
                area_m2=0,
                land_area_m2=0,
                property_type="unknown",
                title_type="unknown",
                listing_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                url=url,
            )
        except Exception as e:
            logger.debug("Failed to parse Lamudi card: %s", e)
            return None

    async def scrape_all(self) -> list[PropertyListing]:
        """Run all scrapers and return combined listings."""
        rumah123, lamudi = await asyncio.gather(
            self.scrape_rumah123(),
            self.scrape_lamudi(),
        )

        all_listings = rumah123 + lamudi
        logger.info("Total: %d listings from %d sources",
                     len(all_listings), 2)
        return all_listings

    def get_district_stats(self) -> dict[str, dict]:
        """Aggregate listing stats by district."""
        stats: dict[str, dict] = {}
        for listing in self.listings:
            d = listing.district_id
            if d not in stats:
                stats[d] = {"count": 0, "total_price_usd": 0, "prices": [], "sources": set()}
            stats[d]["count"] += 1
            if listing.price_usd > 0:
                stats[d]["total_price_usd"] += listing.price_usd
                stats[d]["prices"].append(listing.price_usd)
            stats[d]["sources"].add(listing.source)

        for d, s in stats.items():
            s["avg_price_usd"] = s["total_price_usd"] / s["count"] if s["count"] > 0 else 0
            s["median_price_usd"] = sorted(s["prices"])[len(s["prices"]) // 2] if s["prices"] else 0
            s["sources"] = list(s["sources"])
            del s["prices"]

        return stats

    def to_json(self) -> list[dict]:
        return [asdict(l) for l in self.listings]
