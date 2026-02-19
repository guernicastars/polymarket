"""News source fetchers — one class per source type.

Each fetcher:
  1. Hits the source API/RSS/page
  2. Returns a list of raw articles (title, body, url, published_at)
  3. The orchestrator then runs NLP and writes to ClickHouse

Fetchers are designed to be stateless — dedup happens at the writer level
via article_id (SHA256 of URL).
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from .sources import NewsSource

logger = logging.getLogger(__name__)

HTTPX_TIMEOUT = 30.0
USER_AGENT = "PolymarketSignals/1.0 (conflict-market-intelligence)"


class RSSFetcher:
    """Fetch articles from RSS/Atom feeds."""

    def __init__(self, source: NewsSource):
        self.source = source

    async def fetch(self, since: Optional[datetime] = None) -> list[dict]:
        """Fetch new articles from RSS feed.

        Args:
            since: only return articles published after this time

        Returns:
            list of dicts: {title, body, url, published_at}
        """
        articles = []
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                resp = await client.get(
                    self.source.base_url,
                    headers={
                        "User-Agent": USER_AGENT,
                        **self.source.headers,
                    },
                    follow_redirects=True,
                )
                resp.raise_for_status()

            root = ET.fromstring(resp.text)
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for item in items:
                article = self._parse_item(item)
                if article:
                    if since and article["published_at"] < since:
                        continue
                    articles.append(article)

        except Exception as e:
            logger.error("RSS fetch failed for %s: %s", self.source.name, e)

        logger.info("RSS %s: fetched %d articles", self.source.name, len(articles))
        return articles

    def _parse_item(self, item: ET.Element) -> Optional[dict]:
        """Parse a single RSS/Atom item."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Try RSS format
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        date_el = item.find("pubDate")

        # Try Atom format
        if title_el is None:
            title_el = item.find("atom:title", ns)
        if link_el is None:
            link_el = item.find("atom:link", ns)
        if desc_el is None:
            desc_el = item.find("atom:summary", ns) or item.find("atom:content", ns)
        if date_el is None:
            date_el = item.find("atom:updated", ns) or item.find("atom:published", ns)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        if not title:
            return None

        # Get link (RSS vs Atom)
        if link_el is not None:
            url = link_el.get("href", "") or (link_el.text or "").strip()
        else:
            url = ""

        body = ""
        if desc_el is not None and desc_el.text:
            body = self._strip_html(desc_el.text)

        published_at = self._parse_date(date_el)

        return {
            "title": title,
            "body": body,
            "url": url,
            "published_at": published_at,
        }

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _parse_date(el: Optional[ET.Element]) -> datetime:
        """Parse various date formats."""
        if el is None or not el.text:
            return datetime.utcnow()
        text = el.text.strip()
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",   # RFC 822
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(text, fmt)
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        return datetime.utcnow()


class ISWFetcher:
    """Fetch and parse Institute for the Study of War daily assessments."""

    ASSESSMENT_URL = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment"

    def __init__(self, source: NewsSource):
        self.source = source

    async def fetch(self, since: Optional[datetime] = None) -> list[dict]:
        """Fetch latest ISW assessment.

        ISW publishes daily — we fetch the latest assessment page
        and extract the body text.
        """
        articles = []
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                # ISW assessment index page
                resp = await client.get(
                    self.ASSESSMENT_URL,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )
                resp.raise_for_status()

            # Parse out the latest assessment link and content
            text = resp.text

            # Extract assessment title and date
            title_match = re.search(
                r"Russian Offensive Campaign Assessment[,\s]+(\w+ \d+,? \d{4})",
                text,
            )
            title = title_match.group(0) if title_match else "ISW Daily Assessment"

            # Extract body text (between key markers)
            body = self._extract_body(text)

            articles.append({
                "title": title,
                "body": body[:10000],
                "url": str(resp.url),
                "published_at": datetime.utcnow(),
            })

        except Exception as e:
            logger.error("ISW fetch failed: %s", e)

        logger.info("ISW: fetched %d articles", len(articles))
        return articles

    @staticmethod
    def _extract_body(html: str) -> str:
        """Extract readable text from ISW HTML."""
        # Remove scripts and styles
        clean = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
        # Remove tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()


class DeepStateFetcher:
    """Fetch DeepState map control data.

    DeepState provides a GeoJSON-based API with settlement control status.
    Falls back to scraping the main page for latest updates.
    """

    API_URL = "https://deepstatemap.live/api/history/"
    UPDATES_URL = "https://deepstatemap.live"

    def __init__(self, source: NewsSource):
        self.source = source

    async def fetch(self, since: Optional[datetime] = None) -> list[dict]:
        """Fetch latest control changes from DeepState.

        Returns articles describing control changes.
        """
        articles = []
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                # Try API endpoint first
                resp = await client.get(
                    self.API_URL,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )

                if resp.status_code == 200:
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    articles = self._parse_api_response(data)
                else:
                    # Fallback: scrape main page for update text
                    resp = await client.get(
                        self.UPDATES_URL,
                        headers={"User-Agent": USER_AGENT},
                        follow_redirects=True,
                    )
                    resp.raise_for_status()
                    articles = self._parse_html_updates(resp.text)

        except Exception as e:
            logger.error("DeepState fetch failed: %s", e)

        logger.info("DeepState: fetched %d updates", len(articles))
        return articles

    def _parse_api_response(self, data: dict) -> list[dict]:
        """Parse DeepState API GeoJSON response into articles."""
        articles = []
        if isinstance(data, list):
            for item in data[:20]:  # Latest 20 changes
                title = item.get("title", "DeepState Update")
                body = item.get("description", "")
                date_str = item.get("date", "")
                articles.append({
                    "title": title,
                    "body": body,
                    "url": self.UPDATES_URL,
                    "published_at": self._parse_ds_date(date_str),
                })
        return articles

    def _parse_html_updates(self, html: str) -> list[dict]:
        """Extract update summaries from DeepState HTML."""
        # Look for update blocks
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean)

        # Return as single article with summary
        return [{
            "title": "DeepState Frontline Update",
            "body": clean[:5000],
            "url": self.UPDATES_URL,
            "published_at": datetime.utcnow(),
        }]

    @staticmethod
    def _parse_ds_date(date_str: str) -> datetime:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        return datetime.utcnow()


class TelegramFetcher:
    """Fetch from Telegram public channels via t.me web preview.

    Uses the public web preview (no API key needed for public channels).
    Limited to ~20 most recent messages.
    """

    def __init__(self, source: NewsSource):
        self.source = source
        # Extract channel name from URL
        self.channel = source.base_url.rstrip("/").split("/")[-1]

    async def fetch(self, since: Optional[datetime] = None) -> list[dict]:
        """Fetch latest messages from Telegram public channel."""
        articles = []
        preview_url = f"https://t.me/s/{self.channel}"

        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                resp = await client.get(
                    preview_url,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )
                resp.raise_for_status()

            articles = self._parse_telegram_html(resp.text)

        except Exception as e:
            logger.error("Telegram fetch failed for %s: %s", self.channel, e)

        logger.info("Telegram %s: fetched %d messages", self.channel, len(articles))
        return articles

    def _parse_telegram_html(self, html: str) -> list[dict]:
        """Parse Telegram web preview HTML for messages."""
        articles = []

        # Find message blocks
        message_blocks = re.findall(
            r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )

        # Find timestamps
        time_blocks = re.findall(
            r'datetime="([^"]+)"',
            html,
        )

        for i, block in enumerate(message_blocks[-20:]):  # Last 20
            text = re.sub(r"<[^>]+>", " ", block)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) < 20:
                continue

            published_at = datetime.utcnow()
            if i < len(time_blocks):
                try:
                    published_at = datetime.fromisoformat(
                        time_blocks[i].replace("+00:00", "")
                    )
                except (ValueError, IndexError):
                    pass

            articles.append({
                "title": text[:100],
                "body": text,
                "url": f"https://t.me/{self.channel}",
                "published_at": published_at,
            })

        return articles


def get_fetcher(source: NewsSource):
    """Factory: return appropriate fetcher for a source."""
    if source.name == "Institute for the Study of War":
        return ISWFetcher(source)
    if source.name == "DeepState Map":
        return DeepStateFetcher(source)
    if source.source_type.value == "telegram":
        return TelegramFetcher(source)
    return RSSFetcher(source)
