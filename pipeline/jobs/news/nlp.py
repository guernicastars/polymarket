"""NLP pipeline for news article analysis.

Extracts:
  - Sentiment (-1 to +1, from Ukraine perspective)
  - Settlement mentions (entity extraction)
  - Control change signals
  - Urgency score
  - Category classification

Uses keyword-based approach (no ML dependencies) for speed and reliability.
Can be upgraded to LLM-based extraction later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Optional

from .sources import (
    SETTLEMENT_ALIASES,
    CONTROL_KEYWORDS,
    SENTIMENT_KEYWORDS,
    SOURCES,
)

logger = logging.getLogger(__name__)


class ArticleAnalyzer:
    """Analyze a news article and extract structured signals."""

    def __init__(self):
        # Build reverse lookup: alias → settlement_id
        self._alias_map: dict[str, str] = {}
        for sid, aliases in SETTLEMENT_ALIASES.items():
            for alias in aliases:
                self._alias_map[alias.lower()] = sid

        # Compile settlement regex (longest match first)
        aliases_sorted = sorted(self._alias_map.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(a) for a in aliases_sorted)
        self._settlement_re = re.compile(pattern, re.IGNORECASE)

        # Category keywords
        self._category_keywords = {
            "frontline": ["frontline", "front line", "advance", "retreat", "assault", "attack", "defense", "battle", "fighting", "clashes"],
            "logistics": ["supply", "logistics", "ammunition", "weapons", "delivery", "shipment", "aid package", "convoy"],
            "politics": ["negotiation", "summit", "agreement", "ceasefire", "sanction", "diplomat", "treaty", "peace talks"],
            "weapons": ["missile", "drone", "artillery", "himars", "patriot", "f-16", "leopard", "abrams", "glide bomb", "shahed"],
            "casualties": ["casualty", "casualties", "killed", "wounded", "losses", "dead", "injured"],
            "negotiations": ["peace", "ceasefire", "talks", "agreement", "deal", "negotiat"],
        }

        # Urgency keywords
        self._urgency_keywords = {
            "breaking": 0.9,
            "urgent": 0.8,
            "just in": 0.8,
            "confirmed": 0.6,
            "developing": 0.7,
            "massive": 0.6,
            "critical": 0.7,
            "imminent": 0.8,
            "breakthrough": 0.7,
        }

    def analyze(
        self,
        title: str,
        body: str,
        source_name: str,
        published_at: Optional[datetime] = None,
        url: str = "",
    ) -> dict:
        """Analyze a news article and return structured data.

        Returns dict ready for ClickHouse insertion.
        """
        text = f"{title} {body}".lower()
        source_cfg = SOURCES.get(source_name)
        source_reliability = source_cfg.reliability if source_cfg else 0.5

        # Generate article ID
        article_id = hashlib.sha256(
            (url or f"{source_name}:{title}:{published_at}").encode()
        ).hexdigest()[:16]

        # Extract settlements
        settlements = self._extract_settlements(text)

        # Compute sentiment
        sentiment = self._compute_sentiment(text)

        # Detect control changes
        control_changes = self._detect_control_changes(text, settlements)

        # Classify category
        category = self._classify_category(text)

        # Detect region
        region = self._detect_region(settlements)

        # Compute urgency
        urgency = self._compute_urgency(text)

        # Map settlements to market condition_ids (if known)
        markets = []  # filled downstream by bridge

        return {
            "article_id": article_id,
            "source": source_name,
            "source_url": url,
            "title": title[:500],
            "body": body[:10000],
            "language": source_cfg.language if source_cfg else "en",
            "category": category,
            "region": region,
            "sentiment": round(sentiment, 3),
            "urgency": round(urgency, 3),
            "confidence": round(source_reliability, 3),
            "settlements_mentioned": settlements,
            "markets_mentioned": markets,
            "actors": self._extract_actors(text),
            "control_changes": json.dumps(control_changes),
            "published_at": published_at or datetime.utcnow(),
            "ingested_at": datetime.utcnow(),
        }

    def _extract_settlements(self, text: str) -> list[str]:
        """Find all settlement mentions in text."""
        found = set()
        for match in self._settlement_re.finditer(text):
            alias = match.group().lower()
            sid = self._alias_map.get(alias)
            if sid:
                found.add(sid)
        return sorted(found)

    def _compute_sentiment(self, text: str) -> float:
        """Keyword-based sentiment scoring.

        Returns -1 (bad for UA) to +1 (good for UA).
        """
        total = 0.0
        count = 0
        for keyword, score in SENTIMENT_KEYWORDS.items():
            occurrences = text.count(keyword.lower())
            if occurrences > 0:
                total += score * min(occurrences, 3)  # cap at 3x
                count += min(occurrences, 3)
        return total / max(count, 1)

    def _detect_control_changes(
        self, text: str, settlements: list[str]
    ) -> list[dict]:
        """Detect potential control changes from keywords + settlement context."""
        changes = []
        for keyword, (new_control, confidence) in CONTROL_KEYWORDS.items():
            if keyword.lower() in text:
                # Associate with mentioned settlements
                for sid in settlements:
                    changes.append({
                        "settlement_id": sid,
                        "signal": new_control,
                        "keyword": keyword,
                        "confidence": confidence,
                    })
        return changes

    def _classify_category(self, text: str) -> str:
        """Classify article into primary category."""
        scores = {}
        for category, keywords in self._category_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        return max(scores, key=scores.get) if scores else "general"

    def _detect_region(self, settlements: list[str]) -> str:
        """Infer region from mentioned settlements."""
        if not settlements:
            return "general"

        # Simple heuristic based on known settlements
        donbas = {"pokrovsk", "chasiv_yar", "toretsk", "kramatorsk", "sloviansk",
                  "bakhmut", "avdiivka", "vuhledar", "kurakhove", "selydove",
                  "myrnohrad", "hryshyne", "kostiantynivka", "druzhkivka",
                  "siversk", "lyman", "horlivka", "makiivka", "donetsk_city"}
        south = {"zaporizhzhia", "orikhiv", "melitopol", "tokmak", "mariupol",
                 "berdyansk", "kherson_city", "mykolaiv", "nova_kakhovka"}
        kursk = {"sudzha", "kursk_city"}
        crimea = {"crimea_simferopol", "crimea_sevastopol", "crimea_kerch"}

        settlement_set = set(settlements)
        if settlement_set & kursk:
            return "kursk"
        if settlement_set & crimea:
            return "crimea"
        if settlement_set & south:
            return "south"
        if settlement_set & donbas:
            return "donbas"
        return "general"

    def _compute_urgency(self, text: str) -> float:
        """Score urgency based on keywords."""
        max_urgency = 0.0
        for keyword, score in self._urgency_keywords.items():
            if keyword in text:
                max_urgency = max(max_urgency, score)
        return max_urgency

    def _extract_actors(self, text: str) -> list[str]:
        """Extract mentioned military/political actors."""
        actors = []
        actor_keywords = {
            "ua_army": ["ukrainian forces", "ukrainian army", "zsu", "armed forces of ukraine", "afu"],
            "ru_army": ["russian forces", "russian army", "russian troops"],
            "wagner": ["wagner", "pmc"],
            "azov": ["azov brigade", "azov regiment"],
            "marines": ["marines", "naval infantry"],
            "diplomats": ["zelensky", "zelenskyy", "putin", "lavrov", "nato", "biden", "trump"],
            "volunteers": ["volunteer", "territorial defense"],
        }
        for actor, keywords in actor_keywords.items():
            if any(kw in text for kw in keywords):
                actors.append(actor)
        return actors
