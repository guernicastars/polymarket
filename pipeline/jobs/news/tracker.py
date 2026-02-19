"""News tracker orchestrator — fetches, analyzes, and writes to ClickHouse.

Scheduled as a pipeline job: fetches all enabled sources, runs NLP,
writes articles + frontline state updates + sentiment aggregates.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .sources import SOURCES, NewsSource
from .fetchers import get_fetcher
from .nlp import ArticleAnalyzer

logger = logging.getLogger(__name__)


class NewsTracker:
    """Orchestrates multi-source news ingestion and analysis."""

    def __init__(self, writer, settlement_market_map: Optional[dict] = None):
        """
        Args:
            writer: ClickHouse batched writer (pipeline.clickhouse_writer.BatchedWriter)
            settlement_market_map: {settlement_id: condition_id} for market linking
        """
        self.writer = writer
        self.analyzer = ArticleAnalyzer()
        self.settlement_market_map = settlement_market_map or {}
        self._seen_ids: set[str] = set()  # dedup within session
        self._last_fetch: dict[str, datetime] = {}

    async def run_all(self) -> dict:
        """Fetch from all enabled sources, analyze, and write.

        Returns:
            summary dict with counts per source
        """
        summary = {}
        tasks = []

        for name, source in SOURCES.items():
            if not source.enabled:
                continue
            tasks.append(self._fetch_source(name, source))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(
            [n for n, s in SOURCES.items() if s.enabled],
            results,
        ):
            if isinstance(result, Exception):
                logger.error("Source %s failed: %s", name, result)
                summary[name] = {"error": str(result)}
            else:
                summary[name] = result

        total = sum(
            r.get("articles_written", 0)
            for r in summary.values()
            if isinstance(r, dict) and "articles_written" in r
        )
        logger.info("News tracker complete: %d total articles across %d sources", total, len(summary))
        return summary

    async def run_source(self, source_name: str) -> dict:
        """Fetch from a single source."""
        source = SOURCES.get(source_name)
        if not source:
            return {"error": f"Unknown source: {source_name}"}
        return await self._fetch_source(source_name, source)

    async def _fetch_source(self, name: str, source: NewsSource) -> dict:
        """Fetch, analyze, and write articles from one source."""
        since = self._last_fetch.get(name)
        if since is None:
            since = datetime.utcnow() - timedelta(hours=6)

        fetcher = get_fetcher(source)
        raw_articles = await fetcher.fetch(since=since)

        analyzed = []
        for raw in raw_articles:
            article = self.analyzer.analyze(
                title=raw["title"],
                body=raw["body"],
                source_name=name,
                published_at=raw.get("published_at"),
                url=raw.get("url", ""),
            )

            # Dedup
            if article["article_id"] in self._seen_ids:
                continue
            self._seen_ids.add(article["article_id"])

            # Link settlements to markets
            for sid in article["settlements_mentioned"]:
                cid = self.settlement_market_map.get(sid)
                if cid and cid not in article["markets_mentioned"]:
                    article["markets_mentioned"].append(cid)

            analyzed.append(article)

        # Write to ClickHouse
        articles_written = 0
        frontline_updates = 0
        sentiment_updates = 0

        for article in analyzed:
            try:
                self._write_article(article)
                articles_written += 1

                # Write frontline state updates if control changes detected
                import json
                changes = json.loads(article["control_changes"])
                for change in changes:
                    self._write_frontline_update(
                        change, article["source"], article["published_at"]
                    )
                    frontline_updates += 1

                # Update hourly sentiment aggregate
                for sid in article["settlements_mentioned"]:
                    self._update_sentiment_aggregate(
                        sid, article["published_at"], article["sentiment"],
                        article["urgency"], article["confidence"],
                    )
                    sentiment_updates += 1

            except Exception as e:
                logger.error("Write failed for article %s: %s", article["article_id"], e)

        self._last_fetch[name] = datetime.utcnow()

        return {
            "articles_fetched": len(raw_articles),
            "articles_written": articles_written,
            "frontline_updates": frontline_updates,
            "sentiment_updates": sentiment_updates,
        }

    def _write_article(self, article: dict) -> None:
        """Write article to news_articles table."""
        self.writer.add("news_articles", {
            "article_id": article["article_id"],
            "source": article["source"],
            "source_url": article["source_url"],
            "title": article["title"],
            "body": article["body"],
            "language": article["language"],
            "category": article["category"],
            "region": article["region"],
            "sentiment": article["sentiment"],
            "urgency": article["urgency"],
            "confidence": article["confidence"],
            "settlements_mentioned": article["settlements_mentioned"],
            "markets_mentioned": article["markets_mentioned"],
            "actors": article["actors"],
            "control_changes": article["control_changes"],
            "published_at": article["published_at"],
            "ingested_at": article["ingested_at"],
        })

    def _write_frontline_update(
        self, change: dict, source: str, observed_at: datetime
    ) -> None:
        """Write a frontline state update."""
        signal = change.get("signal", "")
        control_map = {
            "RU": "RU",
            "UA": "UA",
            "CONTESTED": "CONTESTED",
            "RU_ADVANCE": "CONTESTED",
            "UA_ADVANCE": "CONTESTED",
            "UA_LOST": "RU",
        }
        control = control_map.get(signal, "CONTESTED")

        self.writer.add("frontline_state", {
            "settlement_id": change["settlement_id"],
            "control": control,
            "assault_intensity": 0.5 if signal in ("RU_ADVANCE", "CONTESTED") else 0.0,
            "shelling_intensity": 0.0,
            "supply_disruption": 0.0,
            "frontline_distance_km": 0.0 if control in ("CONTESTED", "RU") else 50.0,
            "source": source,
            "confidence": change.get("confidence", 0.5),
            "observed_at": observed_at,
        })

    def _update_sentiment_aggregate(
        self,
        settlement_id: str,
        published_at: datetime,
        sentiment: float,
        urgency: float,
        confidence: float,
    ) -> None:
        """Update hourly sentiment aggregate."""
        hour = published_at.replace(minute=0, second=0, microsecond=0)
        self.writer.add("news_sentiment_hourly", {
            "settlement_id": settlement_id,
            "hour": hour,
            "article_count": 1,
            "avg_sentiment": sentiment,
            "max_urgency": urgency,
            "source_diversity": 1,
            "weighted_sentiment": sentiment * confidence,
            "news_velocity": 1.0,
        })
