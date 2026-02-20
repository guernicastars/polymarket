"""Job: news tracking + market microstructure ingestion.

Phase 4 jobs:
- News tracker: multi-source OSINT ingestion with NLP analysis (every 10 min)
- Microstructure engine: tick-level market quality metrics (every 60 sec)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any
from urllib.parse import quote_plus

import httpx

import clickhouse_connect

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)

# --- Keyword-based sentiment scoring ---
_POSITIVE_WORDS = frozenset([
    "surge", "rally", "win", "gains", "positive", "bullish", "soar", "jump",
    "boost", "recover", "upbeat", "success", "leads", "ahead", "strong",
    "approval", "passed", "victory", "agreement", "deal", "rising", "higher",
    "outperform", "optimism", "confident", "breakthrough", "milestone",
])

_NEGATIVE_WORDS = frozenset([
    "crash", "fall", "loss", "losses", "negative", "bearish", "plunge", "drop",
    "decline", "downturn", "slump", "fear", "risk", "threat", "fail", "failed",
    "scandal", "crisis", "concern", "warning", "uncertainty", "reject", "rejected",
    "lower", "worst", "collapse", "deficit", "controversy", "investigation",
])

# Max markets to search for per cycle
_NEWS_MAX_MARKETS = 50
# Max articles per RSS search
_NEWS_MAX_ARTICLES_PER_SEARCH = 5


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    """Create a read-only ClickHouse client for queries."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


def _compute_sentiment(text: str) -> float:
    """Simple keyword-based sentiment score in [-1.0, 1.0]."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _compute_urgency(title: str) -> float:
    """Simple urgency heuristic from title keywords."""
    lower = title.lower()
    urgency = 0.0
    if any(w in lower for w in ("breaking", "just in", "urgent", "alert")):
        urgency += 0.5
    if any(w in lower for w in ("live", "now", "today", "minutes ago")):
        urgency += 0.3
    if "!" in title:
        urgency += 0.2
    return min(urgency, 1.0)


def _extract_search_terms(question: str) -> str:
    """Extract meaningful search terms from a market question."""
    # Remove common question prefixes
    q = re.sub(r"^(will|does|is|has|can|should|would)\s+", "", question, flags=re.IGNORECASE)
    # Remove question mark and trim
    q = q.rstrip("?").strip()
    # Limit to first ~60 chars to avoid overly specific queries
    if len(q) > 60:
        q = q[:60].rsplit(" ", 1)[0]
    return q


async def _fetch_google_news_rss(
    query: str,
    http_client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Fetch articles from Google News RSS for a search query."""
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = await http_client.get(url, timeout=15.0)
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        articles = []
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            source_el = item.find("source")

            if title_el is None or link_el is None:
                continue

            title = unescape(title_el.text or "").strip()
            link = (link_el.text or "").strip()
            source_name = (source_el.text if source_el is not None else "google_news").strip()

            # Parse published date
            published_at = datetime.now(timezone.utc)
            if pub_el is not None and pub_el.text:
                try:
                    published_at = parsedate_to_datetime(pub_el.text)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            articles.append({
                "title": title,
                "url": link,
                "source": source_name,
                "published_at": published_at,
            })

            if len(articles) >= _NEWS_MAX_ARTICLES_PER_SEARCH:
                break

        return articles
    except Exception as e:
        logger.debug("google_news_rss_error", extra={"query": query, "error": str(e)})
        return []


async def run_news_tracker() -> None:
    """Fetch news from Google News RSS, match to markets, score sentiment.

    Steps:
      1. Fetch top active markets from ClickHouse
      2. For each market, search Google News RSS for related articles
      3. Compute keyword sentiment and urgency per article
      4. Deduplicate by article_id (SHA256 of URL)
      5. Write to news_articles table
      6. Aggregate hourly sentiment per market and write to news_sentiment_hourly
    """
    from pipeline.clickhouse_writer import ClickHouseWriter

    writer = ClickHouseWriter.get_instance()
    client = await asyncio.to_thread(_get_read_client)
    now = datetime.now(timezone.utc)

    try:
        # 1. Fetch top markets by volume with their questions
        markets_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT condition_id, question, category
            FROM markets FINAL
            WHERE active = 1 AND closed = 0 AND question != ''
            ORDER BY volume_24h DESC
            LIMIT {_NEWS_MAX_MARKETS}
            """,
        )

        markets = [
            {"condition_id": row[0], "question": row[1], "category": row[2]}
            for row in markets_result.result_rows
        ]

        if not markets:
            logger.debug("news_tracker_skip", extra={"reason": "no_markets"})
            return

        # 2. Check existing article IDs to avoid duplicates
        existing_ids: set[str] = set()
        try:
            existing_result = await asyncio.to_thread(
                client.query,
                """
                SELECT article_id
                FROM news_articles
                WHERE published_at >= now() - INTERVAL 2 DAY
                """,
            )
            existing_ids = {row[0] for row in existing_result.result_rows}
        except Exception:
            pass  # Table may be empty or not yet created

        # 3. Fetch news for each market via Google News RSS
        article_rows: list[list] = []
        market_articles: dict[str, list[dict]] = defaultdict(list)
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; PolymarketSignals/1.0)"},
            follow_redirects=True,
        ) as http_client:
            for market in markets:
                search_terms = _extract_search_terms(market["question"])
                if len(search_terms) < 5:
                    continue

                articles = await _fetch_google_news_rss(search_terms, http_client)

                for article in articles:
                    article_id = hashlib.sha256(article["url"].encode()).hexdigest()[:32]

                    if article_id in seen_ids or article_id in existing_ids:
                        continue
                    seen_ids.add(article_id)

                    title = article["title"]
                    sentiment = _compute_sentiment(title)
                    urgency = _compute_urgency(title)

                    article_rows.append([
                        article_id,                          # article_id
                        article["source"],                   # source
                        article["url"],                      # source_url
                        title,                               # title
                        "",                                  # body (RSS only gives titles)
                        "en",                                # language
                        market["category"],                  # category
                        "",                                  # region
                        round(sentiment, 4),                 # sentiment
                        round(urgency, 4),                   # urgency
                        0.5,                                 # confidence (default for RSS)
                        [],                                  # settlements_mentioned
                        [market["condition_id"]],            # markets_mentioned
                        [],                                  # actors
                        "{}",                                # control_changes
                        article["published_at"],             # published_at
                        now,                                 # ingested_at
                    ])

                    market_articles[market["condition_id"]].append({
                        "sentiment": sentiment,
                        "urgency": urgency,
                        "source": article["source"],
                        "confidence": 0.5,
                    })

                # Rate limit: small delay between RSS fetches
                await asyncio.sleep(0.5)

        # 4. Write articles
        if article_rows:
            await writer.write("news_articles", article_rows)

        # 5. Aggregate hourly sentiment per market → news_sentiment_hourly
        # Use condition_id as settlement_id since these are market-level aggregates
        hour_bucket = now.replace(minute=0, second=0, microsecond=0)
        sentiment_rows: list[list] = []

        for cid, arts in market_articles.items():
            if not arts:
                continue
            count = len(arts)
            avg_sent = sum(a["sentiment"] for a in arts) / count
            max_urg = max(a["urgency"] for a in arts)
            sources = len(set(a["source"] for a in arts))
            weighted_sent = sum(a["sentiment"] * a["confidence"] for a in arts) / max(
                sum(a["confidence"] for a in arts), 1e-9
            )
            # News velocity: articles this hour (normalized later when more data builds up)
            velocity = float(count)

            sentiment_rows.append([
                cid,                           # settlement_id (using condition_id)
                hour_bucket,                   # hour
                count,                         # article_count
                round(avg_sent, 4),            # avg_sentiment
                round(max_urg, 4),             # max_urgency
                min(sources, 255),             # source_diversity (UInt8)
                round(weighted_sent, 4),       # weighted_sentiment
                round(velocity, 4),            # news_velocity
            ])

        if sentiment_rows:
            await writer.write("news_sentiment_hourly", sentiment_rows)

        await writer.flush_all()

        logger.info(
            "news_tracker_complete",
            extra={
                "total_articles": len(article_rows),
                "markets_with_news": len(market_articles),
                "sources": len(set(a["source"] for arts in market_articles.values() for a in arts)) if market_articles else 0,
            },
        )

    except Exception:
        logger.error("news_tracker_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def run_microstructure() -> None:
    """Compute market microstructure metrics for top markets.

    Reads orderbook_snapshots and market_trades from ClickHouse,
    computes spread, depth, trade flow, and price impact metrics,
    writes to market_microstructure table.
    """
    from pipeline.clickhouse_writer import ClickHouseWriter

    writer = ClickHouseWriter.get_instance()
    client = await asyncio.to_thread(_get_read_client)
    now = datetime.now(timezone.utc)

    try:
        # 1. Get top 100 active markets by volume
        markets_result = await asyncio.to_thread(
            client.query,
            """
            SELECT condition_id
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            ORDER BY volume_24h DESC
            LIMIT 100
            """,
        )
        market_ids = [row[0] for row in markets_result.result_rows]
        if not market_ids:
            logger.debug("microstructure_skip", extra={"reason": "no_markets"})
            return

        placeholders = ", ".join(f"'{c}'" for c in market_ids)

        # 2. Latest orderbook snapshots (spread + depth metrics)
        ob_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                os.condition_id,
                os.bid_prices,
                os.bid_sizes,
                os.ask_prices,
                os.ask_sizes
            FROM orderbook_snapshots os
            INNER JOIN (
                SELECT condition_id, max(snapshot_time) AS max_time
                FROM orderbook_snapshots
                WHERE condition_id IN ({placeholders})
                  AND snapshot_time >= now() - INTERVAL 5 MINUTE
                GROUP BY condition_id
            ) latest ON os.condition_id = latest.condition_id
              AND os.snapshot_time = latest.max_time
            """,
        )

        # Parse orderbook data per market
        ob_data: dict[str, dict] = {}
        for row in ob_result.result_rows:
            cid = row[0]
            bid_prices = row[1] or []
            bid_sizes = row[2] or []
            ask_prices = row[3] or []
            ask_sizes = row[4] or []

            best_bid = float(bid_prices[0]) if bid_prices else 0.0
            best_ask = float(ask_prices[0]) if ask_prices else 0.0
            spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0.0
            mid = (best_ask + best_bid) / 2 if best_ask > 0 and best_bid > 0 else 0.0

            total_bid = sum(float(s) for s in bid_sizes)
            total_ask = sum(float(s) for s in ask_sizes)
            bid_depth_1 = float(bid_sizes[0]) if bid_sizes else 0.0
            ask_depth_1 = float(ask_sizes[0]) if ask_sizes else 0.0
            bid_depth_5 = sum(float(s) for s in bid_sizes[:5])
            ask_depth_5 = sum(float(s) for s in ask_sizes[:5])

            obi = total_bid / max(total_bid + total_ask, 1e-9)
            depth_ratio = (bid_depth_5 + ask_depth_5) / max(bid_depth_1 + ask_depth_1, 1e-9) if (bid_depth_1 + ask_depth_1) > 0 else 0.0

            ob_data[cid] = {
                "spread": spread,
                "mid": mid,
                "bid_depth_1": bid_depth_1,
                "ask_depth_1": ask_depth_1,
                "bid_depth_5": bid_depth_5,
                "ask_depth_5": ask_depth_5,
                "obi": obi,
                "depth_ratio": depth_ratio,
            }

        # 3. Trade flow metrics (5-minute window)
        trade_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                condition_id,
                sumIf(size, side = 'buy') AS buy_vol,
                sumIf(size, side = 'sell') AS sell_vol,
                count() AS trade_count,
                countIf(price * size >= 1000) AS large_count,
                sum(price * size) / greatest(sum(size), 1e-9) AS vwap
            FROM market_trades
            WHERE condition_id IN ({placeholders})
              AND timestamp >= now() - INTERVAL 5 MINUTE
            GROUP BY condition_id
            """,
        )

        trade_data: dict[str, dict] = {}
        for row in trade_result.result_rows:
            cid = row[0]
            buy_vol = float(row[1])
            sell_vol = float(row[2])
            trade_count = int(row[3])
            large_count = int(row[4])
            vwap = float(row[5])
            trade_data[cid] = {
                "buy_volume_5m": buy_vol,
                "sell_volume_5m": sell_vol,
                "trade_count_5m": trade_count,
                "large_trade_count_5m": large_count,
                "vwap_5m": vwap,
            }

        # 4. Kyle's lambda (price impact per dollar traded)
        # Approximation: regress |price change| on trade size over recent trades
        kyle_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                condition_id,
                avg(abs_impact / greatest(usd_size, 1e-9)) AS kyle_lambda
            FROM (
                SELECT
                    condition_id,
                    price * size AS usd_size,
                    abs(price - neighbor(price, -1)) AS abs_impact
                FROM market_trades
                WHERE condition_id IN ({placeholders})
                  AND timestamp >= now() - INTERVAL 30 MINUTE
                ORDER BY condition_id, timestamp
            )
            WHERE usd_size > 0 AND abs_impact > 0
            GROUP BY condition_id
            """,
        )

        kyle_data: dict[str, float] = {}
        for row in kyle_result.result_rows:
            kyle_data[row[0]] = float(row[1])

        # 5. Price impact 1 minute after trade (adverse selection)
        impact_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                t1.condition_id,
                avg(abs(t2.price - t1.price)) AS avg_impact
            FROM market_trades t1
            INNER JOIN market_trades t2
              ON t1.condition_id = t2.condition_id
              AND t2.timestamp > t1.timestamp
              AND t2.timestamp <= t1.timestamp + INTERVAL 1 MINUTE
            WHERE t1.condition_id IN ({placeholders})
              AND t1.timestamp >= now() - INTERVAL 15 MINUTE
              AND t1.price * t1.size >= 500
            GROUP BY t1.condition_id
            """,
        )

        impact_data: dict[str, float] = {}
        for row in impact_result.result_rows:
            impact_data[row[0]] = float(row[1])

        # 6. Assemble rows
        micro_rows: list[list] = []
        for cid in market_ids:
            ob = ob_data.get(cid, {})
            td = trade_data.get(cid, {})

            spread = ob.get("spread", 0.0)
            mid = ob.get("mid", 0.0)
            buy_vol = td.get("buy_volume_5m", 0.0)
            sell_vol = td.get("sell_volume_5m", 0.0)
            total_vol = buy_vol + sell_vol

            # Effective spread: approximate from actual trade VWAP vs mid
            effective_spread = abs(td.get("vwap_5m", 0.0) - mid) * 2 if mid > 0 else 0.0

            # Realized spread: spread minus price impact
            price_impact = impact_data.get(cid, 0.0)
            realized_spread = max(spread - price_impact, 0.0)

            # Toxic flow ratio: buy/sell imbalance as proxy for informed trading
            toxic_ratio = abs(buy_vol - sell_vol) / max(total_vol, 1e-9) if total_vol > 0 else 0.0

            micro_rows.append([
                cid,                                    # condition_id
                round(spread, 8),                       # bid_ask_spread
                round(effective_spread, 8),             # effective_spread
                round(realized_spread, 8),              # realized_spread
                round(ob.get("bid_depth_1", 0.0), 4),  # bid_depth_1
                round(ob.get("ask_depth_1", 0.0), 4),  # ask_depth_1
                round(ob.get("bid_depth_5", 0.0), 4),  # bid_depth_5
                round(ob.get("ask_depth_5", 0.0), 4),  # ask_depth_5
                round(ob.get("obi", 0.0), 6),          # obi
                round(ob.get("depth_ratio", 0.0), 4),  # depth_ratio
                round(buy_vol, 4),                      # buy_volume_5m
                round(sell_vol, 4),                     # sell_volume_5m
                td.get("trade_count_5m", 0),            # trade_count_5m
                td.get("large_trade_count_5m", 0),      # large_trade_count_5m
                round(td.get("vwap_5m", 0.0), 6),      # vwap_5m
                round(kyle_data.get(cid, 0.0), 8),     # kyle_lambda
                round(toxic_ratio, 6),                  # toxic_flow_ratio
                round(price_impact, 8),                 # price_impact_1m
                round(spread, 8),                       # spread_after_trade (approx: current spread)
                0.0,                                    # depth_recovery_sec (not computable from snapshots alone)
                now,                                    # snapshot_time
            ])

        if micro_rows:
            await writer.write("market_microstructure", micro_rows)

        await writer.flush_all()

        logger.info(
            "microstructure_complete",
            extra={"snapshots": len(micro_rows)},
        )

    except Exception:
        logger.error("microstructure_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass
