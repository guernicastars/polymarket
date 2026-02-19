"""ClickHouse bridge — reads live market data, writes network model outputs.

Connects the network settlement graph model to the ClickHouse pipeline:
  - Reads: market_prices, markets, orderbook_snapshots, frontline_state,
           news_sentiment_hourly, composite_signals, market_microstructure
  - Writes: network_vulnerability, network_supply_risk,
            network_cascade_scenarios, network_signals
  - Resolves: polymarket_mapping condition_ids from market slugs

Also provides a BatchedWriter adapter for the microstructure and news tracker
jobs that use dict-based `.add(table, row_dict)` interface.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import clickhouse_connect
from clickhouse_connect.driver.client import Client

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class ClickHouseBridge:
    """Bridge between network model and ClickHouse pipeline data.

    Provides read access to live market/frontline data and write access
    to network model output tables.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8443,
        user: str = "default",
        password: str = "",
        database: str = "polymarket",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database=self.database,
                secure=True,
                compress="lz4",
                connect_timeout=30,
                send_receive_timeout=120,
            )
        return self._client

    def close(self) -> None:
        """Close the underlying connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Market price reads
    # ------------------------------------------------------------------

    def get_market_price(self, condition_id: str) -> Optional[float]:
        """Get latest price for a market from ClickHouse."""
        client = self._get_client()
        rows = client.query("""
            SELECT price
            FROM market_prices
            WHERE condition_id = {cid:String}
              AND outcome = 'Yes'
            ORDER BY timestamp DESC
            LIMIT 1
        """, parameters={"cid": condition_id}).result_rows
        return rows[0][0] if rows else None

    def get_market_prices_bulk(self, condition_ids: list[str]) -> dict[str, float]:
        """Get latest prices for multiple markets."""
        if not condition_ids:
            return {}
        client = self._get_client()
        rows = client.query("""
            SELECT condition_id, argMax(price, timestamp) AS latest_price
            FROM market_prices
            WHERE condition_id IN {cids:Array(String)}
              AND outcome = 'Yes'
              AND timestamp >= now() - INTERVAL 1 DAY
            GROUP BY condition_id
        """, parameters={"cids": condition_ids}).result_rows
        return {r[0]: r[1] for r in rows}

    def get_market_metadata(self, condition_id: str) -> Optional[dict]:
        """Get market metadata (slug, event_slug, end_date, volume, etc.)."""
        client = self._get_client()
        rows = client.query("""
            SELECT
                market_slug, event_slug, end_date, volume_24h, liquidity,
                one_day_price_change, outcomes, outcome_prices, token_ids
            FROM markets FINAL
            WHERE condition_id = {cid:String}
            LIMIT 1
        """, parameters={"cid": condition_id}).result_rows
        if not rows:
            return None
        r = rows[0]
        return {
            "market_slug": r[0],
            "event_slug": r[1],
            "end_date": r[2],
            "volume_24h": r[3],
            "liquidity": r[4],
            "one_day_price_change": r[5],
            "outcomes": r[6],
            "outcome_prices": r[7],
            "token_ids": r[8],
        }

    # ------------------------------------------------------------------
    # Frontline / OSINT reads
    # ------------------------------------------------------------------

    def get_latest_frontline_state(
        self, settlement_id: str
    ) -> Optional[dict]:
        """Get latest frontline state for a settlement."""
        client = self._get_client()
        rows = client.query("""
            SELECT control, assault_intensity, shelling_intensity,
                   supply_disruption, frontline_distance_km, source,
                   confidence, observed_at
            FROM frontline_state
            WHERE settlement_id = {sid:String}
            ORDER BY observed_at DESC
            LIMIT 1
        """, parameters={"sid": settlement_id}).result_rows
        if not rows:
            return None
        r = rows[0]
        return {
            "control": r[0],
            "assault_intensity": r[1],
            "shelling_intensity": r[2],
            "supply_disruption": r[3],
            "frontline_distance_km": r[4],
            "source": r[5],
            "confidence": r[6],
            "observed_at": r[7],
        }

    def get_frontline_bulk(self, settlement_ids: list[str]) -> dict[str, dict]:
        """Get latest frontline state for multiple settlements."""
        if not settlement_ids:
            return {}
        client = self._get_client()
        rows = client.query("""
            SELECT settlement_id,
                   argMax(control, observed_at) AS control,
                   argMax(assault_intensity, observed_at) AS assault,
                   argMax(shelling_intensity, observed_at) AS shelling,
                   argMax(supply_disruption, observed_at) AS supply,
                   argMax(frontline_distance_km, observed_at) AS distance,
                   argMax(confidence, observed_at) AS conf,
                   max(observed_at) AS last_seen
            FROM frontline_state
            WHERE settlement_id IN {sids:Array(String)}
              AND observed_at >= now() - INTERVAL 7 DAY
            GROUP BY settlement_id
        """, parameters={"sids": settlement_ids}).result_rows
        result = {}
        for r in rows:
            result[r[0]] = {
                "control": r[1],
                "assault_intensity": r[2],
                "shelling_intensity": r[3],
                "supply_disruption": r[4],
                "frontline_distance_km": r[5],
                "confidence": r[6],
                "observed_at": r[7],
            }
        return result

    # ------------------------------------------------------------------
    # News sentiment reads
    # ------------------------------------------------------------------

    def get_settlement_sentiment(
        self, settlement_id: str, hours: int = 24
    ) -> Optional[dict]:
        """Get aggregated news sentiment for a settlement over recent hours."""
        client = self._get_client()
        rows = client.query("""
            SELECT
                sum(article_count) AS total_articles,
                avg(avg_sentiment) AS mean_sentiment,
                max(max_urgency) AS peak_urgency,
                avg(weighted_sentiment) AS weighted_sentiment,
                avg(news_velocity) AS avg_velocity
            FROM news_sentiment_hourly
            WHERE settlement_id = {sid:String}
              AND hour >= now() - INTERVAL {hours:UInt32} HOUR
        """, parameters={"sid": settlement_id, "hours": hours}).result_rows
        if not rows or rows[0][0] is None or rows[0][0] == 0:
            return None
        r = rows[0]
        return {
            "total_articles": r[0],
            "mean_sentiment": r[1],
            "peak_urgency": r[2],
            "weighted_sentiment": r[3],
            "avg_velocity": r[4],
        }

    def get_sentiment_bulk(
        self, settlement_ids: list[str], hours: int = 24
    ) -> dict[str, dict]:
        """Get news sentiment for multiple settlements."""
        if not settlement_ids:
            return {}
        client = self._get_client()
        rows = client.query("""
            SELECT
                settlement_id,
                sum(article_count) AS total_articles,
                avg(avg_sentiment) AS mean_sentiment,
                max(max_urgency) AS peak_urgency,
                avg(weighted_sentiment) AS weighted_sentiment,
                avg(news_velocity) AS avg_velocity
            FROM news_sentiment_hourly
            WHERE settlement_id IN {sids:Array(String)}
              AND hour >= now() - INTERVAL {hours:UInt32} HOUR
            GROUP BY settlement_id
        """, parameters={"sids": settlement_ids, "hours": hours}).result_rows
        result = {}
        for r in rows:
            result[r[0]] = {
                "total_articles": r[1],
                "mean_sentiment": r[2],
                "peak_urgency": r[3],
                "weighted_sentiment": r[4],
                "avg_velocity": r[5],
            }
        return result

    # ------------------------------------------------------------------
    # Microstructure reads
    # ------------------------------------------------------------------

    def get_latest_microstructure(self, condition_id: str) -> Optional[dict]:
        """Get latest microstructure snapshot for a market."""
        client = self._get_client()
        rows = client.query("""
            SELECT
                bid_ask_spread, effective_spread, realized_spread,
                obi, depth_ratio, kyle_lambda, toxic_flow_ratio,
                price_impact_1m, buy_volume_5m, sell_volume_5m,
                trade_count_5m, large_trade_count_5m, vwap_5m,
                snapshot_time
            FROM market_microstructure
            WHERE condition_id = {cid:String}
            ORDER BY snapshot_time DESC
            LIMIT 1
        """, parameters={"cid": condition_id}).result_rows
        if not rows:
            return None
        r = rows[0]
        return {
            "bid_ask_spread": r[0],
            "effective_spread": r[1],
            "realized_spread": r[2],
            "obi": r[3],
            "depth_ratio": r[4],
            "kyle_lambda": r[5],
            "toxic_flow_ratio": r[6],
            "price_impact_1m": r[7],
            "buy_volume_5m": r[8],
            "sell_volume_5m": r[9],
            "trade_count_5m": r[10],
            "large_trade_count_5m": r[11],
            "vwap_5m": r[12],
            "snapshot_time": r[13],
        }

    # ------------------------------------------------------------------
    # Market resolution / mapping
    # ------------------------------------------------------------------

    def resolve_market_ids(self, slugs: list[str]) -> dict[str, dict]:
        """Look up condition_id and token_ids for market slugs.

        Used to populate polymarket_mapping.json with real IDs.
        """
        if not slugs:
            return {}
        client = self._get_client()
        rows = client.query("""
            SELECT market_slug, condition_id, token_ids
            FROM markets FINAL
            WHERE market_slug IN {slugs:Array(String)}
              AND active = 1
        """, parameters={"slugs": slugs}).result_rows
        result = {}
        for r in rows:
            result[r[0]] = {
                "condition_id": r[1],
                "token_ids": r[2],
            }
        return result

    def auto_resolve_mapping(self) -> int:
        """Auto-populate polymarket_mapping.json from ClickHouse markets table.

        Returns number of mappings resolved.
        """
        mapping_path = DATA_DIR / "polymarket_mapping.json"
        if not mapping_path.exists():
            logger.warning("polymarket_mapping.json not found at %s", mapping_path)
            return 0

        with open(mapping_path) as f:
            mapping = json.load(f)

        # Collect slugs that need resolution
        slugs_needed = []
        for entry in mapping:
            if not entry.get("condition_id") and entry.get("market_slug"):
                slugs_needed.append(entry["market_slug"])

        if not slugs_needed:
            logger.info("All mappings already resolved")
            return 0

        resolved = self.resolve_market_ids(slugs_needed)

        count = 0
        for entry in mapping:
            slug = entry.get("market_slug", "")
            if slug in resolved:
                entry["condition_id"] = resolved[slug]["condition_id"]
                entry["token_ids"] = resolved[slug]["token_ids"]
                count += 1
                logger.info("Resolved %s → %s", slug, resolved[slug]["condition_id"])

        # Write back
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)

        logger.info("Auto-resolved %d/%d market mappings", count, len(slugs_needed))
        return count

    # ------------------------------------------------------------------
    # Network model writes
    # ------------------------------------------------------------------

    def write_vulnerability_scores(self, scores: list[dict]) -> None:
        """Write vulnerability scores to network_vulnerability table."""
        if not scores:
            return
        client = self._get_client()
        columns = [
            "settlement_id", "connectivity_score", "supply_score",
            "force_balance_score", "terrain_score", "fortification_score",
            "assault_intensity_score", "frontline_score", "composite",
            "computed_at",
        ]
        now = datetime.utcnow()
        rows = []
        for s in scores:
            rows.append([
                s["settlement_id"],
                s.get("connectivity_score", 0.0),
                s.get("supply_score", 0.0),
                s.get("force_balance_score", 0.0),
                s.get("terrain_score", 0.0),
                s.get("fortification_score", 0.0),
                s.get("assault_intensity_score", 0.0),
                s.get("frontline_score", 0.0),
                s.get("composite", 0.0),
                s.get("computed_at", now),
            ])
        client.insert("network_vulnerability", rows, column_names=columns)
        logger.info("Wrote %d vulnerability scores", len(rows))

    def write_supply_risk(self, risks: list[dict]) -> None:
        """Write supply chain risk to network_supply_risk table."""
        if not risks:
            return
        client = self._get_client()
        columns = [
            "settlement_id", "origin", "shortest_cost", "path_redundancy",
            "min_cut_size", "min_cut_nodes", "supply_risk", "computed_at",
        ]
        now = datetime.utcnow()
        rows = []
        for r in risks:
            rows.append([
                r["settlement_id"],
                r.get("origin", "dnipro"),
                r.get("shortest_cost", 0.0),
                r.get("path_redundancy", 0),
                r.get("min_cut_size", 0),
                r.get("min_cut_nodes", []),
                r.get("supply_risk", 0.0),
                r.get("computed_at", now),
            ])
        client.insert("network_supply_risk", rows, column_names=columns)
        logger.info("Wrote %d supply risk scores", len(rows))

    def write_cascade_scenarios(self, scenarios: list[dict]) -> None:
        """Write cascade simulation results to network_cascade_scenarios table."""
        if not scenarios:
            return
        client = self._get_client()
        columns = [
            "trigger_node", "fallen_nodes", "isolated_nodes",
            "supply_cut_nodes", "new_component_count", "severity",
            "computed_at",
        ]
        now = datetime.utcnow()
        rows = []
        for s in scenarios:
            rows.append([
                s["trigger_node"],
                s.get("fallen_nodes", []),
                s.get("isolated_nodes", []),
                s.get("supply_cut_nodes", []),
                s.get("new_component_count", 0),
                s.get("severity", 0.0),
                s.get("computed_at", now),
            ])
        client.insert("network_cascade_scenarios", rows, column_names=columns)
        logger.info("Wrote %d cascade scenarios", len(rows))

    def write_signals(self, signals: list[dict]) -> None:
        """Write trading signals to network_signals table."""
        if not signals:
            return
        client = self._get_client()
        columns = [
            "settlement_id", "market_slug", "model_probability",
            "market_probability", "edge", "direction",
            "kelly_fraction", "confidence", "computed_at",
        ]
        now = datetime.utcnow()
        rows = []
        for s in signals:
            rows.append([
                s["settlement_id"],
                s.get("market_slug", ""),
                s.get("model_probability", 0.0),
                s.get("market_probability", 0.0),
                s.get("edge", 0.0),
                s.get("direction", "HOLD"),
                s.get("kelly_fraction", 0.0),
                s.get("confidence", 0.0),
                s.get("computed_at", now),
            ])
        client.insert("network_signals", rows, column_names=columns)
        logger.info("Wrote %d trading signals", len(rows))

    # ------------------------------------------------------------------
    # Static data → ClickHouse (settlements, edges, timeline)
    # ------------------------------------------------------------------

    def ingest_static_data(self) -> dict:
        """Load static JSON files into ClickHouse tables.

        Per user request: store all data in ClickHouse.
        Creates tables if needed and inserts settlement nodes, edges, and timeline.
        """
        client = self._get_client()
        summary = {}

        # Create tables for static data
        client.command("""
            CREATE TABLE IF NOT EXISTS network_settlements (
                settlement_id     String,
                name              String,
                lat               Float64,
                lon               Float64,
                region            LowCardinality(String) DEFAULT '',
                control           LowCardinality(String) DEFAULT 'UA',
                type              LowCardinality(String) DEFAULT 'town',
                population        UInt32 DEFAULT 0,
                fortification     Float32 DEFAULT 0.5,
                supply_hub        UInt8 DEFAULT 0,
                pm_target         UInt8 DEFAULT 0,
                updated_at        DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY settlement_id
        """)

        client.command("""
            CREATE TABLE IF NOT EXISTS network_edges (
                source            String,
                target            String,
                edge_type         LowCardinality(String),
                weight            Float64 DEFAULT 1.0,
                capacity          Float64 DEFAULT 100.0,
                bidirectional     UInt8 DEFAULT 1,
                updated_at        DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY (source, target, edge_type)
        """)

        client.command("""
            CREATE TABLE IF NOT EXISTS network_timeline (
                event_date        Date,
                label             String,
                settlement_id     String,
                control           LowCardinality(String),
                assault_intensity Float32 DEFAULT 0,
                note              String DEFAULT ''
            ) ENGINE = MergeTree()
            ORDER BY (event_date, settlement_id)
        """)

        # Load settlements
        settlements_path = DATA_DIR / "settlements_ukraine.json"
        if settlements_path.exists():
            with open(settlements_path) as f:
                settlements = json.load(f)
            rows = []
            for s in settlements:
                rows.append([
                    s["id"], s["name"], s["lat"], s["lon"],
                    s.get("region", ""), s.get("control", "UA"),
                    s.get("type", "town"), s.get("population", 0),
                    s.get("fortification", 0.5),
                    1 if s.get("supply_hub") else 0,
                    1 if s.get("pm_target") else 0,
                ])
            cols = [
                "settlement_id", "name", "lat", "lon", "region", "control",
                "type", "population", "fortification", "supply_hub", "pm_target",
            ]
            client.insert("network_settlements", rows, column_names=cols)
            summary["settlements"] = len(rows)
            logger.info("Ingested %d settlements into ClickHouse", len(rows))

        # Load edges
        edges_path = DATA_DIR / "edges_ukraine.json"
        if edges_path.exists():
            with open(edges_path) as f:
                edges = json.load(f)
            rows = []
            for e in edges:
                rows.append([
                    e["source"], e["target"], e.get("type", "highway"),
                    e.get("weight", 1.0), e.get("capacity", 100.0),
                    1 if e.get("bidirectional", True) else 0,
                ])
            cols = ["source", "target", "edge_type", "weight", "capacity", "bidirectional"]
            client.insert("network_edges", rows, column_names=cols)
            summary["edges"] = len(rows)
            logger.info("Ingested %d edges into ClickHouse", len(rows))

        # Load timeline
        timeline_path = DATA_DIR / "timeline.json"
        if timeline_path.exists():
            with open(timeline_path) as f:
                timeline = json.load(f)
            rows = []
            for event in timeline:
                for change in event.get("changes", []):
                    rows.append([
                        event["date"], event["label"],
                        change["id"], change.get("control", "CONTESTED"),
                        change.get("assault_intensity", 0.0),
                        change.get("note", ""),
                    ])
            cols = ["event_date", "label", "settlement_id", "control",
                    "assault_intensity", "note"]
            client.insert("network_timeline", rows, column_names=cols)
            summary["timeline_events"] = len(rows)
            logger.info("Ingested %d timeline records into ClickHouse", len(rows))

        return summary


class BatchedWriter:
    """Dict-based writer adapter for news tracker and microstructure engine.

    Accepts `.add(table, row_dict)` calls, buffers them, and flushes
    via the pipeline ClickHouseWriter when threshold is met.
    """

    def __init__(self, ch_client: Client):
        self._client = ch_client
        self._buffers: dict[str, list[dict]] = {}
        self._flush_size = 500

    def add(self, table: str, row: dict) -> None:
        """Add a row dict to the buffer for a table."""
        if table not in self._buffers:
            self._buffers[table] = []
        self._buffers[table].append(row)

        if len(self._buffers[table]) >= self._flush_size:
            self.flush(table)

    def flush(self, table: Optional[str] = None) -> None:
        """Flush buffered rows to ClickHouse."""
        tables = [table] if table else list(self._buffers.keys())
        for t in tables:
            rows = self._buffers.get(t, [])
            if not rows:
                continue
            try:
                columns = list(rows[0].keys())
                data = [[r.get(c) for c in columns] for r in rows]
                self._client.insert(t, data, column_names=columns)
                logger.info("BatchedWriter flushed %d rows to %s", len(rows), t)
            except Exception as e:
                logger.error("BatchedWriter flush failed for %s: %s", t, e)
            finally:
                self._buffers[t] = []

    def flush_all(self) -> None:
        """Flush all tables."""
        self.flush()
