"""Schema migration — reads SQL files and runs them against ClickHouse."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.clickhouse_writer import ClickHouseWriter

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).parent / "schema"
SCHEMA_FILES = ["001_init.sql", "002_phase2_users.sql", "003_phase3_analytics.sql", "004_network.sql", "005_gnn_predictions.sql", "006_news_tracking.sql"]


def run_migration() -> None:
    """Execute all schema migrations against ClickHouse Cloud."""
    writer = ClickHouseWriter.get_instance()

    for schema_file in SCHEMA_FILES:
        schema_path = SCHEMA_DIR / schema_file
        if not schema_path.exists():
            logger.warning("migration_skip", extra={"file": schema_file, "reason": "not found"})
            continue

        sql = schema_path.read_text()
        writer.run_migration(sql)
        logger.info("migration_applied", extra={"file": schema_file})
