"""Schema migration — reads the SQL file and runs it against ClickHouse."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.clickhouse_writer import ClickHouseWriter

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).parent / "schema" / "001_init.sql"


def run_migration() -> None:
    """Execute the full schema migration against ClickHouse Cloud."""
    writer = ClickHouseWriter.get_instance()

    if not SCHEMA_FILE.exists():
        logger.warning("migration_skip", extra={"reason": "no schema file"})
        return

    sql = SCHEMA_FILE.read_text()
    writer.run_migration(sql)
    logger.info("migration_applied", extra={"file": str(SCHEMA_FILE)})
