"""Entry point for the Polymarket data pipeline."""

from __future__ import annotations

import asyncio
import logging

from pipeline.config import setup_logging
from pipeline.migrate import run_migration
from pipeline.scheduler import PipelineScheduler

logger = logging.getLogger(__name__)


async def _main() -> None:
    setup_logging()
    logger.info("pipeline_starting")

    # Run schema migration before starting the scheduler
    try:
        run_migration()
    except Exception:
        logger.error("migration_failed", exc_info=True)
        raise

    scheduler = PipelineScheduler()
    await scheduler.start()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
