"""Job: enrich discovered wallets with profile data from Gamma API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import PROFILE_BATCH_SIZE
from pipeline.jobs.leaderboard_sync import discovered_wallets

logger = logging.getLogger(__name__)

# Track which wallets have been enriched (avoid re-fetching)
_enriched_wallets: set[str] = set()


async def run_profile_enricher() -> None:
    """Fetch profiles for newly discovered wallets (not yet enriched).

    Processes up to PROFILE_BATCH_SIZE wallets per cycle.
    """
    # Find wallets needing enrichment
    pending = discovered_wallets - _enriched_wallets
    if not pending:
        logger.debug("profile_enricher_skip", extra={"reason": "no_new_wallets"})
        return

    batch = list(pending)[:PROFILE_BATCH_SIZE]
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        rows = []
        enriched_count = 0

        for wallet in batch:
            profile = await client.fetch_public_profile(wallet)

            if profile:
                parsed = DataClient.parse_profile(profile, discovered_via="leaderboard")
                rows.append([
                    parsed["proxy_wallet"] or wallet,
                    parsed["pseudonym"],
                    parsed["name"],
                    parsed["bio"],
                    parsed["profile_image"],
                    parsed["x_username"],
                    parsed["verified_badge"],
                    parsed["display_username_public"],
                    parsed["profile_created_at"],
                    parsed["discovered_via"],
                    now,  # first_seen_at
                    now,  # updated_at
                ])
                enriched_count += 1
            else:
                # Store a minimal profile even if not found (so we don't retry)
                rows.append([
                    wallet,
                    "",   # pseudonym
                    "",   # name
                    "",   # bio
                    "",   # profile_image
                    "",   # x_username
                    0,    # verified_badge
                    0,    # display_username_public
                    datetime(1970, 1, 1, tzinfo=timezone.utc),  # profile_created_at
                    "leaderboard",  # discovered_via
                    now,  # first_seen_at
                    now,  # updated_at
                ])

            _enriched_wallets.add(wallet)
            # Rate limiting
            await asyncio.sleep(0.3)

        if rows:
            await writer.write_profiles(rows)
            await writer.flush_all()

        logger.info(
            "profile_enricher_complete",
            extra={
                "batch_size": len(batch),
                "enriched": enriched_count,
                "total_enriched": len(_enriched_wallets),
                "remaining": len(discovered_wallets - _enriched_wallets),
            },
        )

    finally:
        await client.close()
