"""Job: wallet clustering and insider scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    CLUSTER_MIN_OVERLAP,
    CLUSTER_MIN_SIMILARITY,
    CLUSTER_TIME_WINDOW,
    INSIDER_FRESHNESS_DAYS,
    INSIDER_WIN_RATE_THRESHOLD,
)

logger = logging.getLogger(__name__)


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


async def run_wallet_analyzer() -> None:
    """Run wallet clustering and insider scoring.

    1. Query recent wallet_activity for clustering signals.
    2. Compute pairwise wallet similarity based on timing, market overlap, direction.
    3. Group into clusters using greedy algorithm.
    4. Score each tracked wallet on insider-risk factors.
    5. Write results to wallet_clusters and insider_scores tables.
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # =====================================================================
        # PART 1: WALLET CLUSTERING
        # =====================================================================

        # Fetch recent trade activity (last 24h) for all tracked wallets
        activity_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                condition_id,
                side,
                timestamp
            FROM wallet_activity
            WHERE timestamp >= now() - INTERVAL 24 HOUR
              AND activity_type = 'TRADE'
              AND side != ''
            ORDER BY timestamp
            """,
        )

        # Build per-wallet trade profiles
        # wallet -> [(condition_id, side, timestamp), ...]
        wallet_trades: dict[str, list[tuple[str, str, datetime]]] = defaultdict(list)
        for row in activity_result.result_rows:
            wallet = row[0]
            cid = row[1]
            side = row[2]
            ts = row[3]
            wallet_trades[wallet].append((cid, side, ts))

        wallets = list(wallet_trades.keys())

        # Compute pairwise similarity for wallets with enough trades
        cluster_rows: list[list] = []
        clustered_wallets: set[str] = set()

        if len(wallets) >= 2:
            # Pre-compute per-wallet market sets and trade events
            wallet_markets: dict[str, set[str]] = {}
            wallet_events: dict[str, list[tuple[str, str, float]]] = {}

            for w in wallets:
                trades = wallet_trades[w]
                wallet_markets[w] = {t[0] for t in trades}
                wallet_events[w] = [
                    (t[0], t[1], t[2].timestamp() if hasattr(t[2], 'timestamp') else float(t[2]))
                    for t in trades
                ]

            # Pairwise comparison (limit to avoid O(n^2) explosion for large wallet sets)
            comparison_wallets = wallets[:200]  # Cap comparisons

            for i in range(len(comparison_wallets)):
                for j in range(i + 1, len(comparison_wallets)):
                    w1 = comparison_wallets[i]
                    w2 = comparison_wallets[j]

                    if w1 in clustered_wallets and w2 in clustered_wallets:
                        continue

                    # Market overlap
                    shared = wallet_markets[w1] & wallet_markets[w2]
                    if len(shared) < CLUSTER_MIN_OVERLAP:
                        continue

                    all_markets = wallet_markets[w1] | wallet_markets[w2]
                    overlap = len(shared) / len(all_markets) if all_markets else 0

                    # Direction agreement (in shared markets)
                    w1_directions: dict[str, str] = {}
                    w2_directions: dict[str, str] = {}
                    for cid, side, _ in wallet_events[w1]:
                        if cid in shared:
                            w1_directions[cid] = side
                    for cid, side, _ in wallet_events[w2]:
                        if cid in shared:
                            w2_directions[cid] = side

                    agree_count = sum(
                        1 for cid in shared
                        if cid in w1_directions and cid in w2_directions
                        and w1_directions[cid] == w2_directions[cid]
                    )
                    direction_agree = agree_count / len(shared) if shared else 0

                    # Timing correlation (trades within CLUSTER_TIME_WINDOW seconds)
                    timing_matches = 0
                    timing_total = 0
                    for cid in shared:
                        w1_times = [ts for c, _, ts in wallet_events[w1] if c == cid]
                        w2_times = [ts for c, _, ts in wallet_events[w2] if c == cid]
                        for t1 in w1_times:
                            for t2 in w2_times:
                                timing_total += 1
                                if abs(t1 - t2) <= CLUSTER_TIME_WINDOW:
                                    timing_matches += 1

                    timing_corr = timing_matches / timing_total if timing_total > 0 else 0

                    # Composite similarity
                    similarity = (
                        0.3 * overlap +
                        0.4 * direction_agree +
                        0.3 * timing_corr
                    )

                    if similarity >= CLUSTER_MIN_SIMILARITY:
                        cluster_id = str(uuid.uuid4())
                        cluster_wallets = [w1, w2]
                        common = list(shared)[:20]  # Cap stored common markets

                        cluster_rows.append([
                            cluster_id,
                            cluster_wallets,            # wallets array
                            len(cluster_wallets),       # size
                            similarity,                 # similarity_score
                            timing_corr,                # timing_corr
                            overlap,                    # market_overlap
                            direction_agree,            # direction_agreement
                            common,                     # common_markets
                            "",                         # label
                            now,                        # created_at
                            now,                        # updated_at
                        ])

                        clustered_wallets.add(w1)
                        clustered_wallets.add(w2)

        if cluster_rows:
            await writer.write_clusters(cluster_rows)

        # =====================================================================
        # PART 2: INSIDER SCORING
        # =====================================================================

        # Fetch tracked wallet metadata for scoring
        # We need: wallet age, win rate, market diversity, position sizes
        profile_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                profile_created_at,
                first_seen_at
            FROM trader_profiles FINAL
            """,
        )

        wallet_profiles: dict[str, dict] = {}
        for row in profile_result.result_rows:
            wallet_profiles[row[0]] = {
                "created_at": row[1],
                "first_seen": row[2],
            }

        # Win rate per wallet (resolved markets where they held the winning outcome)
        win_rate_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wp.proxy_wallet,
                countIf(m.resolved = 1 AND m.winning_outcome = wp.outcome) AS wins,
                countIf(m.resolved = 1) AS resolved_count
            FROM (SELECT * FROM wallet_positions FINAL) AS wp
            INNER JOIN (
                SELECT condition_id, resolved, winning_outcome
                FROM markets FINAL
                WHERE resolved = 1
            ) AS m ON wp.condition_id = m.condition_id
            WHERE wp.size > 0
            GROUP BY wp.proxy_wallet
            HAVING resolved_count >= 3
            """,
        )

        wallet_win_rates: dict[str, tuple[int, int]] = {}
        for row in win_rate_result.result_rows:
            wallet_win_rates[row[0]] = (row[1], row[2])  # (wins, total)

        # Position size vs market liquidity
        size_vs_liq_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wp.proxy_wallet,
                avg(wp.current_value / greatest(m.liquidity, 1.0)) AS avg_size_ratio
            FROM (SELECT * FROM wallet_positions FINAL) AS wp
            INNER JOIN (
                SELECT condition_id, liquidity
                FROM markets FINAL
                WHERE active = 1
            ) AS m ON wp.condition_id = m.condition_id
            WHERE wp.size > 0
            GROUP BY wp.proxy_wallet
            """,
        )

        wallet_size_ratios: dict[str, float] = {}
        for row in size_vs_liq_result.result_rows:
            wallet_size_ratios[row[0]] = float(row[1])

        # Market diversity (how many unique markets a wallet trades in)
        diversity_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                uniq(condition_id) AS unique_markets
            FROM wallet_activity
            WHERE activity_type = 'TRADE'
              AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY proxy_wallet
            """,
        )

        wallet_diversity: dict[str, int] = {}
        for row in diversity_result.result_rows:
            wallet_diversity[row[0]] = int(row[1])

        # Compute insider scores
        insider_rows: list[list] = []
        all_scored_wallets = set(wallet_profiles.keys()) | set(wallet_win_rates.keys())

        for wallet in all_scored_wallets:
            # --- Freshness score (0-100) ---
            profile = wallet_profiles.get(wallet, {})
            created = profile.get("created_at")
            if created and hasattr(created, 'timestamp'):
                age_days = (now - created.replace(tzinfo=timezone.utc if created.tzinfo is None else created.tzinfo)).days
            else:
                age_days = 999  # Unknown = not fresh

            if age_days <= INSIDER_FRESHNESS_DAYS:
                freshness = 100 * (1 - age_days / INSIDER_FRESHNESS_DAYS)
            else:
                freshness = 0.0

            # --- Win rate score (0-100) ---
            wins, total = wallet_win_rates.get(wallet, (0, 0))
            if total >= 3:
                wr = wins / total
                if wr >= INSIDER_WIN_RATE_THRESHOLD:
                    win_rate_s = min(100, (wr - 0.5) * 200)  # 0.5 -> 0, 1.0 -> 100
                else:
                    win_rate_s = 0.0
            else:
                win_rate_s = 0.0

            # --- Niche score (0-100) ---
            diversity = wallet_diversity.get(wallet, 0)
            if 0 < diversity <= 3:
                niche_s = 80.0  # Very focused
            elif diversity <= 5:
                niche_s = 40.0
            else:
                niche_s = 0.0

            # --- Size vs liquidity score (0-100) ---
            size_ratio = wallet_size_ratios.get(wallet, 0)
            if size_ratio > 0.1:
                size_s = min(100, size_ratio * 500)  # 0.1 -> 50, 0.2 -> 100
            else:
                size_s = 0.0

            # --- Timing score (0-100) ---
            # Placeholder: use 0 for now; requires event resolution timestamps
            # which we do not yet track. Future enhancement: compare trade timestamps
            # to market resolution or news event timestamps.
            timing_s = 0.0

            # --- Composite score ---
            composite = (
                0.20 * freshness +
                0.30 * win_rate_s +
                0.15 * niche_s +
                0.25 * size_s +
                0.10 * timing_s
            )

            factors_json = json.dumps({
                "freshness": round(freshness, 2),
                "win_rate": round(win_rate_s, 2),
                "niche_focus": round(niche_s, 2),
                "size_vs_liquidity": round(size_s, 2),
                "pre_announcement": round(timing_s, 2),
            })

            insider_rows.append([
                wallet,
                round(composite, 2),        # score
                factors_json,                # factors JSON
                round(freshness, 2),         # freshness_score
                round(win_rate_s, 2),        # win_rate_score
                round(niche_s, 2),           # niche_score
                round(size_s, 2),            # size_score
                round(timing_s, 2),          # timing_score
                now,                         # computed_at
            ])

        if insider_rows:
            await writer.write_insider_scores(insider_rows)

        await writer.flush_all()

        logger.info(
            "wallet_analyzer_complete",
            extra={
                "clusters_found": len(cluster_rows),
                "wallets_scored": len(insider_rows),
            },
        )

    except Exception:
        logger.error("wallet_analyzer_error", exc_info=True)
