"""Job: insider trading detection across Polymarket markets.

Detects suspicious trading patterns via five methods:
1. Pre-news trading — trades preceding major price moves or resolutions
2. Statistical anomaly — z-score analysis of trader metrics vs population
3. Profitability analysis — risk-adjusted returns, Middle East focus
4. Coordinated trading — wallet groups trading together within 5-min windows
5. Composite suspicion scoring — weighted 0-100 score with tier labels
"""

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
    INSIDER_DETECT_PRICE_MOVE_THRESHOLD,
    INSIDER_DETECT_LOOKBACK_HOURS,
    INSIDER_DETECT_ZSCORE_THRESHOLD,
    INSIDER_DETECT_COORDINATION_WINDOW,
    INSIDER_DETECT_COORDINATION_MIN_OVERLAP,
    INSIDER_DETECT_LARGE_TRADE_USD,
    INSIDER_DETECT_TOP_WALLETS,
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


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _suspicion_tier(score: float) -> str:
    if score >= 75:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 25:
        return "medium"
    return "low"


async def run_insider_detector() -> None:
    """Run all insider trading detection methods and write results.

    1. Detect pre-news events (major price moves, resolutions).
    2. Score individual trades that preceded those events.
    3. Compute statistical anomaly scores per trader.
    4. Compute profitability-based suspicion scores.
    5. Detect coordinated trading groups.
    6. Aggregate into composite suspicion profiles.
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # =================================================================
        # PART 1: PRE-NEWS EVENT DETECTION
        # =================================================================
        # Find markets with large price moves (>threshold in 1 hour)
        price_move_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                condition_id,
                min(price) AS min_price,
                max(price) AS max_price,
                argMin(price, timestamp) AS price_start,
                argMax(price, timestamp) AS price_end,
                min(timestamp) AS window_start,
                max(timestamp) AS window_end,
                sum(size * price) AS volume_during
            FROM market_prices
            WHERE timestamp >= now() - INTERVAL {INSIDER_DETECT_LOOKBACK_HOURS} HOUR
            GROUP BY condition_id,
                     toStartOfHour(timestamp)
            HAVING abs(price_end - price_start) / greatest(price_start, 0.01) > {INSIDER_DETECT_PRICE_MOVE_THRESHOLD}
            ORDER BY abs(price_end - price_start) / greatest(price_start, 0.01) DESC
            LIMIT 500
            """,
        )

        pre_news_events: list[dict] = []
        pre_news_rows: list[list] = []

        for row in price_move_result.result_rows:
            cid = row[0]
            price_start = float(row[3])
            price_end = float(row[4])
            w_start = row[5]
            w_end = row[6]
            vol = float(row[7])

            pct_change = (price_end - price_start) / max(price_start, 0.01)
            direction = "up" if pct_change > 0 else "down"

            event = {
                "condition_id": cid,
                "event_type": "price_move",
                "magnitude": abs(pct_change),
                "direction": direction,
                "price_before": price_start,
                "price_after": price_end,
                "window_start": w_start,
                "window_end": w_end,
                "volume_during": vol,
            }
            pre_news_events.append(event)

            pre_news_rows.append([
                str(uuid.uuid4()),       # event_id
                cid,                     # condition_id
                "price_move",            # event_type
                abs(pct_change),         # magnitude
                direction,               # direction
                price_start,             # price_before
                price_end,               # price_after
                vol,                     # volume_during
                "",                      # category (filled below)
                "",                      # event_slug (filled below)
                "",                      # question (filled below)
                w_start,                 # window_start
                w_end,                   # window_end
                now,                     # detected_at
            ])

        # Also detect recently resolved markets using market_events for
        # precise resolution timestamps (research finding: market_events has
        # event_type='resolved' with exact event_time)
        resolution_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                m.condition_id,
                m.winning_outcome,
                m.category,
                m.event_slug,
                m.question,
                coalesce(me.event_time, m.updated_at) AS resolved_at
            FROM (
                SELECT condition_id, winning_outcome, category,
                       event_slug, question, updated_at
                FROM markets FINAL
                WHERE resolved = 1
                  AND updated_at >= now() - INTERVAL {INSIDER_DETECT_LOOKBACK_HOURS} HOUR
            ) AS m
            LEFT JOIN (
                SELECT condition_id, max(event_time) AS event_time
                FROM market_events
                WHERE event_type = 'resolved'
                  AND event_time >= now() - INTERVAL {INSIDER_DETECT_LOOKBACK_HOURS} HOUR
                GROUP BY condition_id
            ) AS me ON m.condition_id = me.condition_id
            """,
        )

        for row in resolution_result.result_rows:
            cid = row[0]
            winning = row[1]
            category = row[2]
            slug = row[3]
            question = row[4]
            resolved_at = row[5]

            event = {
                "condition_id": cid,
                "event_type": "resolution",
                "magnitude": 1.0,
                "direction": "up" if winning == "Yes" else "down",
                "window_start": resolved_at,
                "window_end": resolved_at,
            }
            pre_news_events.append(event)

            pre_news_rows.append([
                str(uuid.uuid4()),       # event_id
                cid,                     # condition_id
                "resolution",            # event_type
                1.0,                     # magnitude
                "up" if winning == "Yes" else "down",
                0.0,                     # price_before
                1.0 if winning == "Yes" else 0.0,
                0.0,                     # volume_during
                category,                # category
                slug,                    # event_slug
                question,                # question
                resolved_at,             # window_start
                resolved_at,             # window_end
                now,                     # detected_at
            ])

        # Enrich price_move events with market metadata
        if pre_news_events:
            move_cids = list({e["condition_id"] for e in pre_news_events if e["event_type"] == "price_move"})
            if move_cids:
                cid_list = ", ".join(f"'{c}'" for c in move_cids[:500])
                meta_result = await asyncio.to_thread(
                    client.query,
                    f"""
                    SELECT condition_id, category, event_slug, question
                    FROM markets FINAL
                    WHERE condition_id IN ({cid_list})
                    """,
                )
                meta_map = {row[0]: (row[1], row[2], row[3]) for row in meta_result.result_rows}

                for pn_row in pre_news_rows:
                    if pn_row[2] == "price_move" and pn_row[1] in meta_map:
                        cat, slug, q = meta_map[pn_row[1]]
                        pn_row[8] = cat       # category
                        pn_row[9] = slug      # event_slug
                        pn_row[10] = q        # question

        if pre_news_rows:
            await writer.write_pre_news_events(pre_news_rows)

        # =================================================================
        # PART 2: PRE-NEWS TRADE SCORING
        # =================================================================
        # For each pre-news event, find trades in the preceding window
        trade_signal_rows: list[list] = []
        # Map: wallet -> list of (pre_news_score, direction_correct)
        wallet_pre_news: dict[str, list[tuple[float, bool]]] = defaultdict(list)

        for event in pre_news_events:
            cid = event["condition_id"]
            w_start = event["window_start"]
            direction = event["direction"]
            magnitude = event["magnitude"]

            # Look for trades in the 24 hours before the event
            pre_trades_result = await asyncio.to_thread(
                client.query,
                f"""
                SELECT
                    t.proxy_wallet,
                    t.condition_id,
                    t.side,
                    t.size,
                    t.usdc_size,
                    t.price,
                    t.transaction_hash,
                    t.timestamp
                FROM wallet_activity t
                WHERE t.condition_id = '{cid}'
                  AND t.activity_type = 'TRADE'
                  AND t.timestamp < toDateTime64('{w_start}', 3)
                  AND t.timestamp >= toDateTime64('{w_start}', 3) - INTERVAL 24 HOUR
                ORDER BY t.timestamp DESC
                LIMIT 200
                """,
            )

            for trow in pre_trades_result.result_rows:
                wallet = trow[0]
                side = trow[2]
                size = float(trow[3])
                usdc = float(trow[4])
                price = float(trow[5])
                tx_hash = trow[6]
                trade_ts = trow[7]

                # Score by timing proximity (closer = higher score)
                if hasattr(w_start, 'timestamp') and hasattr(trade_ts, 'timestamp'):
                    hours_before = (w_start.timestamp() - trade_ts.timestamp()) / 3600.0
                elif hasattr(w_start, 'timestamp'):
                    hours_before = (w_start.timestamp() - float(trade_ts)) / 3600.0
                else:
                    hours_before = 12.0  # fallback

                hours_before = max(0.0, hours_before)

                # Timing score: trades closer to event score higher
                # 0h -> 100, 24h -> 0, exponential decay
                timing_score = 100.0 * (2.0 ** (-hours_before / 4.0))

                # Direction correctness
                correct = (
                    (direction == "up" and side == "BUY") or
                    (direction == "down" and side == "SELL")
                )

                # Size bonus: larger trades are more suspicious
                size_bonus = min(30.0, usdc / 500.0) if usdc > 0 else 0.0

                # Pre-news score
                pre_score = _clamp(
                    timing_score * (1.5 if correct else 0.3) +
                    size_bonus * (1.0 if correct else 0.0) +
                    magnitude * 50  # bigger moves are more significant
                )

                wallet_pre_news[wallet].append((pre_score, correct))

                trade_signal_rows.append([
                    tx_hash or str(uuid.uuid4()),  # trade_id
                    cid,                            # condition_id
                    wallet,                         # proxy_wallet
                    side,                           # side
                    size,                           # size
                    usdc,                           # usdc_size
                    price,                          # price
                    trade_ts,                       # trade_timestamp
                    round(pre_score, 2),            # pre_news_score
                    0.0,                            # statistical_score (filled later)
                    0.0,                            # profitability_score (filled later)
                    0.0,                            # coordination_score (filled later)
                    0.0,                            # composite_score (filled later)
                    "",                             # category
                    "",                             # event_slug
                    1 if correct else 0,            # direction_correct
                    round(hours_before, 2),         # hours_before_move
                    round(magnitude * 100, 2),      # price_move_pct
                    now,                            # scored_at
                ])

        # =================================================================
        # PART 3: STATISTICAL ANOMALY DETECTION
        # =================================================================
        # Compute population statistics, then z-score individual traders
        pop_stats_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                count() AS trade_count,
                countIf(side = 'BUY') / greatest(count(), 1) AS buy_ratio,
                avg(usdc_size) AS avg_size,
                stddevPop(usdc_size) AS std_size,
                uniq(condition_id) AS unique_markets
            FROM wallet_activity
            WHERE activity_type = 'TRADE'
              AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY proxy_wallet
            HAVING trade_count >= 5
            """,
        )

        # Compute population-level means and stds
        all_counts: list[float] = []
        all_avg_sizes: list[float] = []
        all_diversities: list[float] = []
        wallet_stats: dict[str, dict] = {}

        for row in pop_stats_result.result_rows:
            wallet = row[0]
            trade_count = int(row[1])
            buy_ratio = float(row[2])
            avg_size = float(row[3])
            std_size = float(row[4])
            unique_markets = int(row[5])

            all_counts.append(trade_count)
            all_avg_sizes.append(avg_size)
            all_diversities.append(unique_markets)

            wallet_stats[wallet] = {
                "trade_count": trade_count,
                "buy_ratio": buy_ratio,
                "avg_size": avg_size,
                "std_size": std_size,
                "unique_markets": unique_markets,
            }

        # Compute population parameters
        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals: list[float]) -> float:
            if len(vals) < 2:
                return 1.0
            m = _mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / len(vals)
            return max(variance ** 0.5, 0.001)

        pop_mean_count = _mean(all_counts)
        pop_std_count = _std(all_counts)
        pop_mean_size = _mean(all_avg_sizes)
        pop_std_size = _std(all_avg_sizes)
        pop_mean_div = _mean(all_diversities)
        pop_std_div = _std(all_diversities)

        # Win rate per wallet
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

        wallet_win_rates: dict[str, tuple[int, int, float]] = {}
        all_win_rates: list[float] = []
        for row in win_rate_result.result_rows:
            wallet = row[0]
            wins = int(row[1])
            total = int(row[2])
            wr = wins / total if total > 0 else 0.0
            wallet_win_rates[wallet] = (wins, total, wr)
            all_win_rates.append(wr)

        pop_mean_wr = _mean(all_win_rates)
        pop_std_wr = _std(all_win_rates)

        # ROI per wallet
        roi_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                proxy_wallet,
                sum(cash_pnl) AS total_pnl,
                sum(initial_value) AS total_invested
            FROM wallet_positions FINAL
            WHERE size > 0
            GROUP BY proxy_wallet
            HAVING total_invested > 0
            """,
        )

        wallet_roi: dict[str, tuple[float, float, float]] = {}
        all_rois: list[float] = []
        for row in roi_result.result_rows:
            wallet = row[0]
            pnl = float(row[1])
            invested = float(row[2])
            roi = pnl / invested if invested > 0 else 0.0
            wallet_roi[wallet] = (pnl, invested, roi)
            all_rois.append(roi)

        pop_mean_roi = _mean(all_rois)
        pop_std_roi = _std(all_rois)

        # Toxic flow per wallet from market_microstructure
        # (research finding: kyle_lambda and toxic_flow_ratio indicate
        # informed trading in the markets a wallet trades in)
        wallet_toxic_flow: dict[str, float] = {}
        try:
            toxic_result = await asyncio.to_thread(
                client.query,
                """
                SELECT
                    wa.proxy_wallet,
                    avg(mm.toxic_flow_ratio) AS avg_toxic_flow
                FROM wallet_activity wa
                INNER JOIN (
                    SELECT condition_id,
                           argMax(toxic_flow_ratio, snapshot_time) AS toxic_flow_ratio
                    FROM market_microstructure
                    WHERE snapshot_time >= now() - INTERVAL 24 HOUR
                    GROUP BY condition_id
                ) AS mm ON wa.condition_id = mm.condition_id
                WHERE wa.activity_type = 'TRADE'
                  AND wa.timestamp >= now() - INTERVAL 7 DAY
                GROUP BY wa.proxy_wallet
                HAVING count() >= 3
                """,
            )
            for row in toxic_result.result_rows:
                wallet_toxic_flow[row[0]] = float(row[1])
        except Exception:
            logger.debug("toxic_flow_query_skip", extra={"reason": "table_may_be_empty"})

        # Compute per-wallet statistical anomaly scores
        wallet_stat_scores: dict[str, float] = {}
        wallet_zscores: dict[str, dict] = {}

        for wallet, stats in wallet_stats.items():
            z_count = abs(stats["trade_count"] - pop_mean_count) / pop_std_count
            z_size = abs(stats["avg_size"] - pop_mean_size) / pop_std_size
            z_div = abs(stats["unique_markets"] - pop_mean_div) / pop_std_div

            _, _, wr = wallet_win_rates.get(wallet, (0, 0, 0.0))
            z_wr = (wr - pop_mean_wr) / pop_std_wr if pop_std_wr > 0 else 0.0

            _, _, roi = wallet_roi.get(wallet, (0.0, 0.0, 0.0))
            z_roi = (roi - pop_mean_roi) / pop_std_roi if pop_std_roi > 0 else 0.0

            # Count how many metrics exceed threshold
            z_scores = [z_count, z_size, z_div, z_wr, z_roi]
            exceeded = sum(1 for z in z_scores if abs(z) > INSIDER_DETECT_ZSCORE_THRESHOLD)

            # Toxic flow bonus: wallets trading in markets with high
            # informed-trading signals from microstructure analysis
            toxic = wallet_toxic_flow.get(wallet, 0.0)
            toxic_bonus = min(15.0, toxic * 30) if toxic > 0.3 else 0.0

            # Statistical anomaly score: more exceeded metrics = higher score
            stat_score = _clamp(
                exceeded * 20.0 +  # 20 points per exceeded metric
                max(0, z_wr - INSIDER_DETECT_ZSCORE_THRESHOLD) * 10 +  # Extra for high win rate
                max(0, z_roi - INSIDER_DETECT_ZSCORE_THRESHOLD) * 10 +  # Extra for high ROI
                toxic_bonus  # Informed trading signal from microstructure
            )

            wallet_stat_scores[wallet] = stat_score
            wallet_zscores[wallet] = {
                "z_trade_count": round(z_count, 2),
                "z_avg_size": round(z_size, 2),
                "z_diversity": round(z_div, 2),
                "z_win_rate": round(z_wr, 2),
                "z_roi": round(z_roi, 2),
                "exceeded_count": exceeded,
                "toxic_flow": round(toxic, 3),
            }

        # =================================================================
        # PART 4: PROFITABILITY ANALYSIS (Middle East focus)
        # =================================================================
        # Get per-wallet PnL breakdown by category
        category_pnl_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wp.proxy_wallet,
                m.category,
                sum(wp.cash_pnl) AS cat_pnl,
                count() AS cat_positions,
                countIf(wp.cash_pnl > 0) AS cat_wins
            FROM (SELECT * FROM wallet_positions FINAL) AS wp
            INNER JOIN (
                SELECT condition_id, category
                FROM markets FINAL
            ) AS m ON wp.condition_id = m.condition_id
            WHERE wp.size > 0
            GROUP BY wp.proxy_wallet, m.category
            """,
        )

        # wallet -> {category: {pnl, positions, wins}}
        wallet_cat_pnl: dict[str, dict[str, dict]] = defaultdict(dict)
        wallet_total_positions: dict[str, int] = defaultdict(int)

        for row in category_pnl_result.result_rows:
            wallet = row[0]
            cat = row[1]
            pnl = float(row[2])
            positions = int(row[3])
            wins = int(row[4])
            wallet_cat_pnl[wallet][cat] = {
                "pnl": pnl,
                "positions": positions,
                "wins": wins,
            }
            wallet_total_positions[wallet] += positions

        wallet_profit_scores: dict[str, float] = {}
        wallet_mideast_pct: dict[str, float] = {}

        for wallet in wallet_cat_pnl:
            total_pos = wallet_total_positions[wallet]
            if total_pos == 0:
                continue

            # Middle East concentration
            mideast_data = wallet_cat_pnl[wallet].get("Middle East", {})
            mideast_pos = mideast_data.get("positions", 0)
            mideast_pnl = mideast_data.get("pnl", 0.0)
            mideast_wins = mideast_data.get("wins", 0)
            mideast_pct = mideast_pos / total_pos if total_pos > 0 else 0.0
            wallet_mideast_pct[wallet] = mideast_pct

            # Profitability score: high ROI + Middle East focus = suspicious
            pnl_total, invested_total, roi_total = wallet_roi.get(wallet, (0.0, 0.0, 0.0))

            profit_score = 0.0
            # High absolute ROI
            if roi_total > 0.5:
                profit_score += min(40.0, roi_total * 20)
            # Middle East focus bonus
            if mideast_pct > 0.3:
                profit_score += mideast_pct * 30
            # Middle East win rate bonus
            if mideast_pos >= 3:
                me_wr = mideast_wins / mideast_pos
                if me_wr > 0.7:
                    profit_score += (me_wr - 0.5) * 60

            wallet_profit_scores[wallet] = _clamp(profit_score)

        # =================================================================
        # PART 5: COORDINATED TRADING DETECTION
        # =================================================================
        # Find wallet groups trading same markets within 5-min windows
        coord_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT
                proxy_wallet,
                condition_id,
                side,
                usdc_size,
                timestamp
            FROM wallet_activity
            WHERE activity_type = 'TRADE'
              AND timestamp >= now() - INTERVAL 24 HOUR
              AND side != ''
            ORDER BY condition_id, timestamp
            """,
        )

        # Build per-wallet trade events
        wallet_coord_trades: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)
        for row in coord_result.result_rows:
            wallet = row[0]
            cid = row[1]
            side = row[2]
            usdc = float(row[3])
            ts = row[4]
            ts_epoch = ts.timestamp() if hasattr(ts, 'timestamp') else float(ts)
            wallet_coord_trades[wallet].append((cid, side, usdc, ts_epoch))

        coord_wallets = [w for w in wallet_coord_trades if len(wallet_coord_trades[w]) >= 3]
        coord_wallets = coord_wallets[:INSIDER_DETECT_TOP_WALLETS]  # cap

        wallet_coord_scores: dict[str, float] = {}
        coord_group_rows: list[list] = []
        grouped_wallets: set[str] = set()

        if len(coord_wallets) >= 2:
            # Pre-compute per-wallet market sets
            w_markets: dict[str, set[str]] = {}
            w_events: dict[str, list[tuple[str, str, float, float]]] = {}

            for w in coord_wallets:
                trades = wallet_coord_trades[w]
                w_markets[w] = {t[0] for t in trades}
                w_events[w] = trades

            for i in range(len(coord_wallets)):
                for j in range(i + 1, len(coord_wallets)):
                    w1 = coord_wallets[i]
                    w2 = coord_wallets[j]

                    if w1 in grouped_wallets and w2 in grouped_wallets:
                        continue

                    shared = w_markets[w1] & w_markets[w2]
                    if len(shared) < INSIDER_DETECT_COORDINATION_MIN_OVERLAP:
                        continue

                    all_m = w_markets[w1] | w_markets[w2]
                    overlap = len(shared) / len(all_m) if all_m else 0

                    # Direction agreement
                    w1_dirs: dict[str, str] = {}
                    w2_dirs: dict[str, str] = {}
                    w1_sizes: dict[str, float] = {}
                    w2_sizes: dict[str, float] = {}
                    for cid, side, usdc, _ in w_events[w1]:
                        if cid in shared:
                            w1_dirs[cid] = side
                            w1_sizes[cid] = usdc
                    for cid, side, usdc, _ in w_events[w2]:
                        if cid in shared:
                            w2_dirs[cid] = side
                            w2_sizes[cid] = usdc

                    agree = sum(
                        1 for cid in shared
                        if cid in w1_dirs and cid in w2_dirs
                        and w1_dirs[cid] == w2_dirs[cid]
                    )
                    dir_agree = agree / len(shared) if shared else 0

                    # Timing correlation (within COORDINATION_WINDOW seconds)
                    timing_matches = 0
                    timing_total = 0
                    for cid in shared:
                        t1s = [ts for c, _, _, ts in w_events[w1] if c == cid]
                        t2s = [ts for c, _, _, ts in w_events[w2] if c == cid]
                        for t1 in t1s:
                            for t2 in t2s:
                                timing_total += 1
                                if abs(t1 - t2) <= INSIDER_DETECT_COORDINATION_WINDOW:
                                    timing_matches += 1

                    timing_corr = timing_matches / timing_total if timing_total > 0 else 0

                    # Trade size similarity
                    size_ratios = []
                    for cid in shared:
                        if cid in w1_sizes and cid in w2_sizes:
                            s1, s2 = w1_sizes[cid], w2_sizes[cid]
                            if max(s1, s2) > 0:
                                size_ratios.append(min(s1, s2) / max(s1, s2))
                    size_sim = _mean(size_ratios) if size_ratios else 0.0

                    # Composite coordination score
                    coord_score = (
                        0.25 * overlap +
                        0.30 * dir_agree +
                        0.30 * timing_corr +
                        0.15 * size_sim
                    )

                    if coord_score >= 0.7:
                        # Get categories
                        common_cats: set[str] = set()
                        for cid in shared:
                            # We don't have category per-trade here, skip for now
                            pass

                        total_vol = sum(u for _, _, u, _ in w_events[w1]) + sum(u for _, _, u, _ in w_events[w2])

                        group_id = str(uuid.uuid4())
                        coord_group_rows.append([
                            group_id,                    # group_id
                            [w1, w2],                    # wallets
                            2,                           # size
                            round(coord_score, 4),       # correlation_score
                            round(timing_corr, 4),       # timing_correlation
                            round(overlap, 4),           # market_overlap
                            round(dir_agree, 4),         # direction_agreement
                            round(size_sim, 4),          # size_similarity
                            list(shared)[:20],           # common_markets
                            [],                          # common_categories
                            round(total_vol, 2),         # total_volume
                            0.0,                         # avg_suspicion (filled later)
                            "",                          # label
                            now,                         # detected_at
                            now,                         # updated_at
                        ])

                        wallet_coord_scores[w1] = max(
                            wallet_coord_scores.get(w1, 0.0),
                            _clamp(coord_score * 100)
                        )
                        wallet_coord_scores[w2] = max(
                            wallet_coord_scores.get(w2, 0.0),
                            _clamp(coord_score * 100)
                        )
                        grouped_wallets.add(w1)
                        grouped_wallets.add(w2)

        # =================================================================
        # PART 6: COMPOSITE SUSPICION PROFILES
        # =================================================================
        # Weights: pre_news (0.30), statistical (0.25), profitability (0.20),
        #          coordination (0.15), category_focus (0.10)
        WEIGHTS = {
            "pre_news": 0.30,
            "statistical": 0.25,
            "profitability": 0.20,
            "coordination": 0.15,
            "category_focus": 0.10,
        }

        all_wallets = set(wallet_stats.keys()) | set(wallet_pre_news.keys()) | set(wallet_cat_pnl.keys())
        # Only score wallets with enough activity
        all_wallets = {w for w in all_wallets if w in wallet_stats and wallet_stats[w]["trade_count"] >= 5}

        profile_rows: list[list] = []

        for wallet in all_wallets:
            # Pre-news score: average of individual trade pre-news scores
            pn_scores = wallet_pre_news.get(wallet, [])
            if pn_scores:
                avg_pn = sum(s for s, _ in pn_scores) / len(pn_scores)
                correct_rate = sum(1 for _, c in pn_scores if c) / len(pn_scores)
                pn_component = _clamp(avg_pn * (1 + correct_rate))
            else:
                pn_component = 0.0

            # Statistical anomaly score
            stat_component = wallet_stat_scores.get(wallet, 0.0)

            # Profitability score
            profit_component = wallet_profit_scores.get(wallet, 0.0)

            # Coordination score
            coord_component = wallet_coord_scores.get(wallet, 0.0)

            # Category focus score (Middle East concentration)
            me_pct = wallet_mideast_pct.get(wallet, 0.0)
            cat_focus_component = _clamp(me_pct * 100)

            # Weighted composite
            suspicion = (
                WEIGHTS["pre_news"] * pn_component +
                WEIGHTS["statistical"] * stat_component +
                WEIGHTS["profitability"] * profit_component +
                WEIGHTS["coordination"] * coord_component +
                WEIGHTS["category_focus"] * cat_focus_component
            )
            suspicion = _clamp(suspicion)
            tier = _suspicion_tier(suspicion)

            # Stats
            stats = wallet_stats.get(wallet, {})
            wins, total, wr = wallet_win_rates.get(wallet, (0, 0, 0.0))
            pnl_total, _, roi_total = wallet_roi.get(wallet, (0.0, 0.0, 0.0))
            flagged_count = len(pn_scores)

            factors_json = json.dumps({
                "pre_news": round(pn_component, 2),
                "statistical": round(stat_component, 2),
                "profitability": round(profit_component, 2),
                "coordination": round(coord_component, 2),
                "category_focus": round(cat_focus_component, 2),
                "z_scores": wallet_zscores.get(wallet, {}),
                "mideast_pct": round(me_pct, 3),
            })

            profile_rows.append([
                wallet,                          # proxy_wallet
                round(suspicion, 2),             # suspicion_score
                tier,                            # suspicion_tier
                round(pn_component, 2),          # pre_news_score
                round(stat_component, 2),        # statistical_score
                round(profit_component, 2),      # profitability_score
                round(coord_component, 2),       # coordination_score
                round(cat_focus_component, 2),   # category_focus_score
                factors_json,                    # factors
                stats.get("trade_count", 0),     # total_trades
                round(wr, 4),                    # win_rate
                round(roi_total, 4),             # avg_roi
                round(me_pct, 4),                # mideast_trade_pct
                flagged_count,                   # flagged_trade_count
                round(pnl_total, 2),             # total_pnl
                now,                             # first_flagged_at
                now,                             # computed_at
            ])

        # Update trade signals with their wallet's composite scores
        wallet_composites = {row[0]: row[1] for row in profile_rows}
        for ts_row in trade_signal_rows:
            wallet = ts_row[2]
            ts_row[9] = round(wallet_stat_scores.get(wallet, 0.0), 2)    # statistical_score
            ts_row[10] = round(wallet_profit_scores.get(wallet, 0.0), 2)  # profitability_score
            ts_row[11] = round(wallet_coord_scores.get(wallet, 0.0), 2)   # coordination_score
            ts_row[12] = round(wallet_composites.get(wallet, 0.0), 2)     # composite_score

        # Update coordinated groups with avg suspicion
        for grp_row in coord_group_rows:
            group_wallets = grp_row[1]
            scores = [wallet_composites.get(w, 0.0) for w in group_wallets]
            grp_row[11] = round(sum(scores) / len(scores) if scores else 0.0, 2)

        # =================================================================
        # WRITE ALL RESULTS
        # =================================================================
        if trade_signal_rows:
            await writer.write_insider_trade_signals(trade_signal_rows)
        if profile_rows:
            await writer.write_suspicion_profiles(profile_rows)
        if coord_group_rows:
            await writer.write_coordinated_groups(coord_group_rows)

        await writer.flush_all()

        logger.info(
            "insider_detector_complete",
            extra={
                "pre_news_events": len(pre_news_events),
                "trades_scored": len(trade_signal_rows),
                "profiles_scored": len(profile_rows),
                "coordinated_groups": len(coord_group_rows),
                "critical_tier": sum(1 for r in profile_rows if r[2] == "critical"),
                "high_tier": sum(1 for r in profile_rows if r[2] == "high"),
            },
        )

    except Exception:
        logger.error("insider_detector_error", exc_info=True)
