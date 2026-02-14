"""Job: compute composite signal scores per market."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    COMPOSITE_TOP_MARKETS,
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


def _clamp(value: float, lo: float = -100.0, hi: float = 100.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


async def run_signal_compositor() -> None:
    """Compute composite signal scores for top active markets.

    Components (each normalized to -100..+100):
    1. OBI direction: from latest orderbook snapshot
    2. Volume anomaly: 4h volume vs 7d average
    3. Large trade bias: net buy/sell from large trades (24h)
    4. Momentum: from hourly OHLCV (24h price change)
    5. Smart money direction: net buy/sell from top-ranked wallets
    6. Concentration risk: top-5 holder share (high = risky = negative)
    7. Arbitrage flag: 1 if active arbitrage opportunity exists
    8. Insider activity: average insider score of wallets active in this market
    """
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # Get top active markets
        markets_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT condition_id
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            ORDER BY volume_24h DESC
            LIMIT {COMPOSITE_TOP_MARKETS}
            """,
        )

        market_ids = [row[0] for row in markets_result.result_rows]
        if not market_ids:
            logger.debug("signal_compositor_skip", extra={"reason": "no_markets"})
            return

        # --- 1. OBI scores ---
        obi_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                os.condition_id,
                arraySum(os.bid_sizes) / greatest(arraySum(os.bid_sizes) + arraySum(os.ask_sizes), 0.001) AS obi
            FROM orderbook_snapshots os
            INNER JOIN (
                SELECT condition_id, max(snapshot_time) AS max_time
                FROM orderbook_snapshots
                WHERE snapshot_time >= now() - INTERVAL 5 MINUTE
                GROUP BY condition_id
            ) latest ON os.condition_id = latest.condition_id
              AND os.snapshot_time = latest.max_time
            WHERE (arraySum(os.bid_sizes) + arraySum(os.ask_sizes)) > 0
            """,
        )

        obi_scores: dict[str, float] = {}
        for row in obi_result.result_rows:
            # OBI 0.5 = neutral (0), 0 = full bearish (-100), 1 = full bullish (+100)
            obi_scores[row[0]] = _clamp((float(row[1]) - 0.5) * 200)

        # --- 2. Volume anomaly scores ---
        vol_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                t.condition_id,
                sum(t.size) AS vol_4h,
                m.volume_1wk / 7 / 6 AS avg_4h_vol
            FROM market_trades t
            INNER JOIN (
                SELECT condition_id, volume_1wk
                FROM markets FINAL
                WHERE active = 1 AND closed = 0 AND volume_1wk > 0
            ) AS m ON t.condition_id = m.condition_id
            WHERE t.timestamp >= now() - INTERVAL 4 HOUR
            GROUP BY t.condition_id, m.volume_1wk
            """,
        )

        volume_scores: dict[str, float] = {}
        for row in vol_result.result_rows:
            vol_4h = float(row[1])
            avg_4h = float(row[2])
            if avg_4h > 0:
                ratio = vol_4h / avg_4h
                # ratio 1.0 = normal (0), 3.0+ = strong anomaly (+100)
                volume_scores[row[0]] = _clamp((ratio - 1.0) * 50)
            else:
                volume_scores[row[0]] = 0.0

        # --- 3. Large trade bias ---
        trade_bias_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                condition_id,
                sumIf(price * size, side = 'buy') AS buy_usd,
                sumIf(price * size, side = 'sell') AS sell_usd
            FROM market_trades
            WHERE timestamp >= now() - INTERVAL 24 HOUR
              AND price * size >= 1000
            GROUP BY condition_id
            """,
        )

        trade_bias_scores: dict[str, float] = {}
        for row in trade_bias_result.result_rows:
            buy = float(row[1])
            sell = float(row[2])
            total = buy + sell
            if total > 0:
                # Net buy ratio: 1.0 = all buys (+100), 0.0 = all sells (-100)
                net_ratio = (buy - sell) / total
                trade_bias_scores[row[0]] = _clamp(net_ratio * 100)
            else:
                trade_bias_scores[row[0]] = 0.0

        # --- 4. Momentum (24h price change from Gamma API data) ---
        momentum_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                condition_id,
                one_day_price_change
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
            """,
        )

        momentum_scores: dict[str, float] = {}
        for row in momentum_result.result_rows:
            change = float(row[1])
            # Price change is typically -1 to +1 range. Scale to -100..+100.
            momentum_scores[row[0]] = _clamp(change * 200)

        # --- 5. Smart money direction ---
        smart_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wa.condition_id,
                sumIf(wa.usdc_size, wa.side = 'BUY') AS sm_buy,
                sumIf(wa.usdc_size, wa.side = 'SELL') AS sm_sell
            FROM wallet_activity wa
            INNER JOIN (
                SELECT proxy_wallet
                FROM trader_rankings FINAL
                WHERE category = 'OVERALL' AND time_period = 'ALL' AND order_by = 'PNL'
                  AND rank <= 50
            ) AS tr ON wa.proxy_wallet = tr.proxy_wallet
            WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
              AND wa.activity_type = 'TRADE'
            GROUP BY wa.condition_id
            """,
        )

        smart_scores: dict[str, float] = {}
        for row in smart_result.result_rows:
            buy = float(row[1])
            sell = float(row[2])
            total = buy + sell
            if total > 0:
                net = (buy - sell) / total
                smart_scores[row[0]] = _clamp(net * 100)
            else:
                smart_scores[row[0]] = 0.0

        # --- 6. Concentration risk ---
        conc_result = await asyncio.to_thread(
            client.query,
            """
            WITH holder_stats AS (
                SELECT
                    condition_id,
                    sum(amount) AS total_amount,
                    arraySlice(groupArray(amount), 1, 5) AS top5_amounts
                FROM (
                    SELECT condition_id, amount
                    FROM market_holders FINAL
                    ORDER BY amount DESC
                )
                GROUP BY condition_id
                HAVING total_amount > 0
            )
            SELECT
                condition_id,
                arraySum(top5_amounts) / total_amount AS top5_share
            FROM holder_stats
            """,
        )

        concentration_scores: dict[str, float] = {}
        for row in conc_result.result_rows:
            share = float(row[1])
            # High concentration is risk (negative signal)
            # share 0.5 = neutral (0), 1.0 = full concentration (-100)
            concentration_scores[row[0]] = _clamp(-(share - 0.5) * 200)

        # --- 7. Arbitrage flags ---
        arb_result = await asyncio.to_thread(
            client.query,
            """
            SELECT DISTINCT condition_id
            FROM arbitrage_opportunities FINAL
            WHERE status = 'open'
            """,
        )

        arb_flags: set[str] = {row[0] for row in arb_result.result_rows}

        # --- 8. Insider activity ---
        insider_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                wa.condition_id,
                avg(ins.score) AS avg_insider_score
            FROM wallet_activity wa
            INNER JOIN (
                SELECT proxy_wallet, score
                FROM insider_scores FINAL
                WHERE score > 20
            ) AS ins ON wa.proxy_wallet = ins.proxy_wallet
            WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
              AND wa.activity_type = 'TRADE'
            GROUP BY wa.condition_id
            """,
        )

        insider_scores_map: dict[str, float] = {}
        for row in insider_result.result_rows:
            insider_scores_map[row[0]] = float(row[1])

        # =====================================================================
        # COMPOSITE CALCULATION
        # =====================================================================

        # Weights for each component
        WEIGHTS = {
            "obi": 0.20,
            "volume": 0.10,
            "trade_bias": 0.15,
            "momentum": 0.15,
            "smart_money": 0.25,
            "concentration": 0.10,
            "insider": 0.05,
        }

        signal_rows: list[list] = []

        for cid in market_ids:
            obi_s = obi_scores.get(cid, 0.0)
            vol_s = volume_scores.get(cid, 0.0)
            bias_s = trade_bias_scores.get(cid, 0.0)
            mom_s = momentum_scores.get(cid, 0.0)
            smart_s = smart_scores.get(cid, 0.0)
            conc_s = concentration_scores.get(cid, 0.0)
            arb_f = 1 if cid in arb_flags else 0
            ins_s = insider_scores_map.get(cid, 0.0)

            # Weighted composite
            composite = (
                WEIGHTS["obi"] * obi_s +
                WEIGHTS["volume"] * vol_s +
                WEIGHTS["trade_bias"] * bias_s +
                WEIGHTS["momentum"] * mom_s +
                WEIGHTS["smart_money"] * smart_s +
                WEIGHTS["concentration"] * conc_s +
                WEIGHTS["insider"] * (ins_s - 50)  # Center insider around 0
            )
            composite = _clamp(composite)

            # Confidence: how many signal sources had non-zero data
            sources = [obi_s, vol_s, bias_s, mom_s, smart_s, conc_s]
            active_sources = sum(1 for s in sources if abs(s) > 1.0)
            confidence = active_sources / len(sources)

            components_json = json.dumps({
                "obi": round(obi_s, 2),
                "volume_anomaly": round(vol_s, 2),
                "large_trade_bias": round(bias_s, 2),
                "momentum": round(mom_s, 2),
                "smart_money": round(smart_s, 2),
                "concentration": round(conc_s, 2),
                "arbitrage": arb_f,
                "insider": round(ins_s, 2),
            })

            signal_rows.append([
                cid,                         # condition_id
                round(composite, 2),         # score
                round(confidence, 3),        # confidence
                components_json,             # components JSON
                round(obi_s, 2),            # obi_score
                round(vol_s, 2),            # volume_score
                round(bias_s, 2),           # trade_bias_score
                round(mom_s, 2),            # momentum_score
                round(smart_s, 2),          # smart_money_score
                round(conc_s, 2),           # concentration_score
                arb_f,                       # arbitrage_flag
                round(ins_s, 2),            # insider_activity
                now,                         # computed_at
            ])

        if signal_rows:
            await writer.write_composite_signals(signal_rows)

        await writer.flush_all()

        logger.info(
            "signal_compositor_complete",
            extra={
                "markets_scored": len(signal_rows),
            },
        )

    except Exception:
        logger.error("signal_compositor_error", exc_info=True)
