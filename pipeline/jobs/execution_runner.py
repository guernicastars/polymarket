"""Job: execution cycle — read signals, apply risk checks, place orders.

This is the critical loop that turns predictions into positions. On each
cycle it:

1. Reads the latest composite signals and GNN predictions from ClickHouse
2. Combines them via the signal ensemble (quality-weighted)
3. For each signal with sufficient edge:
   a. Computes position size via fractional Kelly criterion
   b. Runs pre-trade risk checks (drawdown, exposure, liquidity)
   c. Places the order through the execution engine
   d. Records the fill in the position manager
4. Marks all open positions to market
5. Checks for positions to close (edge reversed, stop-loss, take-profit)
6. Writes execution state to ClickHouse for monitoring

The cycle runs every EXECUTION_INTERVAL seconds (default 2 minutes).
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import clickhouse_connect

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    EXECUTION_DRY_RUN,
    EXECUTION_FUNDER_ADDRESS,
    EXECUTION_INITIAL_CAPITAL,
    EXECUTION_PRIVATE_KEY,
    RISK_KELLY_FRACTION,
    RISK_MIN_EDGE,
)
from pipeline.execution.engine import ExecutionEngine, OrderRequest, OrderStatus
from pipeline.execution.position_manager import PositionManager
from pipeline.execution.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons (initialized on first run)
# ---------------------------------------------------------------------------

_engine: Optional[ExecutionEngine] = None
_position_manager: Optional[PositionManager] = None
_risk_manager: Optional[RiskManager] = None


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    """Create a read-only ClickHouse client."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


def _kelly_fraction(p_model: float, p_market: float) -> float:
    """Fractional Kelly criterion for binary prediction markets.

    Returns a position size fraction (0 to 1) representing the optimal
    fraction of capital to risk. Uses quarter-Kelly for safety.
    """
    if p_market <= 0.01 or p_market >= 0.99:
        return 0.0

    if p_model > p_market:
        b = (1.0 / p_market) - 1.0
        q = 1.0 - p_model
        f = (p_model * b - q) / b if b > 0 else 0.0
    else:
        p_no = 1.0 - p_model
        p_no_market = 1.0 - p_market
        b = (1.0 / p_no_market) - 1.0
        q = 1.0 - p_no
        f = (p_no * b - q) / b if b > 0 else 0.0

    return max(f, 0.0) * RISK_KELLY_FRACTION


def _get_instances() -> tuple[ExecutionEngine, PositionManager, RiskManager]:
    """Lazy-initialize the execution singletons."""
    global _engine, _position_manager, _risk_manager

    if _engine is None:
        _engine = ExecutionEngine(
            private_key=EXECUTION_PRIVATE_KEY,
            funder_address=EXECUTION_FUNDER_ADDRESS,
            dry_run=EXECUTION_DRY_RUN,
        )
    if _position_manager is None:
        _position_manager = PositionManager(initial_capital=EXECUTION_INITIAL_CAPITAL)
    if _risk_manager is None:
        _risk_manager = RiskManager(_position_manager)

    return _engine, _position_manager, _risk_manager


# ---------------------------------------------------------------------------
# Main execution cycle
# ---------------------------------------------------------------------------


async def run_execution_cycle() -> None:
    """Execute one trading cycle: signals → risk checks → orders → positions.

    This function is called by the scheduler every EXECUTION_INTERVAL seconds.
    """
    engine, pm, rm = _get_instances()
    await engine.initialize()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        client = await asyncio.to_thread(_get_read_client)

        # ---------------------------------------------------------------
        # 1. Read latest composite signals (from signal_compositor job)
        # ---------------------------------------------------------------
        signals_result = await asyncio.to_thread(
            client.query,
            """
            SELECT
                cs.condition_id,
                cs.score,
                cs.confidence,
                cs.obi_score,
                cs.volume_score,
                cs.trade_bias_score,
                cs.momentum_score,
                cs.smart_money_score,
                cs.concentration_score,
                cs.arbitrage_flag,
                cs.insider_activity,
                m.last_trade_price,
                m.volume_24h,
                m.liquidity
            FROM composite_signals cs
            INNER JOIN (
                SELECT condition_id, max(computed_at) AS latest
                FROM composite_signals
                WHERE computed_at >= now() - INTERVAL 10 MINUTE
                GROUP BY condition_id
            ) latest ON cs.condition_id = latest.condition_id
                AND cs.computed_at = latest.latest
            INNER JOIN (
                SELECT condition_id, last_trade_price, volume_24h, liquidity
                FROM markets FINAL
                WHERE active = 1 AND closed = 0
            ) m ON cs.condition_id = m.condition_id
            WHERE cs.confidence >= 0.3
            ORDER BY abs(cs.score) DESC
            LIMIT 100
            """,
        )

        # ---------------------------------------------------------------
        # 2. Read latest Bayesian predictions (from bayesian_runner job)
        #    Falls back to GNN predictions if Bayesian layer unavailable
        # ---------------------------------------------------------------
        bayesian_preds: dict[str, dict] = {}
        try:
            bayesian_result = await asyncio.to_thread(
                client.query,
                """
                SELECT
                    condition_id,
                    posterior_mean,
                    market_price,
                    edge,
                    direction,
                    kelly_fraction,
                    confidence,
                    n_evidence_sources,
                    evidence_agreement
                FROM bayesian_predictions FINAL
                WHERE predicted_at >= now() - INTERVAL 5 MINUTE
                ORDER BY predicted_at DESC
                LIMIT 1 BY condition_id
                LIMIT 200
                """,
            )
            for row in bayesian_result.result_rows:
                bayesian_preds[row[0]] = {
                    "prob": float(row[1]),
                    "market_price": float(row[2]),
                    "edge": float(row[3]),
                    "direction": row[4],
                    "kelly": float(row[5]),
                    "confidence": float(row[6]),
                    "n_sources": int(row[7]),
                    "agreement": float(row[8]),
                }
        except Exception:
            logger.debug("bayesian_predictions_unavailable")

        # Fallback: read raw GNN predictions if Bayesian layer empty
        gnn_preds: dict[str, dict] = {}
        if not bayesian_preds:
            try:
                gnn_result = await asyncio.to_thread(
                    client.query,
                    """
                    SELECT
                        condition_id,
                        calibrated_prob,
                        market_price,
                        edge,
                        direction,
                        kelly_fraction,
                        position_size_usd,
                        confidence
                    FROM gnn_predictions
                    WHERE predicted_at >= now() - INTERVAL 15 MINUTE
                    ORDER BY predicted_at DESC
                    LIMIT 100
                    """,
                )
                for row in gnn_result.result_rows:
                    cid = row[0]
                    if cid not in gnn_preds:
                        gnn_preds[cid] = {
                            "prob": float(row[1]),
                            "market_price": float(row[2]),
                            "edge": float(row[3]),
                            "direction": row[4],
                            "kelly": float(row[5]),
                            "size_usd": float(row[6]),
                            "confidence": float(row[7]),
                        }
            except Exception:
                logger.debug("gnn_predictions_unavailable")

        # ---------------------------------------------------------------
        # 3. Resolve token IDs for markets we want to trade
        # ---------------------------------------------------------------
        all_cids = list({row[0] for row in signals_result.result_rows})
        if not all_cids:
            logger.debug("execution_skip", extra={"reason": "no_signals"})
            return

        token_result = await asyncio.to_thread(
            client.query,
            f"""
            SELECT condition_id, tokens_yes_id, tokens_no_id, tick_size, neg_risk
            FROM markets FINAL
            WHERE condition_id IN ({','.join(f"'{c}'" for c in all_cids)})
            """,
        )

        token_map: dict[str, dict] = {}
        for row in token_result.result_rows:
            token_map[row[0]] = {
                "yes_token": row[1] if row[1] else "",
                "no_token": row[2] if row[2] else "",
                "tick_size": str(row[3]) if row[3] else "0.01",
                "neg_risk": bool(row[4]) if row[4] else False,
            }

        # ---------------------------------------------------------------
        # 4. For each signal, decide whether to trade
        # ---------------------------------------------------------------
        orders_placed = 0
        orders_rejected = 0

        for row in signals_result.result_rows:
            cid = row[0]
            composite_score = float(row[1])
            confidence = float(row[2])
            market_price = float(row[11])
            liquidity = float(row[13])

            if cid not in token_map:
                continue

            tokens = token_map[cid]
            if not tokens["yes_token"]:
                continue

            # Use Bayesian posterior if available, else fall back
            bayesian = bayesian_preds.get(cid)
            gnn = gnn_preds.get(cid)

            if bayesian and bayesian["confidence"] > 0.1:
                # Bayesian combiner already merged GNN + composite + market
                model_prob = bayesian["prob"]
                edge = bayesian["edge"]
                signal_source = "bayesian"
            elif gnn and gnn["confidence"] > 0.3:
                # Fallback: raw GNN prediction
                model_prob = gnn["prob"]
                edge = gnn["edge"]
                signal_source = "gnn"
            else:
                # Fallback: composite signal only
                model_prob = _composite_to_prob(composite_score, market_price)
                edge = model_prob - market_price
                signal_source = "composite"

            abs_edge = abs(edge)

            if abs_edge < RISK_MIN_EDGE:
                continue

            # Direction and token selection
            if edge > 0:
                side = "BUY"
                token_id = tokens["yes_token"]
                price = market_price
            else:
                side = "BUY"
                token_id = tokens["no_token"]
                price = 1.0 - market_price
                model_prob = 1.0 - model_prob

            # Kelly sizing — use Bayesian kelly if available
            if bayesian and signal_source == "bayesian" and bayesian["kelly"] > 0:
                kf = bayesian["kelly"]
            else:
                kf = _kelly_fraction(model_prob, price)
            if kf <= 0:
                continue

            size_usd = pm.capital * kf
            size_tokens = size_usd / price if price > 0 else 0

            if size_usd < 5.0:  # minimum $5 order
                continue

            # Build order request
            request = OrderRequest(
                condition_id=cid,
                token_id=token_id,
                side=side,
                price=round(price, 2),
                size=round(size_tokens, 2),
                signal_source=signal_source,
                edge=round(edge, 4),
                kelly_fraction=round(kf, 4),
                confidence=round(confidence, 3),
                tick_size=tokens["tick_size"],
                neg_risk=tokens["neg_risk"],
            )

            # Risk check
            check = rm.check_order(request, liquidity=liquidity)

            if not check.approved:
                logger.debug(
                    "order_rejected_risk",
                    extra={
                        "condition_id": cid[:12],
                        "violation": check.violation.value if check.violation else "",
                        "message": check.message,
                    },
                )
                orders_rejected += 1
                continue

            # Adjust size if risk manager capped it
            if check.adjusted_size and check.adjusted_size < request.size:
                request.size = round(check.adjusted_size, 2)

            # Place order
            result = await engine.place_order(request)

            # Persist order to ClickHouse (always — including DRY_RUN)
            order_row = [
                result.order_id,
                request.condition_id,
                request.token_id,
                request.side,
                request.price,
                request.size,
                request.edge,
                request.kelly_fraction,
                request.confidence,
                request.signal_source,
                result.status.value,
                result.error_msg,
                result.fill_price,
                result.filled_size,
                result.latency_ms,
                result.submitted_at or now,
                now,
            ]
            await writer.write("execution_orders", [order_row])

            if result.status in (
                OrderStatus.MATCHED,
                OrderStatus.LIVE,
                OrderStatus.DRY_RUN,
            ):
                pm.record_fill(result)
                orders_placed += 1

        # ---------------------------------------------------------------
        # 5. Mark open positions to market
        # ---------------------------------------------------------------
        if pm.positions:
            pos_cids = list(pm.positions.keys())
            price_result = await asyncio.to_thread(
                client.query,
                f"""
                SELECT condition_id, last_trade_price
                FROM markets FINAL
                WHERE condition_id IN ({','.join(f"'{c}'" for c in pos_cids)})
                """,
            )
            prices = {row[0]: float(row[1]) for row in price_result.result_rows}
            pm.mark_to_market(prices)

        # ---------------------------------------------------------------
        # 6. Check for positions to close (edge reversed)
        # ---------------------------------------------------------------
        positions_closed = 0
        for cid in list(pm.positions.keys()):
            pos = pm.positions[cid]
            # Find current signal for this position
            current_signal = None
            for row in signals_result.result_rows:
                if row[0] == cid:
                    current_signal = float(row[1])
                    break

            should_close = False
            reason = ""

            # Close if signal has flipped
            if current_signal is not None:
                if pos.side == "BUY" and current_signal < -20:
                    should_close = True
                    reason = "signal_reversed"
                elif pos.side == "SELL" and current_signal > 20:
                    should_close = True
                    reason = "signal_reversed"

            # Close if stop-loss hit (position down > 20%)
            if pos.pnl_pct < -0.20:
                should_close = True
                reason = "stop_loss"

            # Close if take-profit hit (position up > 50%)
            if pos.pnl_pct > 0.50:
                should_close = True
                reason = "take_profit"

            if should_close:
                realized = pm.close_position(cid, pos.current_price)
                rm.record_realized_pnl(realized)
                positions_closed += 1
                logger.info(
                    "position_exit",
                    extra={
                        "condition_id": cid[:12],
                        "reason": reason,
                        "realized_pnl": round(realized, 2),
                    },
                )

        # ---------------------------------------------------------------
        # 7. Persist positions, take snapshot, and log
        # ---------------------------------------------------------------
        # Write current positions to ClickHouse
        position_rows = pm.to_rows()
        if position_rows:
            await writer.write("execution_positions", position_rows)

        snapshot = pm.snapshot()

        # Write portfolio snapshot to ClickHouse
        snapshot_row = [
            snapshot.timestamp,
            snapshot.capital,
            snapshot.total_value,
            snapshot.n_positions,
            snapshot.total_unrealized_pnl,
            snapshot.total_realized_pnl,
            snapshot.total_cost_basis,
            snapshot.max_position_value,
            snapshot.high_water_mark,
            snapshot.current_drawdown,
            "DRY_RUN" if engine.dry_run else "LIVE",
        ]
        await writer.write("execution_snapshots", [snapshot_row])

        await writer.flush_all()

        logger.info(
            "execution_cycle_complete",
            extra={
                "orders_placed": orders_placed,
                "orders_rejected": orders_rejected,
                "positions_closed": positions_closed,
                "n_positions": snapshot.n_positions,
                "capital": round(snapshot.capital, 2),
                "total_value": round(snapshot.total_value, 2),
                "drawdown": round(snapshot.current_drawdown, 4),
                "daily_pnl": round(rm.daily_realized_pnl, 2),
                "mode": "DRY_RUN" if engine.dry_run else "LIVE",
            },
        )

    except Exception:
        logger.error("execution_cycle_error", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composite_to_prob(score: float, market_price: float) -> float:
    """Convert composite signal score (-100..+100) to probability estimate.

    Uses the market price as anchor and shifts it based on the signal
    strength. A score of +100 shifts the probability up by ~15% from market,
    -100 shifts down by ~15%.

    Args:
        score: Composite signal score (-100 to +100).
        market_price: Current market price (0 to 1).

    Returns:
        Estimated true probability (0.01 to 0.99).
    """
    # Signal shifts probability by up to 15% of remaining range
    if score > 0:
        shift = (score / 100.0) * 0.15 * (1.0 - market_price)
    else:
        shift = (score / 100.0) * 0.15 * market_price

    prob = market_price + shift
    return max(0.01, min(0.99, prob))


# ---------------------------------------------------------------------------
# Accessors for scheduler / health check
# ---------------------------------------------------------------------------


def get_execution_status() -> dict:
    """Return execution layer status for health checks."""
    _, pm, rm = _get_instances()
    status = rm.status()
    status["mode"] = "DRY_RUN" if EXECUTION_DRY_RUN else "LIVE"
    return status
