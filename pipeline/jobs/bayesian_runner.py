"""Job: run Bayesian combiner update cycle for all active markets.

Called every BAYESIAN_UPDATE_INTERVAL (default 2 minutes).

Flow:
  1. Read latest GNN predictions from gnn_predictions table
  2. Read latest composite signals from composite_signals table
  3. Read current market prices
  4. For each market with sufficient signal coverage:
     a. Convert signals to likelihood ratios via adapters
     b. Run BayesianCombiner.update() to produce posterior
     c. Write BayesianPrediction to bayesian_predictions table
  5. Sync posterior state to ClickHouse
  6. Update calibration tracker with any newly resolved markets
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import clickhouse_connect

from pipeline.bayesian.combiner import BayesianCombiner
from pipeline.bayesian.evidence import (
    CompositeSignalAdapter,
    GNNEvidenceAdapter,
)
from pipeline.bayesian.state import PosteriorStateStore
from pipeline.bayesian.calibration import CalibrationTracker
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    BAYESIAN_PRIOR_STRENGTH,
    BAYESIAN_MARKET_EFFICIENCY,
    BAYESIAN_DECAY_HALFLIFE,
    BAYESIAN_TOP_MARKETS,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)

# Module-level singletons (initialized on first run)
_combiner: BayesianCombiner | None = None
_state_store: PosteriorStateStore | None = None
_calibration: CalibrationTracker | None = None
_gnn_adapter: GNNEvidenceAdapter | None = None
_composite_adapter: CompositeSignalAdapter | None = None
_initialized = False


def _get_read_client() -> clickhouse_connect.driver.client.Client:
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


async def _initialize() -> None:
    """Initialize singletons on first run."""
    global _combiner, _state_store, _calibration
    global _gnn_adapter, _composite_adapter, _initialized

    _combiner = BayesianCombiner(
        prior_strength=BAYESIAN_PRIOR_STRENGTH,
        market_efficiency=BAYESIAN_MARKET_EFFICIENCY,
        decay_halflife_hours=BAYESIAN_DECAY_HALFLIFE,
    )
    client = _get_read_client()
    _state_store = PosteriorStateStore(client)
    await _state_store.load_state()

    _calibration = CalibrationTracker(buffer_size=2000)
    _gnn_adapter = GNNEvidenceAdapter(reliability=0.6, calibration_shrinkage=0.5)
    _composite_adapter = CompositeSignalAdapter(reliability=0.4, temperature=50.0)
    _initialized = True

    logger.info(
        "Bayesian runner initialized: %d cached posteriors",
        _state_store.active_markets,
    )


async def run_bayesian_update() -> None:
    """Execute one Bayesian update cycle."""
    global _initialized

    if not _initialized:
        await _initialize()

    writer = ClickHouseWriter.get_instance()
    client = _get_read_client()
    now = datetime.now(timezone.utc)

    try:
        # 1. Read latest GNN predictions (last 10 min)
        gnn_preds = await asyncio.to_thread(
            client.query,
            f"""SELECT condition_id, calibrated_prob, confidence, market_price
                FROM gnn_predictions FINAL
                WHERE predicted_at > now() - INTERVAL 10 MINUTE
                ORDER BY predicted_at DESC
                LIMIT 1 BY condition_id""",
        )
        gnn_map: dict[str, dict] = {}
        for row in gnn_preds.result_rows:
            gnn_map[row[0]] = {
                "prob": float(row[1]),
                "confidence": float(row[2]),
                "market_price": float(row[3]),
            }

        # 2. Read latest composite signals
        composite = await asyncio.to_thread(
            client.query,
            f"""SELECT condition_id, score, confidence
                FROM composite_signals FINAL
                ORDER BY computed_at DESC
                LIMIT 1 BY condition_id
                LIMIT {BAYESIAN_TOP_MARKETS}""",
        )
        composite_map: dict[str, dict] = {}
        for row in composite.result_rows:
            composite_map[row[0]] = {
                "score": float(row[1]),
                "confidence": float(row[2]),
            }

        # 3. Read current market prices for all markets
        all_cids = set(gnn_map.keys()) | set(composite_map.keys())
        if not all_cids:
            logger.info("No signals available for Bayesian update")
            return

        price_rows = await asyncio.to_thread(
            client.query,
            f"""SELECT condition_id, price
                FROM market_latest_price FINAL
                WHERE outcome = 'Yes'""",
        )
        prices: dict[str, float] = {}
        for row in price_rows.result_rows:
            prices[row[0]] = float(row[1])

        # 4. Run Bayesian update per market
        prediction_rows = []
        for cid in all_cids:
            mkt_price = prices.get(cid, 0.5)
            if mkt_price <= 0.01 or mkt_price >= 0.99:
                continue  # Skip extreme markets

            evidence = []

            # GNN evidence
            gnn = gnn_map.get(cid)
            if gnn:
                ev = _gnn_adapter.to_evidence(
                    calibrated_prob=gnn["prob"],
                    model_uncertainty=1.0 - gnn["confidence"],
                    is_cold_start=False,
                )
                evidence.append(ev)

            # Composite evidence
            comp = composite_map.get(cid)
            if comp:
                ev = _composite_adapter.to_evidence(
                    score=comp["score"],
                    signal_confidence=comp["confidence"],
                )
                evidence.append(ev)

            # Get existing posterior
            existing = _state_store.get(cid)

            # Run Bayesian update
            pred = _combiner.update(
                condition_id=cid,
                evidence=evidence,
                current_market_price=mkt_price,
                existing_posterior=existing,
            )

            # Update state store
            from pipeline.bayesian.combiner import BetaPosterior

            _state_store.set(cid, BetaPosterior(
                alpha=pred.posterior_alpha,
                beta=pred.posterior_beta,
                last_updated=now,
                n_updates=existing.n_updates + 1 if existing else 1,
            ))

            # Record for calibration
            _calibration.record(
                condition_id=cid,
                source="bayesian",
                predicted_prob=pred.posterior_mean,
                market_price=mkt_price,
            )

            # Build row for ClickHouse
            prediction_rows.append([
                cid,
                pred.posterior_mean,
                pred.posterior_alpha,
                pred.posterior_beta,
                pred.credible_lo,
                pred.credible_hi,
                mkt_price,
                pred.edge,
                pred.confidence,
                pred.n_evidence_sources,
                pred.evidence_agreement,
                pred.direction,
                pred.kelly_fraction,
                0.0,  # position_size_usd (computed by execution)
                pred.evidence_detail,
                now,
            ])

        # 5. Write predictions
        if prediction_rows:
            await writer.write("bayesian_predictions", prediction_rows)
            await writer.flush_all()

        # 6. Sync posterior state
        await _state_store.sync_to_clickhouse(writer)

        logger.info(
            "Bayesian update: %d markets scored, %d with GNN, %d with composite",
            len(prediction_rows),
            len(gnn_map),
            len(composite_map),
        )

    except Exception:
        logger.error("bayesian_update_error", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def run_calibration_flush() -> None:
    """Flush calibration metrics to ClickHouse. Called hourly."""
    if not _initialized or _calibration is None:
        return

    writer = ClickHouseWriter.get_instance()
    _calibration.recompute_adjustments()
    await _calibration.write_to_clickhouse(writer)

    # Log comparison if available
    comparison = _calibration.brier_score_vs_market()
    if comparison:
        model_bs, market_bs = comparison
        better = "BETTER" if model_bs < market_bs else "WORSE"
        logger.info(
            "Calibration: model Brier=%.4f vs market Brier=%.4f → %s",
            model_bs, market_bs, better,
        )
