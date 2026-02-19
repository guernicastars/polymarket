"""APScheduler-based job scheduler for the Polymarket pipeline."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import (
    ARBITRAGE_SCAN_INTERVAL,
    EXECUTION_INTERVAL,
    FORCE_INCLUDE_TOKEN_IDS,
    HEALTH_CHECK_PORT,
    HOLDER_SYNC_INTERVAL,
    LEADERBOARD_SYNC_INTERVAL,
    MARKET_SYNC_INTERVAL,
    MICROSTRUCTURE_INTERVAL,
    NEWS_TRACKER_INTERVAL,
    ORDERBOOK_INTERVAL,
    POSITION_SYNC_INTERVAL,
    PRICE_POLL_INTERVAL,
    PRICE_POLL_MAX_TOKENS,
    PROFILE_ENRICH_INTERVAL,
    SIGNAL_COMPOSITE_INTERVAL,
    SIMILARITY_SCORER_INTERVAL,
    TRADE_COLLECT_INTERVAL,
    WALLET_ANALYZE_INTERVAL,
    WS_MAX_TOTAL_TOKENS,
)
from pipeline.api.ws_client import WebSocketClient
from pipeline.jobs.market_sync import run_market_sync, active_condition_ids
from pipeline.jobs.orderbook_snapshot import run_orderbook_snapshot
from pipeline.jobs.price_poller import run_price_poller
from pipeline.jobs.trade_collector import run_trade_collector
from pipeline.jobs.leaderboard_sync import run_leaderboard_sync
from pipeline.jobs.holder_sync import run_holder_sync
from pipeline.jobs.position_sync import run_position_sync
from pipeline.jobs.profile_enricher import run_profile_enricher
from pipeline.jobs.arbitrage_scanner import run_arbitrage_scanner
from pipeline.jobs.wallet_analyzer import run_wallet_analyzer
from pipeline.jobs.signal_compositor import run_signal_compositor
from pipeline.jobs.news_runner import run_news_tracker, run_microstructure
from pipeline.jobs.similarity_scorer import run_similarity_scorer
from pipeline.jobs.execution_runner import run_execution_cycle, get_execution_status

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Orchestrates all pipeline jobs with APScheduler."""

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._active_token_ids: list[str] = []
        self._ws_client: WebSocketClient | None = None
        self._writer = ClickHouseWriter.get_instance()
        self._shutdown_event = asyncio.Event()
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Register jobs, start the scheduler, and block until shutdown."""
        # Initial market sync to populate token list
        logger.info("initial_market_sync")
        try:
            self._active_token_ids = await run_market_sync()
        except Exception:
            logger.error("initial_sync_failed", exc_info=True)
            self._active_token_ids = []

        # Initial leaderboard sync to seed tracked wallets
        logger.info("initial_leaderboard_sync")
        try:
            await run_leaderboard_sync()
        except Exception:
            logger.error("initial_leaderboard_sync_failed", exc_info=True)

        # Start WebSocket listener (limited to top tokens by volume)
        self._ws_client = WebSocketClient(callback=self._ws_callback)
        ws_tokens = self._active_token_ids[:WS_MAX_TOTAL_TOKENS]
        if ws_tokens:
            await self._ws_client.start(ws_tokens)
            logger.info(
                "ws_tokens_limited",
                extra={
                    "total_active": len(self._active_token_ids),
                    "ws_subscribed": len(ws_tokens),
                },
            )

        # Register scheduled jobs
        self._scheduler.add_job(
            self._job_market_sync,
            "interval",
            seconds=MARKET_SYNC_INTERVAL,
            id="market_sync",
            name="Market Sync",
        )
        self._scheduler.add_job(
            self._job_price_poller,
            "interval",
            seconds=PRICE_POLL_INTERVAL,
            id="price_poller",
            name="Price Poller",
        )
        self._scheduler.add_job(
            self._job_trade_collector,
            "interval",
            seconds=TRADE_COLLECT_INTERVAL,
            id="trade_collector",
            name="Trade Collector",
        )
        self._scheduler.add_job(
            self._job_orderbook_snapshot,
            "interval",
            seconds=ORDERBOOK_INTERVAL,
            id="orderbook_snapshot",
            name="Orderbook Snapshot",
        )
        # --- Phase 2 jobs ---
        self._scheduler.add_job(
            self._job_leaderboard_sync,
            "interval",
            seconds=LEADERBOARD_SYNC_INTERVAL,
            id="leaderboard_sync",
            name="Leaderboard Sync",
        )
        self._scheduler.add_job(
            self._job_holder_sync,
            "interval",
            seconds=HOLDER_SYNC_INTERVAL,
            id="holder_sync",
            name="Holder Sync",
        )
        self._scheduler.add_job(
            self._job_position_sync,
            "interval",
            seconds=POSITION_SYNC_INTERVAL,
            id="position_sync",
            name="Position Sync",
        )
        self._scheduler.add_job(
            self._job_profile_enricher,
            "interval",
            seconds=PROFILE_ENRICH_INTERVAL,
            id="profile_enricher",
            name="Profile Enricher",
        )

        # --- Phase 3 jobs ---
        self._scheduler.add_job(
            self._job_arbitrage_scanner,
            "interval",
            seconds=ARBITRAGE_SCAN_INTERVAL,
            id="arbitrage_scanner",
            name="Arbitrage Scanner",
        )
        self._scheduler.add_job(
            self._job_wallet_analyzer,
            "interval",
            seconds=WALLET_ANALYZE_INTERVAL,
            id="wallet_analyzer",
            name="Wallet Analyzer",
        )
        self._scheduler.add_job(
            self._job_signal_compositor,
            "interval",
            seconds=SIGNAL_COMPOSITE_INTERVAL,
            id="signal_compositor",
            name="Signal Compositor",
        )

        # --- Phase 4 jobs ---
        self._scheduler.add_job(
            self._job_news_tracker,
            "interval",
            seconds=NEWS_TRACKER_INTERVAL,
            id="news_tracker",
            name="News Tracker",
        )
        self._scheduler.add_job(
            self._job_microstructure,
            "interval",
            seconds=MICROSTRUCTURE_INTERVAL,
            id="microstructure",
            name="Microstructure Engine",
        )
        self._scheduler.add_job(
            self._job_similarity_scorer,
            "interval",
            seconds=SIMILARITY_SCORER_INTERVAL,
            id="similarity_scorer",
            name="Similarity Scorer",
        )

        # --- Phase 5: Execution ---
        self._scheduler.add_job(
            self._job_execution_cycle,
            "interval",
            seconds=EXECUTION_INTERVAL,
            id="execution_cycle",
            name="Execution Cycle",
        )

        # Periodic buffer flush for stale data
        self._scheduler.add_job(
            self._writer.flush_stale,
            "interval",
            seconds=5,
            id="buffer_flush",
            name="Buffer Flush",
        )

        self._scheduler.start()
        logger.info(
            "scheduler_started",
            extra={"active_tokens": len(self._active_token_ids)},
        )

        # Start health check server
        await self._start_health_server()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        # Block until shutdown
        await self._shutdown_event.wait()
        await self._stop()

    async def _stop(self) -> None:
        logger.info("scheduler_stopping")
        self._scheduler.shutdown(wait=False)

        if self._ws_client:
            await self._ws_client.stop()

        await self._writer.flush_all()

        if self._health_runner:
            await self._health_runner.cleanup()

        logger.info("scheduler_stopped")

    def _signal_handler(self) -> None:
        logger.info("shutdown_signal_received")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Job wrappers (catch exceptions so scheduler keeps running)
    # ------------------------------------------------------------------

    async def _job_market_sync(self) -> None:
        try:
            token_ids = await run_market_sync()
            if token_ids != self._active_token_ids:
                self._active_token_ids = token_ids
                ws_tokens = token_ids[:WS_MAX_TOTAL_TOKENS]
                if self._ws_client:
                    await self._ws_client.update_subscriptions(ws_tokens)
                logger.info(
                    "token_list_updated",
                    extra={
                        "total": len(token_ids),
                        "ws_subscribed": len(ws_tokens),
                    },
                )
        except Exception:
            logger.error("market_sync_error", exc_info=True)

    async def _job_price_poller(self) -> None:
        try:
            # Only poll top-N tokens; WS covers the rest in real-time
            poll_tokens = self._active_token_ids[:PRICE_POLL_MAX_TOKENS]
            # Merge force-include tokens (Ukraine network model targets)
            forced = [t for t in FORCE_INCLUDE_TOKEN_IDS if t not in poll_tokens]
            if forced:
                poll_tokens = poll_tokens + forced
            await run_price_poller(poll_tokens)
        except Exception:
            logger.error("price_poller_error", exc_info=True)

    async def _job_trade_collector(self) -> None:
        try:
            await run_trade_collector()
        except Exception:
            logger.error("trade_collector_error", exc_info=True)

    async def _job_orderbook_snapshot(self) -> None:
        try:
            # Merge force-include tokens for orderbook snapshots too
            all_tokens = self._active_token_ids
            forced = [t for t in FORCE_INCLUDE_TOKEN_IDS if t not in all_tokens]
            if forced:
                all_tokens = all_tokens + forced
            await run_orderbook_snapshot(all_tokens)
        except Exception:
            logger.error("orderbook_snapshot_error", exc_info=True)

    # --- Phase 2 job wrappers ---

    async def _job_leaderboard_sync(self) -> None:
        try:
            await run_leaderboard_sync()
        except Exception:
            logger.error("leaderboard_sync_error", exc_info=True)

    async def _job_holder_sync(self) -> None:
        try:
            await run_holder_sync(active_condition_ids)
        except Exception:
            logger.error("holder_sync_error", exc_info=True)

    async def _job_position_sync(self) -> None:
        try:
            await run_position_sync()
        except Exception:
            logger.error("position_sync_error", exc_info=True)

    async def _job_profile_enricher(self) -> None:
        try:
            await run_profile_enricher()
        except Exception:
            logger.error("profile_enricher_error", exc_info=True)

    # --- Phase 3 job wrappers ---

    async def _job_arbitrage_scanner(self) -> None:
        try:
            await run_arbitrage_scanner()
        except Exception:
            logger.error("arbitrage_scanner_error", exc_info=True)

    async def _job_wallet_analyzer(self) -> None:
        try:
            await run_wallet_analyzer()
        except Exception:
            logger.error("wallet_analyzer_error", exc_info=True)

    async def _job_signal_compositor(self) -> None:
        try:
            await run_signal_compositor()
        except Exception:
            logger.error("signal_compositor_error", exc_info=True)

    # --- Phase 4 job wrappers ---

    async def _job_news_tracker(self) -> None:
        try:
            await run_news_tracker()
        except Exception:
            logger.error("news_tracker_error", exc_info=True)

    async def _job_microstructure(self) -> None:
        try:
            await run_microstructure()
        except Exception:
            logger.error("microstructure_error", exc_info=True)

    async def _job_similarity_scorer(self) -> None:
        try:
            await run_similarity_scorer()
        except Exception:
            logger.error("similarity_scorer_error", exc_info=True)

    # --- Phase 5 job wrapper ---

    async def _job_execution_cycle(self) -> None:
        try:
            await run_execution_cycle()
        except Exception:
            logger.error("execution_cycle_error", exc_info=True)

    # ------------------------------------------------------------------
    # WebSocket callback
    # ------------------------------------------------------------------

    async def _ws_callback(self, table: str, rows: list[list[Any]]) -> None:
        await self._writer.write(table, rows)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _start_health_server(self) -> None:
        self._health_app = web.Application()
        self._health_app.router.add_get("/health", self._health_handler)

        self._health_runner = web.AppRunner(self._health_app)
        await self._health_runner.setup()
        site = web.TCPSite(self._health_runner, "0.0.0.0", HEALTH_CHECK_PORT)
        await site.start()
        logger.info("health_server_started", extra={"port": HEALTH_CHECK_PORT})

    async def _health_handler(self, request: web.Request) -> web.Response:
        from pipeline.jobs.leaderboard_sync import discovered_wallets
        try:
            exec_status = get_execution_status()
        except Exception:
            exec_status = {"error": "not_initialized"}
        return web.json_response({
            "status": "ok",
            "active_tokens": len(self._active_token_ids),
            "tracked_wallets": len(discovered_wallets),
            "scheduler_running": self._scheduler.running,
            "phase3_jobs": ["arbitrage_scanner", "wallet_analyzer", "signal_compositor"],
            "phase4_jobs": ["news_tracker", "microstructure", "similarity_scorer"],
            "phase5_execution": exec_status,
        })
