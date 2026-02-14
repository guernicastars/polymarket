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
    HEALTH_CHECK_PORT,
    MARKET_SYNC_INTERVAL,
    ORDERBOOK_INTERVAL,
    PRICE_POLL_INTERVAL,
    TRADE_COLLECT_INTERVAL,
)
from pipeline.api.ws_client import WebSocketClient
from pipeline.jobs.market_sync import run_market_sync
from pipeline.jobs.orderbook_snapshot import run_orderbook_snapshot
from pipeline.jobs.price_poller import run_price_poller
from pipeline.jobs.trade_collector import run_trade_collector

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

        # Start WebSocket listener
        self._ws_client = WebSocketClient(callback=self._ws_callback)
        if self._active_token_ids:
            await self._ws_client.start(self._active_token_ids)

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
                if self._ws_client:
                    await self._ws_client.update_subscriptions(token_ids)
                logger.info(
                    "token_list_updated",
                    extra={"count": len(token_ids)},
                )
        except Exception:
            logger.error("market_sync_error", exc_info=True)

    async def _job_price_poller(self) -> None:
        try:
            await run_price_poller(self._active_token_ids)
        except Exception:
            logger.error("price_poller_error", exc_info=True)

    async def _job_trade_collector(self) -> None:
        try:
            await run_trade_collector()
        except Exception:
            logger.error("trade_collector_error", exc_info=True)

    async def _job_orderbook_snapshot(self) -> None:
        try:
            await run_orderbook_snapshot(self._active_token_ids)
        except Exception:
            logger.error("orderbook_snapshot_error", exc_info=True)

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
        return web.json_response({
            "status": "ok",
            "active_tokens": len(self._active_token_ids),
            "scheduler_running": self._scheduler.running,
        })
