"""Pipeline configuration loaded from environment variables."""

import logging
import os
import sys

from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# ClickHouse connection
# ---------------------------------------------------------------------------
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "polymarket")

# ---------------------------------------------------------------------------
# Polymarket API base URLs
# ---------------------------------------------------------------------------
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# ---------------------------------------------------------------------------
# Polling intervals (seconds)
# ---------------------------------------------------------------------------
MARKET_SYNC_INTERVAL = 300       # 5 minutes
PRICE_POLL_INTERVAL = 30         # 30 seconds
TRADE_COLLECT_INTERVAL = 60      # 1 minute
ORDERBOOK_INTERVAL = 60          # 1 minute

# ---------------------------------------------------------------------------
# Writer / buffer settings
# ---------------------------------------------------------------------------
BUFFER_FLUSH_SIZE = 10_000       # Flush after this many rows
BUFFER_FLUSH_INTERVAL = 10.0     # Flush after this many seconds
WRITER_MAX_RETRIES = 3
WRITER_BASE_BACKOFF = 1.0        # Seconds, doubles per retry

# ---------------------------------------------------------------------------
# API client settings
# ---------------------------------------------------------------------------
PRICE_BATCH_SIZE = 50            # Tokens per POST /prices call
ORDERBOOK_TOP_N = 100            # Top N markets by volume for orderbook snapshots
WS_MAX_TOKENS_PER_CONN = 500    # Max instruments per WebSocket connection
WS_RECONNECT_BASE_DELAY = 1.0   # Seconds, doubles per retry
WS_RECONNECT_MAX_DELAY = 60.0
HTTP_TIMEOUT = 30.0              # httpx timeout in seconds

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTH_CHECK_PORT = int(os.environ.get("HEALTH_CHECK_PORT", "8080"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure structured JSON logging."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("clickhouse_connect").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
