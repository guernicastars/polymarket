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
PRICE_BATCH_SIZE = 100           # Tokens per POST /prices call
PRICE_POLL_MAX_TOKENS = 1000    # Only poll top N tokens (by volume) for prices
ORDERBOOK_TOP_N = 100            # Top N markets by volume for orderbook snapshots
WS_MAX_TOKENS_PER_CONN = 500    # Max instruments per WebSocket connection
WS_MAX_TOTAL_TOKENS = 5000      # Limit WebSocket to top N tokens by volume
WS_RECONNECT_BASE_DELAY = 1.0   # Seconds, doubles per retry
WS_RECONNECT_MAX_DELAY = 60.0
HTTP_TIMEOUT = 30.0              # httpx timeout in seconds

# ---------------------------------------------------------------------------
# Phase 2: User/wallet data polling intervals
# ---------------------------------------------------------------------------
LEADERBOARD_SYNC_INTERVAL = 3600       # 1 hour
HOLDER_SYNC_INTERVAL = 900              # 15 minutes
POSITION_SYNC_INTERVAL = 300            # 5 minutes
PROFILE_ENRICH_INTERVAL = 600           # 10 minutes (batch enrichment cycle)

# Phase 2: Tuning
LEADERBOARD_MAX_RESULTS = 200           # Top N per category/period/order combo
HOLDER_SYNC_TOP_MARKETS = 50            # Top N markets by volume for holder tracking
TRACKED_WALLET_MAX = 500                # Max wallets to track positions for
PROFILE_BATCH_SIZE = 20                 # Wallets per profile enrichment cycle

# ---------------------------------------------------------------------------
# Phase 3: Advanced analytics intervals
# ---------------------------------------------------------------------------
ARBITRAGE_SCAN_INTERVAL = 120           # 2 minutes
WALLET_ANALYZE_INTERVAL = 1800          # 30 minutes
SIGNAL_COMPOSITE_INTERVAL = 300         # 5 minutes

# Phase 3: Tuning
ARB_FEE_THRESHOLD = 0.02               # Min |sum - 1.0| to flag as arbitrage
ARB_RELATED_MARKET_THRESHOLD = 0.05    # Min pricing inconsistency for related markets
CLUSTER_TIME_WINDOW = 60               # Seconds: trades within this window are "synchronized"
CLUSTER_MIN_OVERLAP = 3                # Min shared markets to consider clustering
CLUSTER_MIN_SIMILARITY = 0.6           # Min similarity score to form a cluster
INSIDER_FRESHNESS_DAYS = 30            # Wallet age below this is "fresh"
INSIDER_WIN_RATE_THRESHOLD = 0.75      # Win rate above this in niche markets is suspicious
COMPOSITE_TOP_MARKETS = 500            # Compute composite signals for top N markets by volume

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
