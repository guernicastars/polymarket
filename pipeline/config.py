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

# Phase 4: Market similarity graph
SIMILARITY_SCORER_INTERVAL = 3600      # 1 hour
SIMILARITY_TOP_MARKETS = 500           # Top N markets for similarity computation
SIMILARITY_MIN_WEIGHT = 0.15           # Minimum similarity to keep an edge

# ---------------------------------------------------------------------------
# Phase 4: Force-include tokens (Ukraine network model targets)
# These tokens are always polled regardless of volume ranking.
# ---------------------------------------------------------------------------
FORCE_INCLUDE_TOKEN_IDS = [
    # Pokrovsk (March 31) — Yes + No tokens
    "66269306641072351636961565307986748333577224262245957192572668300728882961637",
    "84921298575469347580540454437533951955834830815691221822388171560096631172626",
    # Kostyantynivka (Dec 31, 2026)
    "83552904656813968939383082097054433404653657244784709614448703928529504455469",
    "47029152085101973226255505664149277121974669906212969183902148537312535402801",
    # Toretske (March 31, 2026)
    "109615118914262848726134459562538492202453288986973576852948325569038771941384",
    "43421958307449721647059425840567504352004753118032821157855035393935581214602",
    # Kupiansk (March 31)
    "45089382414483578938726883454769004518004044065888676737276582661191075840434",
    "21827338396710725624553757329770437895248298147118091304039965270238613692151",
    # Lyman (March 31, 2026)
    "34254338241290516754163779889987574790033932358499741518448025208727830568442",
    "58719878550625298503107256203979685198416062590647075610592705883102153544406",
    # Orikhiv (March 31)
    "18383110985753553577320023911481935085081650068070789350086929125763745346449",
    "12525252575110381972187982414817603396721377689110909558525035146809321485502",
    # Hryshyne (March 31, 2026)
    "18730978422362583344635704599452590050275452404041361625425790481762026934944",
    "10253671204995955855444782611640888907645658450627326046642066563302828880619",
    # Huliaipole (February 28)
    "28667860392142823151877400432052515597411262182770824014128139317797432047906",
    "113701124335488338863940974262884472507110377142235791753359365110404680612694",
    # Sloviansk (June 30)
    "35387107368752328644982641185985574345241154905795306214146121961491417818763",
    "83019259287355956702470381078770351410935779073564095064662359996647132628409",
    # Borova (March 31)
    "77236426235125758956565839580730454202569477043200936799693583110955110463130",
    "54683957912530266109779005719731303044583126915152694606057917247853053555828",
]

# ---------------------------------------------------------------------------
# Phase 4: News tracking and market microstructure
# ---------------------------------------------------------------------------
NEWS_TRACKER_INTERVAL = 600             # 10 minutes
MICROSTRUCTURE_INTERVAL = 60            # 60 seconds

# ---------------------------------------------------------------------------
# Phase 5: Execution layer
# ---------------------------------------------------------------------------
EXECUTION_DRY_RUN = os.environ.get("EXECUTION_DRY_RUN", "true").lower() == "true"
EXECUTION_PRIVATE_KEY = os.environ.get("EXECUTION_PRIVATE_KEY", "")
EXECUTION_FUNDER_ADDRESS = os.environ.get("EXECUTION_FUNDER_ADDRESS", "")
EXECUTION_CHAIN_ID = int(os.environ.get("EXECUTION_CHAIN_ID", "137"))  # Polygon
EXECUTION_SIGNATURE_TYPE = int(os.environ.get("EXECUTION_SIGNATURE_TYPE", "0"))  # 0=EOA
EXECUTION_INTERVAL = 120               # 2 minutes — execution cycle
EXECUTION_INITIAL_CAPITAL = float(os.environ.get("EXECUTION_INITIAL_CAPITAL", "1000"))

# Risk management
RISK_MAX_DRAWDOWN_PCT = 0.15            # Halt trading at 15% drawdown from HWM
RISK_MAX_POSITION_EXPOSURE_PCT = 0.10   # Max 10% of portfolio per market
RISK_MAX_PORTFOLIO_EXPOSURE_PCT = 0.60  # Max 60% of portfolio deployed
RISK_MAX_CONCURRENT_POSITIONS = 20      # Max 20 simultaneous positions
RISK_MIN_EDGE = 0.03                    # Min 3% edge to trade (after spread)
RISK_MIN_LIQUIDITY = 5_000.0            # Min $5K market liquidity
RISK_DAILY_LOSS_LIMIT_PCT = 0.05        # Halt after 5% daily loss
RISK_KELLY_FRACTION = 0.25              # Quarter Kelly for safety

# ---------------------------------------------------------------------------
# Phase 6: Bayesian prediction layer (two-layer architecture)
# ---------------------------------------------------------------------------

# Layer 1: Online GNN-TCN
ONLINE_GNN_UPDATE_INTERVAL = 900        # 15 minutes — incremental SGD
ONLINE_GNN_PREDICT_INTERVAL = 300       # 5 minutes — predictions

# Layer 2: Bayesian combiner
BAYESIAN_UPDATE_INTERVAL = 120          # 2 minutes — run Bayesian update
BAYESIAN_CALIBRATION_INTERVAL = 3600    # 1 hour — flush calibration metrics
BAYESIAN_PRIOR_STRENGTH = 20.0          # Concentration of market-price prior
BAYESIAN_MARKET_EFFICIENCY = 0.85       # How much to trust market (0-1)
BAYESIAN_DECAY_HALFLIFE = 4.0           # Hours: posterior decays to market price
BAYESIAN_TOP_MARKETS = 500              # Markets to score per cycle

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
