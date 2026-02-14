# Phase 2 Spec — User/Wallet Data System

> Single source of truth for all Phase 2 implementation. Every DDL statement, Python class/function signature, TypeScript type, query function, and component is defined here. Agents implement directly from this spec.

## Overview

Phase 2 adds **user/wallet-level data ingestion** to the existing Polymarket pipeline and a **Whales dashboard page** to the analytics UI. It builds on the Phase 1 signals infrastructure (orderbook imbalance, volume anomalies, large trades, technicals) by adding the "who" dimension — tracking which wallets are trading, their positions, rankings, and profiles.

### New Data Sources

All endpoints are **public, no authentication required**:

| Endpoint | API | Purpose |
|----------|-----|---------|
| `GET /v1/leaderboard` | Data API | Top traders by PnL/volume across categories |
| `GET /positions?user=` | Data API | Current open positions for a wallet |
| `GET /activity?user=` | Data API | Full trade/activity history for a wallet |
| `GET /holders?market=` | Data API | Top 20 holders per market |
| `GET /trades?user=` | Data API | Trade history for a wallet |
| `GET /value?user=` | Data API | Total portfolio value for a wallet |
| `GET /public-profile?address=` | Gamma API | Wallet profile (name, bio, image, X handle) |

---

## 1. ClickHouse Schema (DDL)

Add to `pipeline/schema/002_phase2_users.sql`. Run automatically on startup after `001_init.sql`.

### 1.1 `trader_rankings` — Leaderboard Snapshots

Stores periodic snapshots of the Polymarket leaderboard across categories and time periods. ReplacingMergeTree deduplicates by wallet+category+time_period+order_by, keeping the latest `snapshot_time`.

```sql
CREATE TABLE IF NOT EXISTS trader_rankings
(
    -- Identity
    proxy_wallet       String,                            -- 0x-prefixed wallet address
    user_name          String DEFAULT '',                 -- Display name (pseudonym)
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),  -- Profile image URL

    -- Ranking
    rank               UInt32,                            -- Leaderboard position
    category           LowCardinality(String),            -- OVERALL, POLITICS, SPORTS, CRYPTO, etc.
    time_period        LowCardinality(String),            -- DAY, WEEK, MONTH, ALL
    order_by           LowCardinality(String),            -- PNL or VOL

    -- Metrics
    pnl                Float64 DEFAULT 0,                 -- Profit/loss for the period
    volume             Float64 DEFAULT 0,                 -- Trading volume for the period

    -- Profile flags
    verified_badge     UInt8 DEFAULT 0,                   -- 1 = verified
    x_username         String DEFAULT '',                 -- X/Twitter handle

    -- Timestamp
    snapshot_time      DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX category_idx category TYPE set(20) GRANULARITY 4
)
ENGINE = ReplacingMergeTree(snapshot_time)
ORDER BY (proxy_wallet, category, time_period, order_by)
SETTINGS index_granularity = 8192;
```

### 1.2 `market_holders` — Top Holders per Market

Stores top holder snapshots per market. ReplacingMergeTree deduplicates by market+wallet+outcome_index, keeping the latest `snapshot_time`.

```sql
CREATE TABLE IF NOT EXISTS market_holders
(
    -- Market identity
    condition_id       LowCardinality(String),            -- FK to markets
    token_id           String DEFAULT '',                  -- Token ID

    -- Holder identity
    proxy_wallet       String,                             -- Holder wallet address
    pseudonym          String DEFAULT '',                  -- Display name
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),   -- Profile image URL
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Holdings
    amount             Float64 DEFAULT 0,                  -- Token balance

    -- Timestamp
    snapshot_time      DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(snapshot_time)
ORDER BY (condition_id, proxy_wallet, outcome_index)
SETTINGS index_granularity = 8192;
```

### 1.3 `wallet_positions` — Tracked Wallet Positions

Stores current open positions for tracked wallets. ReplacingMergeTree deduplicates by wallet+condition_id+outcome, keeping the latest `updated_at`.

```sql
CREATE TABLE IF NOT EXISTS wallet_positions
(
    -- Identity
    proxy_wallet       String,                             -- Wallet address
    condition_id       LowCardinality(String),             -- Market condition ID
    asset              String DEFAULT '',                   -- Token ID
    outcome            LowCardinality(String),             -- 'Yes' / 'No'
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Position data
    size               Float64 DEFAULT 0,                  -- Number of tokens held
    avg_price          Float64 DEFAULT 0,                  -- Average entry price
    initial_value      Float64 DEFAULT 0,                  -- Cost basis (USDC)
    current_value      Float64 DEFAULT 0,                  -- Mark-to-market value (USDC)
    cur_price          Float64 DEFAULT 0,                  -- Current market price

    -- PnL
    cash_pnl           Float64 DEFAULT 0,                  -- Unrealized PnL (USDC)
    percent_pnl        Float64 DEFAULT 0,                  -- Percent return
    realized_pnl       Float64 DEFAULT 0,                  -- Realized PnL (USDC)

    -- Market metadata (denormalized for query convenience)
    title              String DEFAULT '',                   -- Market question
    market_slug        String DEFAULT '',                   -- URL slug
    end_date           DateTime64(3) DEFAULT toDateTime64('2099-01-01', 3),

    -- Timestamp
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (proxy_wallet, condition_id, outcome)
SETTINGS index_granularity = 8192;
```

### 1.4 `wallet_activity` — Wallet Trade/Activity History

Append-only table of all activity events for tracked wallets. MergeTree for time-series storage.

```sql
CREATE TABLE IF NOT EXISTS wallet_activity
(
    -- Identity
    proxy_wallet       String,                             -- Wallet address
    condition_id       LowCardinality(String),             -- Market condition ID
    asset              String DEFAULT '',                   -- Token ID

    -- Activity data
    activity_type      LowCardinality(String),             -- TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE
    side               LowCardinality(String) DEFAULT '',  -- BUY or SELL (for trades)
    outcome            LowCardinality(String) DEFAULT '',  -- Outcome name
    outcome_index      UInt8 DEFAULT 0,                    -- 0 = Yes, 1 = No

    -- Amounts
    size               Float64 DEFAULT 0,                  -- Token amount
    usdc_size          Float64 DEFAULT 0,                  -- USDC equivalent
    price              Float64 DEFAULT 0,                  -- Execution price

    -- On-chain
    transaction_hash   String DEFAULT '',                   -- Polygon tx hash

    -- Market metadata (denormalized)
    title              String DEFAULT '',                   -- Market question
    market_slug        String DEFAULT '',                   -- URL slug

    -- Timestamp
    timestamp          DateTime64(3) CODEC(DoubleDelta, LZ4),
    ts_date            Date MATERIALIZED toDate(timestamp),

    -- Indexes
    INDEX wallet_idx proxy_wallet TYPE bloom_filter GRANULARITY 4,
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX tx_idx transaction_hash TYPE bloom_filter GRANULARITY 4,
    INDEX type_idx activity_type TYPE set(10) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (proxy_wallet, timestamp, condition_id)
TTL timestamp + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;
```

### 1.5 `trader_profiles` — Wallet Profile Data

Stores enriched profile information for discovered wallets. ReplacingMergeTree deduplicates by wallet, keeping the latest `updated_at`.

```sql
CREATE TABLE IF NOT EXISTS trader_profiles
(
    -- Identity
    proxy_wallet       String,                             -- Primary key
    pseudonym          String DEFAULT '',                   -- Display name
    name               String DEFAULT '',                   -- Real name (if set)
    bio                String DEFAULT '' CODEC(ZSTD(3)),    -- User bio
    profile_image      String DEFAULT '' CODEC(ZSTD(3)),    -- Profile image URL

    -- Social
    x_username         String DEFAULT '',                   -- X/Twitter handle
    verified_badge     UInt8 DEFAULT 0,                    -- 1 = verified

    -- Metadata
    display_username_public UInt8 DEFAULT 0,               -- 1 = username is public
    profile_created_at DateTime64(3) DEFAULT toDateTime64('1970-01-01', 3),

    -- Tracking metadata
    discovered_via     LowCardinality(String) DEFAULT '',  -- 'leaderboard', 'holders', 'manual'
    first_seen_at      DateTime64(3) DEFAULT now64(3),     -- When we first discovered this wallet
    updated_at         DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX pseudonym_idx pseudonym TYPE tokenbf_v1(1024, 3, 0) GRANULARITY 4,
    INDEX x_idx x_username TYPE bloom_filter GRANULARITY 4
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (proxy_wallet)
SETTINGS index_granularity = 8192;
```

---

## 2. Pipeline: Data API Client Extension

Extend the existing `pipeline/api/data_client.py` with new methods. The existing `DataClient` class currently has `fetch_recent_trades`, `fetch_all_recent_trades`, and `parse_trade`. Add the following methods:

### 2.1 New Methods on `DataClient`

```python
# --- Add to pipeline/api/data_client.py ---

# New constant at module level:
_LEADERBOARD_CATEGORIES = [
    "OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE",
    "MENTIONS", "WEATHER", "ECONOMICS", "TECH", "FINANCE",
]
_LEADERBOARD_TIME_PERIODS = ["DAY", "WEEK", "MONTH", "ALL"]
_LEADERBOARD_ORDER_TYPES = ["PNL", "VOL"]

async def fetch_leaderboard(
    self,
    *,
    category: str = "OVERALL",
    time_period: str = "ALL",
    order_by: str = "PNL",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """GET /v1/leaderboard — fetch trader rankings.

    Returns list of {rank, proxyWallet, userName, vol, pnl, profileImage,
    xUsername, verifiedBadge}.
    """
    params = {
        "category": category,
        "timePeriod": time_period,
        "orderBy": order_by,
        "limit": limit,
        "offset": offset,
    }
    try:
        resp = await self._client.get("/v1/leaderboard", params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning(
            "fetch_leaderboard_error",
            extra={"category": category, "time_period": time_period},
            exc_info=True,
        )
        return []

async def fetch_leaderboard_page(
    self,
    *,
    category: str = "OVERALL",
    time_period: str = "ALL",
    order_by: str = "PNL",
    max_results: int = 200,
) -> list[dict]:
    """Paginate through leaderboard up to max_results entries."""
    all_entries: list[dict] = []
    offset = 0
    limit = 50  # API max per page

    while len(all_entries) < max_results and offset <= 1000:
        entries = await self.fetch_leaderboard(
            category=category,
            time_period=time_period,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )
        if not entries:
            break
        all_entries.extend(entries)
        if len(entries) < limit:
            break
        offset += limit

    return all_entries[:max_results]

async def fetch_positions(
    self,
    wallet: str,
    *,
    limit: int = 500,
    offset: int = 0,
    size_threshold: float = 1.0,
    sort_by: str = "CURRENT",
) -> list[dict]:
    """GET /positions — fetch current positions for a wallet.

    Returns list of position objects with size, avgPrice, currentValue,
    cashPnl, percentPnl, realizedPnl, etc.
    """
    params: dict = {
        "user": wallet,
        "limit": limit,
        "offset": offset,
        "sizeThreshold": size_threshold,
        "sortBy": sort_by,
        "sortDirection": "DESC",
    }
    try:
        resp = await self._client.get("/positions", params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning(
            "fetch_positions_error",
            extra={"wallet": wallet},
            exc_info=True,
        )
        return []

async def fetch_all_positions(
    self,
    wallet: str,
    *,
    max_pages: int = 5,
) -> list[dict]:
    """Paginate through all positions for a wallet."""
    all_positions: list[dict] = []
    offset = 0
    limit = 500

    for _ in range(max_pages):
        positions = await self.fetch_positions(
            wallet, limit=limit, offset=offset,
        )
        if not positions:
            break
        all_positions.extend(positions)
        if len(positions) < limit:
            break
        offset += limit

    return all_positions

async def fetch_activity(
    self,
    wallet: str,
    *,
    activity_types: list[str] | None = None,
    start: int | None = None,
    end: int | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """GET /activity — fetch activity history for a wallet.

    activity_types: list of TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE
    start/end: Unix timestamps for time range filtering.
    """
    params: dict = {
        "user": wallet,
        "limit": limit,
        "offset": offset,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }
    if activity_types:
        params["type"] = ",".join(activity_types)
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    try:
        resp = await self._client.get("/activity", params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning(
            "fetch_activity_error",
            extra={"wallet": wallet},
            exc_info=True,
        )
        return []

async def fetch_holders(
    self,
    condition_id: str,
    *,
    limit: int = 20,
) -> list[dict]:
    """GET /holders — fetch top holders for a market.

    Returns list of {token, holders: [{proxyWallet, amount, pseudonym, ...}]}.
    """
    params = {"market": condition_id, "limit": limit}
    try:
        resp = await self._client.get("/holders", params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning(
            "fetch_holders_error",
            extra={"condition_id": condition_id},
            exc_info=True,
        )
        return []

async def fetch_value(self, wallet: str) -> float:
    """GET /value — fetch total portfolio value for a wallet.

    Returns the USD value as a float (0.0 if unavailable).
    """
    try:
        resp = await self._client.get("/value", params={"user": wallet})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return float(data[0].get("value", 0))
        return 0.0
    except Exception:
        logger.warning(
            "fetch_value_error",
            extra={"wallet": wallet},
            exc_info=True,
        )
        return 0.0

async def fetch_public_profile(self, wallet: str) -> dict | None:
    """GET /public-profile (Gamma API) — fetch wallet profile.

    Note: This hits the Gamma API, not the Data API. We use a separate
    httpx client with the Gamma base URL.
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                f"{GAMMA_API_URL}/public-profile",
                params={"address": wallet},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.warning(
            "fetch_profile_error",
            extra={"wallet": wallet},
            exc_info=True,
        )
        return None
```

### 2.2 New Static Parse Methods on `DataClient`

```python
@staticmethod
def parse_leaderboard_entry(raw: dict, category: str, time_period: str, order_by: str) -> dict:
    """Convert a raw leaderboard entry into a schema-compatible dict."""
    return {
        "proxy_wallet": raw.get("proxyWallet", ""),
        "user_name": raw.get("userName", ""),
        "profile_image": raw.get("profileImage", ""),
        "rank": int(raw.get("rank", 0)),
        "category": category,
        "time_period": time_period,
        "order_by": order_by,
        "pnl": float(raw.get("pnl") or 0),
        "volume": float(raw.get("vol") or 0),
        "verified_badge": 1 if raw.get("verifiedBadge") else 0,
        "x_username": raw.get("xUsername", ""),
    }

@staticmethod
def parse_position(raw: dict) -> dict:
    """Convert a raw position object into a schema-compatible dict."""
    return {
        "proxy_wallet": raw.get("proxyWallet", ""),
        "condition_id": raw.get("conditionId", ""),
        "asset": raw.get("asset", ""),
        "outcome": raw.get("outcome", ""),
        "outcome_index": int(raw.get("outcomeIndex", 0)),
        "size": float(raw.get("size") or 0),
        "avg_price": float(raw.get("avgPrice") or 0),
        "initial_value": float(raw.get("initialValue") or 0),
        "current_value": float(raw.get("currentValue") or 0),
        "cur_price": float(raw.get("curPrice") or 0),
        "cash_pnl": float(raw.get("cashPnl") or 0),
        "percent_pnl": float(raw.get("percentPnl") or 0),
        "realized_pnl": float(raw.get("realizedPnl") or 0),
        "title": raw.get("title", ""),
        "market_slug": raw.get("slug", ""),
        "end_date": DataClient._parse_dt(raw.get("endDate")),
    }

@staticmethod
def parse_activity(raw: dict) -> dict:
    """Convert a raw activity object into a schema-compatible dict."""
    ts_raw = raw.get("timestamp")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        ts = datetime.now(timezone.utc)

    return {
        "proxy_wallet": raw.get("proxyWallet", ""),
        "condition_id": raw.get("conditionId", ""),
        "asset": raw.get("asset", ""),
        "activity_type": raw.get("type", "TRADE"),
        "side": (raw.get("side") or "").upper(),
        "outcome": raw.get("outcome", ""),
        "outcome_index": int(raw.get("outcomeIndex", 0)),
        "size": float(raw.get("size") or 0),
        "usdc_size": float(raw.get("usdcSize") or 0),
        "price": float(raw.get("price") or 0),
        "transaction_hash": raw.get("transactionHash", ""),
        "title": raw.get("title", ""),
        "market_slug": raw.get("slug", ""),
        "timestamp": ts,
    }

@staticmethod
def parse_holder(raw: dict, condition_id: str, token_id: str) -> dict:
    """Convert a raw holder object into a schema-compatible dict."""
    return {
        "condition_id": condition_id,
        "token_id": token_id,
        "proxy_wallet": raw.get("proxyWallet", ""),
        "pseudonym": raw.get("pseudonym", ""),
        "profile_image": raw.get("profileImage", ""),
        "outcome_index": int(raw.get("outcomeIndex", 0)),
        "amount": float(raw.get("amount") or 0),
    }

@staticmethod
def parse_profile(raw: dict, discovered_via: str = "leaderboard") -> dict:
    """Convert a raw profile object into a schema-compatible dict."""
    created_raw = raw.get("createdAt")
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        created = datetime(1970, 1, 1, tzinfo=timezone.utc)

    return {
        "proxy_wallet": raw.get("proxyWallet", ""),
        "pseudonym": raw.get("pseudonym", ""),
        "name": raw.get("name", ""),
        "bio": raw.get("bio", ""),
        "profile_image": raw.get("profileImage", ""),
        "x_username": raw.get("xUsername", ""),
        "verified_badge": 1 if raw.get("verifiedBadge") else 0,
        "display_username_public": 1 if raw.get("displayUsernamePublic") else 0,
        "profile_created_at": created,
        "discovered_via": discovered_via,
    }

@staticmethod
def _parse_dt(raw: str | None) -> datetime:
    """Parse ISO datetime string, same as GammaClient._parse_dt."""
    if not raw:
        return datetime(2099, 1, 1, tzinfo=timezone.utc)
    try:
        cleaned = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return datetime(2099, 1, 1, tzinfo=timezone.utc)
```

### 2.3 New Import Required

Add to the existing imports at the top of `data_client.py`:

```python
from pipeline.config import DATA_API_URL, GAMMA_API_URL, HTTP_TIMEOUT
```

(Note: `GAMMA_API_URL` is a new import needed for `fetch_public_profile`.)

---

## 3. Pipeline: New Config Constants

Add to `pipeline/config.py`:

```python
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
```

---

## 4. Pipeline: Batched Writer Extension

Add new table column definitions and convenience methods to `pipeline/clickhouse_writer.py`.

### 4.1 New TABLE_COLUMNS Entries

```python
# Add to TABLE_COLUMNS dict:
"trader_rankings": [
    "proxy_wallet", "user_name", "profile_image",
    "rank", "category", "time_period", "order_by",
    "pnl", "volume",
    "verified_badge", "x_username",
    "snapshot_time",
],
"market_holders": [
    "condition_id", "token_id",
    "proxy_wallet", "pseudonym", "profile_image", "outcome_index",
    "amount",
    "snapshot_time",
],
"wallet_positions": [
    "proxy_wallet", "condition_id", "asset", "outcome", "outcome_index",
    "size", "avg_price", "initial_value", "current_value", "cur_price",
    "cash_pnl", "percent_pnl", "realized_pnl",
    "title", "market_slug", "end_date",
    "updated_at",
],
"wallet_activity": [
    "proxy_wallet", "condition_id", "asset",
    "activity_type", "side", "outcome", "outcome_index",
    "size", "usdc_size", "price",
    "transaction_hash",
    "title", "market_slug",
    "timestamp",
],
"trader_profiles": [
    "proxy_wallet", "pseudonym", "name", "bio", "profile_image",
    "x_username", "verified_badge",
    "display_username_public", "profile_created_at",
    "discovered_via", "first_seen_at", "updated_at",
],
```

### 4.2 New Convenience Methods

```python
async def write_rankings(self, rows: list[list[Any]]) -> None:
    await self.write("trader_rankings", rows)

async def write_holders(self, rows: list[list[Any]]) -> None:
    await self.write("market_holders", rows)

async def write_positions(self, rows: list[list[Any]]) -> None:
    await self.write("wallet_positions", rows)

async def write_activity(self, rows: list[list[Any]]) -> None:
    await self.write("wallet_activity", rows)

async def write_profiles(self, rows: list[list[Any]]) -> None:
    await self.write("trader_profiles", rows)
```

---

## 5. Pipeline Jobs

### 5.1 `pipeline/jobs/leaderboard_sync.py` — Leaderboard Scraper (Hourly)

Polls across all category/time_period/order_by combinations. Discovers new wallets for tracking.

```python
"""Job: sync trader leaderboard data from the Data API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import (
    DataClient,
    _LEADERBOARD_CATEGORIES,
    _LEADERBOARD_TIME_PERIODS,
    _LEADERBOARD_ORDER_TYPES,
)
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import LEADERBOARD_MAX_RESULTS

logger = logging.getLogger(__name__)

# Module-level set of discovered wallet addresses (shared with other jobs)
discovered_wallets: set[str] = set()


async def run_leaderboard_sync() -> set[str]:
    """Fetch leaderboard across all combos and write to trader_rankings.

    Returns the set of all discovered wallet addresses (for downstream jobs).
    """
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)
    new_wallets: set[str] = set()

    try:
        total_rows = 0

        for category in _LEADERBOARD_CATEGORIES:
            for time_period in _LEADERBOARD_TIME_PERIODS:
                for order_by in _LEADERBOARD_ORDER_TYPES:
                    entries = await client.fetch_leaderboard_page(
                        category=category,
                        time_period=time_period,
                        order_by=order_by,
                        max_results=LEADERBOARD_MAX_RESULTS,
                    )

                    rows = []
                    for entry in entries:
                        parsed = DataClient.parse_leaderboard_entry(
                            entry, category, time_period, order_by,
                        )
                        wallet = parsed["proxy_wallet"]
                        if wallet:
                            new_wallets.add(wallet)
                            rows.append([
                                parsed["proxy_wallet"],
                                parsed["user_name"],
                                parsed["profile_image"],
                                parsed["rank"],
                                parsed["category"],
                                parsed["time_period"],
                                parsed["order_by"],
                                parsed["pnl"],
                                parsed["volume"],
                                parsed["verified_badge"],
                                parsed["x_username"],
                                now,
                            ])

                    if rows:
                        await writer.write_rankings(rows)
                        total_rows += len(rows)

                    # Small delay between API calls to be respectful
                    await asyncio.sleep(0.2)

        await writer.flush_all()
        discovered_wallets.update(new_wallets)

        logger.info(
            "leaderboard_sync_complete",
            extra={
                "total_entries": total_rows,
                "unique_wallets": len(new_wallets),
                "total_tracked": len(discovered_wallets),
            },
        )
        return discovered_wallets

    finally:
        await client.close()
```

### 5.2 `pipeline/jobs/holder_sync.py` — Market Holder Tracker (Every 15 min)

Polls top holders for the highest-volume active markets.

```python
"""Job: sync top holders for active markets from the Data API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import HOLDER_SYNC_TOP_MARKETS

logger = logging.getLogger(__name__)


async def run_holder_sync(active_condition_ids: list[str]) -> None:
    """Fetch top holders for the top N markets by volume.

    active_condition_ids should be pre-sorted by volume (descending),
    as provided by market_sync.
    """
    if not active_condition_ids:
        logger.debug("holder_sync_skip", extra={"reason": "no_markets"})
        return

    markets_to_poll = active_condition_ids[:HOLDER_SYNC_TOP_MARKETS]
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        total_holders = 0

        for condition_id in markets_to_poll:
            holder_groups = await client.fetch_holders(condition_id)

            rows = []
            for group in holder_groups:
                token_id = group.get("token", "")
                holders = group.get("holders") or []

                for h in holders:
                    parsed = DataClient.parse_holder(h, condition_id, token_id)
                    rows.append([
                        parsed["condition_id"],
                        parsed["token_id"],
                        parsed["proxy_wallet"],
                        parsed["pseudonym"],
                        parsed["profile_image"],
                        parsed["outcome_index"],
                        parsed["amount"],
                        now,
                    ])

            if rows:
                await writer.write_holders(rows)
                total_holders += len(rows)

            # Rate limiting
            await asyncio.sleep(0.15)

        await writer.flush_all()
        logger.info(
            "holder_sync_complete",
            extra={
                "markets_polled": len(markets_to_poll),
                "total_holders": total_holders,
            },
        )

    finally:
        await client.close()
```

### 5.3 `pipeline/jobs/position_sync.py` — Position/Activity Poller (Every 5 min)

Polls positions and recent activity for tracked wallets (seeded from leaderboard).

```python
"""Job: sync positions and activity for tracked wallets."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import TRACKED_WALLET_MAX
from pipeline.jobs.leaderboard_sync import discovered_wallets

logger = logging.getLogger(__name__)

# Track the last activity timestamp per wallet for dedup
_last_activity_ts: dict[str, datetime] = {}


async def run_position_sync() -> None:
    """Fetch positions and recent activity for all tracked wallets.

    Tracked wallets are sourced from the leaderboard_sync discovered_wallets set.
    """
    wallets = list(discovered_wallets)[:TRACKED_WALLET_MAX]
    if not wallets:
        logger.debug("position_sync_skip", extra={"reason": "no_tracked_wallets"})
        return

    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        total_positions = 0
        total_activities = 0

        for wallet in wallets:
            # --- Positions ---
            positions = await client.fetch_all_positions(wallet, max_pages=2)
            pos_rows = []
            for raw_pos in positions:
                parsed = DataClient.parse_position(raw_pos)
                if not parsed["condition_id"]:
                    continue
                pos_rows.append([
                    parsed["proxy_wallet"],
                    parsed["condition_id"],
                    parsed["asset"],
                    parsed["outcome"],
                    parsed["outcome_index"],
                    parsed["size"],
                    parsed["avg_price"],
                    parsed["initial_value"],
                    parsed["current_value"],
                    parsed["cur_price"],
                    parsed["cash_pnl"],
                    parsed["percent_pnl"],
                    parsed["realized_pnl"],
                    parsed["title"],
                    parsed["market_slug"],
                    parsed["end_date"],
                    now,
                ])

            if pos_rows:
                await writer.write_positions(pos_rows)
                total_positions += len(pos_rows)

            # --- Activity (recent only) ---
            activities = await client.fetch_activity(wallet, limit=100)
            act_rows = []
            for raw_act in activities:
                parsed = DataClient.parse_activity(raw_act)
                ts = parsed["timestamp"]

                # Deduplicate: skip activities older than last seen
                if wallet in _last_activity_ts and ts <= _last_activity_ts[wallet]:
                    continue

                act_rows.append([
                    parsed["proxy_wallet"],
                    parsed["condition_id"],
                    parsed["asset"],
                    parsed["activity_type"],
                    parsed["side"],
                    parsed["outcome"],
                    parsed["outcome_index"],
                    parsed["size"],
                    parsed["usdc_size"],
                    parsed["price"],
                    parsed["transaction_hash"],
                    parsed["title"],
                    parsed["market_slug"],
                    parsed["timestamp"],
                ])

            if act_rows:
                await writer.write_activity(act_rows)
                total_activities += len(act_rows)

                # Update watermark
                latest_ts = max(
                    DataClient.parse_activity(a)["timestamp"]
                    for a in activities
                    if DataClient.parse_activity(a)["proxy_wallet"] == wallet
                )
                _last_activity_ts[wallet] = latest_ts

            # Rate limiting between wallets
            await asyncio.sleep(0.1)

        await writer.flush_all()
        logger.info(
            "position_sync_complete",
            extra={
                "wallets_polled": len(wallets),
                "total_positions": total_positions,
                "total_activities": total_activities,
            },
        )

    finally:
        await client.close()
```

### 5.4 `pipeline/jobs/profile_enricher.py` — Profile Enrichment (Every 10 min)

Enriches newly discovered wallets with Gamma API profile data.

```python
"""Job: enrich discovered wallets with profile data from Gamma API."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pipeline.api.data_client import DataClient
from pipeline.clickhouse_writer import ClickHouseWriter
from pipeline.config import PROFILE_BATCH_SIZE
from pipeline.jobs.leaderboard_sync import discovered_wallets

logger = logging.getLogger(__name__)

# Track which wallets have been enriched (avoid re-fetching)
_enriched_wallets: set[str] = set()


async def run_profile_enricher() -> None:
    """Fetch profiles for newly discovered wallets (not yet enriched).

    Processes up to PROFILE_BATCH_SIZE wallets per cycle.
    """
    # Find wallets needing enrichment
    pending = discovered_wallets - _enriched_wallets
    if not pending:
        logger.debug("profile_enricher_skip", extra={"reason": "no_new_wallets"})
        return

    batch = list(pending)[:PROFILE_BATCH_SIZE]
    client = DataClient()
    writer = ClickHouseWriter.get_instance()
    now = datetime.now(timezone.utc)

    try:
        rows = []
        enriched_count = 0

        for wallet in batch:
            profile = await client.fetch_public_profile(wallet)

            if profile:
                parsed = DataClient.parse_profile(profile, discovered_via="leaderboard")
                rows.append([
                    parsed["proxy_wallet"] or wallet,
                    parsed["pseudonym"],
                    parsed["name"],
                    parsed["bio"],
                    parsed["profile_image"],
                    parsed["x_username"],
                    parsed["verified_badge"],
                    parsed["display_username_public"],
                    parsed["profile_created_at"],
                    parsed["discovered_via"],
                    now,  # first_seen_at
                    now,  # updated_at
                ])
                enriched_count += 1
            else:
                # Store a minimal profile even if not found (so we don't retry)
                rows.append([
                    wallet,
                    "",   # pseudonym
                    "",   # name
                    "",   # bio
                    "",   # profile_image
                    "",   # x_username
                    0,    # verified_badge
                    0,    # display_username_public
                    datetime(1970, 1, 1, tzinfo=timezone.utc),  # profile_created_at
                    "leaderboard",  # discovered_via
                    now,  # first_seen_at
                    now,  # updated_at
                ])

            _enriched_wallets.add(wallet)
            # Rate limiting
            await asyncio.sleep(0.3)

        if rows:
            await writer.write_profiles(rows)
            await writer.flush_all()

        logger.info(
            "profile_enricher_complete",
            extra={
                "batch_size": len(batch),
                "enriched": enriched_count,
                "total_enriched": len(_enriched_wallets),
                "remaining": len(discovered_wallets - _enriched_wallets),
            },
        )

    finally:
        await client.close()
```

---

## 6. Scheduler Updates

Add Phase 2 jobs to `pipeline/scheduler.py`.

### 6.1 New Imports

```python
from pipeline.config import (
    # ...existing imports...
    LEADERBOARD_SYNC_INTERVAL,
    HOLDER_SYNC_INTERVAL,
    POSITION_SYNC_INTERVAL,
    PROFILE_ENRICH_INTERVAL,
)
from pipeline.jobs.leaderboard_sync import run_leaderboard_sync
from pipeline.jobs.holder_sync import run_holder_sync
from pipeline.jobs.position_sync import run_position_sync
from pipeline.jobs.profile_enricher import run_profile_enricher
```

### 6.2 New Instance Variable

Add to `PipelineScheduler.__init__`:

```python
self._active_condition_ids: list[str] = []
```

### 6.3 Updated `_job_market_sync` to Track condition_ids

The existing `_job_market_sync` returns token IDs. We need to also capture condition IDs (sorted by volume) for the holder sync job. Modify `run_market_sync` to also return condition IDs, or derive them in the scheduler.

**Approach**: Add a module-level list in `market_sync.py` that the scheduler can read:

```python
# In pipeline/jobs/market_sync.py, add after the existing code:
# Module-level list of active condition IDs sorted by volume (updated each sync)
active_condition_ids: list[str] = []
```

In the existing `run_market_sync()`, populate it:

```python
# Inside run_market_sync(), after building markets list and sorting by volume:
global active_condition_ids
active_condition_ids = [
    m["condition_id"] for m in markets
    if m["active"] and not m["closed"]
]
```

Import in scheduler:

```python
from pipeline.jobs.market_sync import run_market_sync, active_condition_ids
```

### 6.4 New Job Registrations

Add inside `PipelineScheduler.start()`, after the existing job registrations:

```python
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
```

### 6.5 New Job Wrappers

Add to the `PipelineScheduler` class:

```python
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
```

### 6.6 Initial Leaderboard Sync

Add to `PipelineScheduler.start()`, after the initial market sync:

```python
# Initial leaderboard sync to seed tracked wallets
logger.info("initial_leaderboard_sync")
try:
    await run_leaderboard_sync()
except Exception:
    logger.error("initial_leaderboard_sync_failed", exc_info=True)
```

### 6.7 Updated Health Check

Update `_health_handler` to include Phase 2 stats:

```python
async def _health_handler(self, request: web.Request) -> web.Response:
    from pipeline.jobs.leaderboard_sync import discovered_wallets
    return web.json_response({
        "status": "ok",
        "active_tokens": len(self._active_token_ids),
        "tracked_wallets": len(discovered_wallets),
        "scheduler_running": self._scheduler.running,
    })
```

---

## 7. Dashboard: TypeScript Types

Add to `dashboard/src/types/market.ts`:

```typescript
// --- Phase 2 Whale/User Types ---

export interface TraderRanking {
  proxy_wallet: string;
  user_name: string;
  profile_image: string;
  rank: number;
  category: string;
  time_period: string;
  order_by: string;
  pnl: number;
  volume: number;
  verified_badge: number;
  x_username: string;
  snapshot_time: string;
}

export interface MarketHolder {
  condition_id: string;
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  outcome_index: number;
  amount: number;
  snapshot_time: string;
}

export interface WalletPosition {
  proxy_wallet: string;
  condition_id: string;
  outcome: string;
  size: number;
  avg_price: number;
  current_value: number;
  cur_price: number;
  cash_pnl: number;
  percent_pnl: number;
  realized_pnl: number;
  title: string;
  market_slug: string;
  updated_at: string;
}

export interface WalletActivity {
  proxy_wallet: string;
  condition_id: string;
  activity_type: string;
  side: string;
  outcome: string;
  size: number;
  usdc_size: number;
  price: number;
  transaction_hash: string;
  title: string;
  timestamp: string;
}

export interface TraderProfile {
  proxy_wallet: string;
  pseudonym: string;
  name: string;
  bio: string;
  profile_image: string;
  x_username: string;
  verified_badge: number;
  first_seen_at: string;
}

export interface WhaleActivityFeed {
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  condition_id: string;
  activity_type: string;
  side: string;
  outcome: string;
  size: number;
  usdc_size: number;
  price: number;
  title: string;
  timestamp: string;
}

export interface SmartMoneyPosition {
  proxy_wallet: string;
  pseudonym: string;
  profile_image: string;
  rank: number;
  condition_id: string;
  outcome: string;
  size: number;
  current_value: number;
  cash_pnl: number;
  percent_pnl: number;
  title: string;
}

export interface PositionConcentration {
  condition_id: string;
  question: string;
  total_holders: number;
  total_amount: number;
  top5_amount: number;
  top5_share: number;
  top_holder_wallet: string;
  top_holder_amount: number;
}

export interface WhalesOverview {
  tracked_wallets: number;
  whale_trades_24h: number;
  total_whale_positions: number;
  unique_markets_held: number;
}
```

---

## 8. Dashboard: Query Functions

Add to `dashboard/src/lib/queries.ts`. Import the new types.

### 8.1 Updated Imports

```typescript
import type {
  // ...existing imports...
  TraderRanking,
  WhaleActivityFeed,
  SmartMoneyPosition,
  MarketHolder,
  TraderProfile,
  PositionConcentration,
  WhalesOverview,
} from "@/types/market";
```

### 8.2 `getLeaderboard`

```typescript
export async function getLeaderboard(
  category = "OVERALL",
  timePeriod = "ALL",
  orderBy = "PNL",
  limit = 50
): Promise<TraderRanking[]> {
  return query<TraderRanking>(
    `SELECT
      proxy_wallet,
      user_name,
      profile_image,
      rank,
      category,
      time_period,
      order_by,
      pnl,
      volume,
      verified_badge,
      x_username,
      snapshot_time
    FROM trader_rankings FINAL
    WHERE category = {category:String}
      AND time_period = {timePeriod:String}
      AND order_by = {orderBy:String}
    ORDER BY rank ASC
    LIMIT {limit:UInt32}`,
    { category, timePeriod, orderBy, limit }
  );
}
```

### 8.3 `getWhaleActivity`

```typescript
export async function getWhaleActivity(limit = 50): Promise<WhaleActivityFeed[]> {
  return query<WhaleActivityFeed>(
    `SELECT
      wa.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      wa.condition_id,
      wa.activity_type,
      wa.side,
      wa.outcome,
      wa.size,
      wa.usdc_size,
      wa.price,
      wa.title,
      wa.timestamp
    FROM wallet_activity wa
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON wa.proxy_wallet = tp.proxy_wallet
    WHERE wa.timestamp >= now() - INTERVAL 24 HOUR
      AND wa.usdc_size >= 500
    ORDER BY wa.timestamp DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.4 `getSmartMoneyPositions`

```typescript
export async function getSmartMoneyPositions(limit = 50): Promise<SmartMoneyPosition[]> {
  return query<SmartMoneyPosition>(
    `SELECT
      wp.proxy_wallet,
      tp.pseudonym,
      tp.profile_image,
      tr.rank,
      wp.condition_id,
      wp.outcome,
      wp.size,
      wp.current_value,
      wp.cash_pnl,
      wp.percent_pnl,
      wp.title
    FROM (SELECT * FROM wallet_positions FINAL) AS wp
    INNER JOIN (
      SELECT proxy_wallet, min(rank) AS rank
      FROM trader_rankings FINAL
      WHERE category = 'OVERALL' AND time_period = 'ALL' AND order_by = 'PNL'
      GROUP BY proxy_wallet
    ) AS tr ON wp.proxy_wallet = tr.proxy_wallet
    LEFT JOIN (
      SELECT proxy_wallet, pseudonym, profile_image
      FROM trader_profiles FINAL
    ) AS tp ON wp.proxy_wallet = tp.proxy_wallet
    WHERE wp.size > 0
      AND wp.current_value > 100
    ORDER BY tr.rank ASC, wp.current_value DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.5 `getTopHolders`

```typescript
export async function getTopHolders(
  conditionId: string,
  limit = 20
): Promise<MarketHolder[]> {
  return query<MarketHolder>(
    `SELECT
      condition_id,
      proxy_wallet,
      pseudonym,
      profile_image,
      outcome_index,
      amount,
      snapshot_time
    FROM market_holders FINAL
    WHERE condition_id = {conditionId:String}
    ORDER BY amount DESC
    LIMIT {limit:UInt32}`,
    { conditionId, limit }
  );
}
```

### 8.6 `getTraderProfile`

```typescript
export async function getTraderProfile(
  wallet: string
): Promise<TraderProfile | null> {
  const rows = await query<TraderProfile>(
    `SELECT
      proxy_wallet,
      pseudonym,
      name,
      bio,
      profile_image,
      x_username,
      verified_badge,
      first_seen_at
    FROM trader_profiles FINAL
    WHERE proxy_wallet = {wallet:String}
    LIMIT 1`,
    { wallet }
  );
  return rows[0] ?? null;
}
```

### 8.7 `getPositionConcentration`

```typescript
export async function getPositionConcentration(
  limit = 30
): Promise<PositionConcentration[]> {
  return query<PositionConcentration>(
    `WITH holder_stats AS (
      SELECT
        condition_id,
        count() AS total_holders,
        sum(amount) AS total_amount,
        arraySlice(
          groupArray(amount) AS amounts_sorted,
          1, 5
        ) AS top5_amounts,
        arraySlice(
          groupArray(proxy_wallet) AS wallets_sorted,
          1, 1
        ) AS top_wallets
      FROM (
        SELECT condition_id, proxy_wallet, amount
        FROM market_holders FINAL
        ORDER BY amount DESC
      )
      GROUP BY condition_id
      HAVING total_holders >= 3
    )
    SELECT
      hs.condition_id,
      m.question,
      hs.total_holders,
      hs.total_amount,
      arraySum(hs.top5_amounts) AS top5_amount,
      arraySum(hs.top5_amounts) / greatest(hs.total_amount, 0.01) AS top5_share,
      hs.top_wallets[1] AS top_holder_wallet,
      hs.top5_amounts[1] AS top_holder_amount
    FROM holder_stats hs
    INNER JOIN (SELECT condition_id, question FROM markets FINAL) AS m
      ON hs.condition_id = m.condition_id
    ORDER BY top5_share DESC
    LIMIT {limit:UInt32}`,
    { limit }
  );
}
```

### 8.8 `getWhalesOverview`

```typescript
export async function getWhalesOverview(): Promise<WhalesOverview> {
  const rows = await query<WhalesOverview>(
    `SELECT
      (SELECT uniq(proxy_wallet) FROM trader_rankings FINAL) AS tracked_wallets,
      (
        SELECT count()
        FROM wallet_activity
        WHERE timestamp >= now() - INTERVAL 24 HOUR
          AND usdc_size >= 500
      ) AS whale_trades_24h,
      (
        SELECT count()
        FROM (SELECT * FROM wallet_positions FINAL)
        WHERE size > 0
      ) AS total_whale_positions,
      (
        SELECT uniq(condition_id)
        FROM (SELECT * FROM wallet_positions FINAL)
        WHERE size > 0
      ) AS unique_markets_held`
  );
  return rows[0] ?? {
    tracked_wallets: 0,
    whale_trades_24h: 0,
    total_whale_positions: 0,
    unique_markets_held: 0,
  };
}
```

---

## 9. Dashboard: Pages and Components

### 9.1 Page: `/whales` (`dashboard/src/app/whales/page.tsx`)

Server component with `export const dynamic = "force-dynamic"`. Follows the same Suspense pattern as `/signals`.

**Layout:**
```
[Stats Cards Row] — 4 cards: Tracked Wallets, Whale Trades (24h), Total Positions, Markets Held
[Tabs: Leaderboard | Activity Feed | Smart Money | Concentration]
  [Leaderboard Tab]      → LeaderboardTable (with category/period filters)
  [Activity Tab]         → WhaleActivityTable
  [Smart Money Tab]      → SmartMoneyTable
  [Concentration Tab]    → ConcentrationTable
```

**Server sections:**
```typescript
import { Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getWhalesOverview,
  getLeaderboard,
  getWhaleActivity,
  getSmartMoneyPositions,
  getPositionConcentration,
} from "@/lib/queries";
import {
  WhalesStatsCards,
  WhalesStatsCardsSkeleton,
} from "@/components/whales-stats-cards";
import { WhalesTabs } from "@/components/whales-tabs";

export const dynamic = "force-dynamic";

async function WhalesStatsSection() {
  const stats = await getWhalesOverview();
  return <WhalesStatsCards stats={stats} />;
}

async function WhalesContent() {
  const [leaderboard, activity, smartMoney, concentration] = await Promise.all([
    getLeaderboard("OVERALL", "ALL", "PNL", 50),
    getWhaleActivity(50),
    getSmartMoneyPositions(50),
    getPositionConcentration(30),
  ]);
  return (
    <WhalesTabs
      leaderboard={leaderboard}
      activity={activity}
      smartMoney={smartMoney}
      concentration={concentration}
    />
  );
}

export default function WhalesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Whales</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Top trader rankings, whale activity, and smart money positions
        </p>
      </div>

      <Suspense fallback={<WhalesStatsCardsSkeleton />}>
        <WhalesStatsSection />
      </Suspense>

      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">Whale Intelligence</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense
            fallback={
              <div className="space-y-3">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-12 bg-[#1e1e2e] rounded animate-pulse"
                  />
                ))}
              </div>
            }
          >
            <WhalesContent />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}
```

### 9.2 Component: `WhalesStatsCards` (`dashboard/src/components/whales-stats-cards.tsx`)

Client component. Four stat cards.

| Card | Value | Icon (lucide-react) | Color |
|------|-------|---------------------|-------|
| Tracked Wallets | `stats.tracked_wallets` | `Users` | `text-violet-400` |
| Whale Trades (24h) | `stats.whale_trades_24h` | `Zap` | `text-amber-400` |
| Total Positions | `stats.total_whale_positions` | `Wallet` | `text-emerald-400` |
| Markets Held | `stats.unique_markets_held` | `BarChart3` | `text-blue-400` |

```typescript
"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Users, Zap, Wallet, BarChart3 } from "lucide-react";
import { formatNumber } from "@/lib/format";
import type { WhalesOverview } from "@/types/market";

interface WhalesStatsCardsProps {
  stats: WhalesOverview;
}

export function WhalesStatsCards({ stats }: WhalesStatsCardsProps) {
  const cards = [
    { label: "Tracked Wallets", value: stats.tracked_wallets, icon: Users, color: "text-violet-400" },
    { label: "Whale Trades (24h)", value: stats.whale_trades_24h, icon: Zap, color: "text-amber-400" },
    { label: "Total Positions", value: stats.total_whale_positions, icon: Wallet, color: "text-emerald-400" },
    { label: "Markets Held", value: stats.unique_markets_held, icon: BarChart3, color: "text-blue-400" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((card) => (
        <Card key={card.label} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-2 mb-1">
              <card.icon className={`h-4 w-4 ${card.color}`} />
              <span className="text-xs text-muted-foreground">{card.label}</span>
            </div>
            <p className="text-xl font-bold">{formatNumber(card.value)}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function WhalesStatsCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse mb-2" />
            <div className="h-6 w-16 bg-[#1e1e2e] rounded animate-pulse" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
```

### 9.3 Component: `WhalesTabs` (`dashboard/src/components/whales-tabs.tsx`)

Client component wrapping the four table components.

```typescript
"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LeaderboardTable } from "./leaderboard-table";
import { WhaleActivityTable } from "./whale-activity-table";
import { SmartMoneyTable } from "./smart-money-table";
import { ConcentrationTable } from "./concentration-table";
import type {
  TraderRanking,
  WhaleActivityFeed,
  SmartMoneyPosition,
  PositionConcentration,
} from "@/types/market";

interface WhalesTabsProps {
  leaderboard: TraderRanking[];
  activity: WhaleActivityFeed[];
  smartMoney: SmartMoneyPosition[];
  concentration: PositionConcentration[];
}

export function WhalesTabs({
  leaderboard,
  activity,
  smartMoney,
  concentration,
}: WhalesTabsProps) {
  return (
    <Tabs defaultValue="leaderboard">
      <TabsList>
        <TabsTrigger value="leaderboard">
          Leaderboard ({leaderboard.length})
        </TabsTrigger>
        <TabsTrigger value="activity">
          Activity ({activity.length})
        </TabsTrigger>
        <TabsTrigger value="smart-money">
          Smart Money ({smartMoney.length})
        </TabsTrigger>
        <TabsTrigger value="concentration">
          Concentration ({concentration.length})
        </TabsTrigger>
      </TabsList>
      <TabsContent value="leaderboard">
        <LeaderboardTable data={leaderboard} />
      </TabsContent>
      <TabsContent value="activity">
        <WhaleActivityTable data={activity} />
      </TabsContent>
      <TabsContent value="smart-money">
        <SmartMoneyTable data={smartMoney} />
      </TabsContent>
      <TabsContent value="concentration">
        <ConcentrationTable data={concentration} />
      </TabsContent>
    </Tabs>
  );
}
```

### 9.4 Component: `LeaderboardTable` (`dashboard/src/components/leaderboard-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Rank | `rank` | `#1`, `#2`, ... |
| Trader | `user_name` + `profile_image` | Avatar + name (or truncated wallet) |
| PnL | `pnl` | `formatUSD()` with green/red color |
| Volume | `volume` | `formatUSD()` |
| Verified | `verified_badge` | Badge icon |
| X | `x_username` | Link to X profile |

### 9.5 Component: `WhaleActivityTable` (`dashboard/src/components/whale-activity-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Trader | `pseudonym` + `profile_image` | Avatar + name (or truncated wallet) |
| Market | `title` (link to `/market/[condition_id]`) | Truncated text |
| Type | `activity_type` | Badge (colored by type) |
| Side | `side` | Badge: green "Buy", red "Sell" |
| Size | `usdc_size` | `formatUSD()` |
| Price | `price` | `formatPrice()` |
| Time | `timestamp` | Relative time ("2m ago") |

### 9.6 Component: `SmartMoneyTable` (`dashboard/src/components/smart-money-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Rank | `rank` | `#1`, `#2`, ... |
| Trader | `pseudonym` + `profile_image` | Avatar + name |
| Market | `title` (link to `/market/[condition_id]`) | Truncated text |
| Outcome | `outcome` | Badge |
| Position | `current_value` | `formatUSD()` |
| PnL | `cash_pnl` | `formatUSD()` with green/red |
| Return | `percent_pnl` | `formatPct()` |

### 9.7 Component: `ConcentrationTable` (`dashboard/src/components/concentration-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Market | `question` (link to `/market/[condition_id]`) | Truncated text |
| Holders | `total_holders` | `formatNumber()` |
| Total Held | `total_amount` | `formatNumber()` tokens |
| Top-5 Share | `top5_share` | Percentage with progress bar |
| Top Holder | `top_holder_wallet` | Truncated address |
| Top Amount | `top_holder_amount` | `formatNumber()` tokens |

---

## 10. Navigation Update

Update `dashboard/src/app/layout.tsx` to add the `/whales` route.

```typescript
import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Zap,
  Users,    // NEW
} from "lucide-react";

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/#markets", label: "Markets", icon: BarChart3 },
  { href: "/#trending", label: "Trending", icon: TrendingUp },
  { href: "/signals", label: "Signals", icon: Zap },
  { href: "/whales", label: "Whales", icon: Users },    // NEW
];
```

---

## 11. Market Detail Enhancement

Add holder data to the existing market detail page (`/market/[id]`).

### 11.1 New Section in `dashboard/src/app/market/[id]/page.tsx`

After the existing orderbook/trades sections, add a "Top Holders" section:

```typescript
async function HoldersSection({ conditionId }: { conditionId: string }) {
  const holders = await getTopHolders(conditionId, 20);
  if (holders.length === 0) return null;
  return <TopHoldersTable data={holders} />;
}
```

### 11.2 Component: `TopHoldersTable` (`dashboard/src/components/top-holders-table.tsx`)

| Column | Source | Format |
|--------|--------|--------|
| Holder | `pseudonym` or truncated `proxy_wallet` | Text |
| Outcome | `outcome_index` | Badge ("Yes" / "No") |
| Amount | `amount` | `formatNumber()` |
| Updated | `snapshot_time` | Relative time |

---

## 12. Schema Migration Strategy

### 12.1 New Migration File

Create `pipeline/schema/002_phase2_users.sql` with all 5 CREATE TABLE statements from Section 1.

### 12.2 Migration Runner Update

The existing `main.py` reads and executes `001_init.sql` on startup. Update to also execute `002_phase2_users.sql`:

```python
# In the migration section of main.py or wherever schema is initialized:
schema_files = ["001_init.sql", "002_phase2_users.sql"]
for schema_file in schema_files:
    schema_path = Path(__file__).parent / "schema" / schema_file
    if schema_path.exists():
        writer.run_migration(schema_path.read_text())
```

---

## 13. File-by-File Change List

### New Files (create)

| File | Type | Description |
|------|------|-------------|
| `pipeline/schema/002_phase2_users.sql` | SQL DDL | 5 new ClickHouse tables |
| `pipeline/jobs/leaderboard_sync.py` | Python Job | Hourly leaderboard scraper |
| `pipeline/jobs/holder_sync.py` | Python Job | 15-min market holder tracker |
| `pipeline/jobs/position_sync.py` | Python Job | 5-min wallet position/activity poller |
| `pipeline/jobs/profile_enricher.py` | Python Job | 10-min profile enrichment |
| `dashboard/src/app/whales/page.tsx` | Server Component | Whales page |
| `dashboard/src/components/whales-stats-cards.tsx` | Client Component | Stats cards |
| `dashboard/src/components/whales-tabs.tsx` | Client Component | Tab wrapper |
| `dashboard/src/components/leaderboard-table.tsx` | Client Component | Leaderboard table |
| `dashboard/src/components/whale-activity-table.tsx` | Client Component | Activity feed table |
| `dashboard/src/components/smart-money-table.tsx` | Client Component | Smart money positions table |
| `dashboard/src/components/concentration-table.tsx` | Client Component | Position concentration table |
| `dashboard/src/components/top-holders-table.tsx` | Client Component | Market detail holders table |

### Modified Files (edit)

| File | Changes |
|------|---------|
| `pipeline/config.py` | Add 8 new constants (intervals + tuning params) |
| `pipeline/api/data_client.py` | Add 7 new fetch methods, 5 parse methods, 1 utility. Add `GAMMA_API_URL` import |
| `pipeline/clickhouse_writer.py` | Add 5 new TABLE_COLUMNS entries + 5 convenience methods |
| `pipeline/scheduler.py` | Add 4 new job imports, 4 job registrations, 4 job wrappers, initial leaderboard sync, health check update |
| `pipeline/jobs/market_sync.py` | Add `active_condition_ids` module-level list, populate in `run_market_sync` |
| `pipeline/main.py` | Update migration runner to include `002_phase2_users.sql` |
| `dashboard/src/types/market.ts` | Add 9 new interfaces |
| `dashboard/src/lib/queries.ts` | Add 8 new query functions + import updates |
| `dashboard/src/app/layout.tsx` | Add `Users` icon import + `/whales` nav item |
| `dashboard/src/app/market/[id]/page.tsx` | Add HoldersSection with getTopHolders |

### Unchanged Files

| File | Notes |
|------|-------|
| `pipeline/schema/001_init.sql` | No changes to existing tables |
| `pipeline/api/clob_client.py` | No changes |
| `pipeline/api/gamma_client.py` | No changes |
| `pipeline/api/ws_client.py` | No changes |
| `pipeline/jobs/price_poller.py` | No changes |
| `pipeline/jobs/trade_collector.py` | No changes |
| `pipeline/jobs/orderbook_snapshot.py` | No changes |
| `dashboard/src/lib/clickhouse.ts` | Client singleton unchanged |
| `dashboard/src/lib/format.ts` | Existing formatters sufficient |
| All Phase 1 signal components | No modifications needed |

---

## 14. API Rate Limit Budget

Estimated API calls per hour at steady state:

| Job | Calls/Cycle | Cycle/Hour | Calls/Hour |
|-----|-------------|------------|------------|
| Leaderboard Sync | 10 categories x 4 periods x 2 orders x ~4 pages = 320 | 1 | 320 |
| Holder Sync | 50 markets x 1 call = 50 | 4 | 200 |
| Position Sync | 500 wallets x 2 calls (positions + activity) = 1000 | 12 | 12,000 |
| Profile Enricher | 20 wallets x 1 call = 20 | 6 | 120 |
| **Total** | | | **~12,640** |

The Data API does not publish rate limits. The heaviest load is position_sync at ~12K calls/hour (~3.3/sec). This is well below CLOB API limits (1500/10s). If needed, reduce `TRACKED_WALLET_MAX` or increase `POSITION_SYNC_INTERVAL`.

---

## 15. Design Decisions & Rationale

1. **ReplacingMergeTree for rankings/positions/holders**: These are "current state" tables — we only care about the latest snapshot. ReplacingMergeTree deduplicates on FINAL queries, same pattern as the existing `markets` table.

2. **MergeTree for wallet_activity**: This is append-only time-series data, same pattern as `market_trades`. Partitioned monthly with 1-year TTL.

3. **Denormalized market metadata in positions/activity**: Avoids JOINs at query time. The Data API includes `title`, `slug`, `outcome` in every response, so we store it.

4. **Module-level `discovered_wallets` set**: Simple in-memory wallet discovery mechanism shared between leaderboard_sync and position_sync. Lost on restart, but repopulated by the initial leaderboard sync on startup.

5. **Dedup via watermarks**: Same pattern as the existing `trade_collector` — track the latest timestamp per wallet to avoid re-inserting old activity.

6. **Profile enrichment as separate job**: Decouples the slow Gamma API profile lookups from the fast leaderboard ingestion. Processes a batch per cycle, eventually enriching all discovered wallets.

7. **`fetch_public_profile` uses Gamma API**: The profile endpoint is on the Gamma API, not the Data API. The method creates a separate httpx client for this one-off call.

8. **Position concentration via ClickHouse aggregation**: HHI and top-N share are computed at query time from `market_holders`. No materialized views needed given the small data volume (max 20 holders per market x 50 markets = 1000 rows).

9. **Tabs pattern**: Matches Phase 1 signals page. Pre-fetch all datasets server-side, pass to a client-side tab wrapper for instant switching.

10. **Nav with Users icon**: Distinguishes from Signals (Zap icon). "Whales" is the conventional term in crypto analytics.

---

## 16. Acceptance Criteria

- [ ] `002_phase2_users.sql` creates all 5 tables without errors on ClickHouse Cloud
- [ ] `docker compose up -d --build` starts pipeline with all 8 jobs (4 existing + 4 new)
- [ ] `curl http://localhost:8080/health` returns `tracked_wallets` count
- [ ] Leaderboard sync populates `trader_rankings` with data across categories
- [ ] Holder sync populates `market_holders` for top markets
- [ ] Position sync populates `wallet_positions` and `wallet_activity` for tracked wallets
- [ ] Profile enricher populates `trader_profiles` for discovered wallets
- [ ] `/whales` page loads with 4 stats cards showing live counts
- [ ] Leaderboard tab shows ranked traders sorted by PnL
- [ ] Activity tab shows whale trades from the last 24h
- [ ] Smart Money tab shows positions of top-ranked traders
- [ ] Concentration tab shows position concentration metrics per market
- [ ] Market detail page shows top holders section
- [ ] Sidebar navigation includes "Whales" link pointing to `/whales`
- [ ] `npm run build` succeeds with no type errors
- [ ] All existing pages (overview, signals, market detail) are unaffected
- [ ] Pipeline handles API errors gracefully (no crashes on 404/timeout)
