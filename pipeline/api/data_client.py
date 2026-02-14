"""Client for the Polymarket Data API (trades, activity)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from pipeline.config import DATA_API_URL, GAMMA_API_URL, HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_LEADERBOARD_CATEGORIES = [
    "OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE",
    "MENTIONS", "WEATHER", "ECONOMICS", "TECH", "FINANCE",
]
_LEADERBOARD_TIME_PERIODS = ["DAY", "WEEK", "MONTH", "ALL"]
_LEADERBOARD_ORDER_TYPES = ["PNL", "VOL"]


class DataClient:
    """Fetch public trade data from the Data API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=DATA_API_URL,
            timeout=HTTP_TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_recent_trades(
        self,
        *,
        market: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict]:
        """GET /trades — fetch recent trades, optionally filtered by market.

        No auth required; uses public endpoint.
        """
        params: dict = {"limit": limit, "offset": offset}
        if market:
            params["market"] = market

        try:
            resp = await self._client.get("/trades", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("fetch_trades_error", extra={"market": market}, exc_info=True)
            return []

    async def fetch_all_recent_trades(
        self,
        *,
        market: str | None = None,
        max_pages: int = 5,
    ) -> list[dict]:
        """Paginate through recent trades up to max_pages pages."""
        all_trades: list[dict] = []
        offset = 0
        limit = 500

        for _ in range(max_pages):
            trades = await self.fetch_recent_trades(
                market=market, limit=limit, offset=offset,
            )
            if not trades:
                break
            all_trades.extend(trades)
            if len(trades) < limit:
                break
            offset += limit

        return all_trades

    @staticmethod
    def parse_trade(raw: dict) -> dict | None:
        """Convert a raw Data API trade into a schema-compatible dict.

        Expected raw keys: conditionId, asset, size, price, side,
        timestamp, outcome, transactionHash.
        """
        condition_id = raw.get("conditionId")
        if not condition_id:
            return None

        side_raw = (raw.get("side") or "").upper()
        side = "buy" if side_raw == "BUY" else "sell"

        ts_raw = raw.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            ts = datetime.now(timezone.utc)

        # Use transactionHash as trade_id (unique per on-chain tx)
        trade_id = raw.get("transactionHash") or ""

        return {
            "condition_id": condition_id,
            "token_id": raw.get("asset", ""),
            "outcome": raw.get("outcome", ""),
            "price": float(raw.get("price") or 0),
            "size": float(raw.get("size") or 0),
            "side": side,
            "trade_id": trade_id,
            "timestamp": ts,
        }

    # ------------------------------------------------------------------
    # Phase 2: Leaderboard
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Positions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Activity
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Holders
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Portfolio value
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Public profile (Gamma API)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Phase 2: Parse helpers
    # ------------------------------------------------------------------

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
