"""ClickHouse bridge for reading market prices and writing network signals.

TODO Phase 2:
- Read live Polymarket prices from market_prices / markets tables
- Write vulnerability scores, supply risk, cascade results, signals
- Map polymarket_mapping.json condition_ids to live market data
"""

from __future__ import annotations

from typing import Optional


class ClickHouseBridge:
    """Stub: bridge between network model and ClickHouse pipeline data."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8443,
        user: str = "default",
        password: str = "",
        database: str = "polymarket",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def get_market_price(self, condition_id: str) -> Optional[float]:
        """Get latest price for a market from ClickHouse."""
        raise NotImplementedError("ClickHouse bridge not yet implemented")

    def get_market_prices_bulk(self, condition_ids: list[str]) -> dict[str, float]:
        """Get latest prices for multiple markets."""
        raise NotImplementedError("ClickHouse bridge not yet implemented")

    def write_vulnerability_scores(self, scores: list[dict]) -> None:
        """Write vulnerability scores to network_vulnerability table."""
        raise NotImplementedError("ClickHouse bridge not yet implemented")

    def write_signals(self, signals: list[dict]) -> None:
        """Write trading signals to network_signals table."""
        raise NotImplementedError("ClickHouse bridge not yet implemented")
