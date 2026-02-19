"""Position manager — tracks open positions, fills, and P&L.

Maintains an in-memory position book that is reconciled against the CLOB
on each cycle. Computes per-position and portfolio-level P&L metrics used
by the risk manager for drawdown monitoring and position limits.

The position book is also persisted to ClickHouse for historical analysis.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from pipeline.execution.engine import ExecutionEngine, OrderResult, OrderStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """A single open position in a market."""

    condition_id: str = Field(..., description="Market condition ID.")
    token_id: str = Field(..., description="Token ID (Yes or No).")
    side: str = Field(..., description="BUY or SELL.")
    entry_price: float = Field(..., description="Volume-weighted average entry price.")
    size: float = Field(default=0.0, description="Current position size (tokens).")
    cost_basis: float = Field(default=0.0, description="Total cost in USDC.")
    current_price: float = Field(default=0.0, description="Latest market price.")
    unrealized_pnl: float = Field(default=0.0, description="Mark-to-market P&L.")
    realized_pnl: float = Field(default=0.0, description="P&L from closed portions.")
    order_ids: list[str] = Field(default_factory=list, description="Associated order IDs.")
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signal_source: str = Field(default="ensemble")
    edge_at_entry: float = Field(default=0.0, description="Edge when position opened.")

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.size * self.current_price

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of cost basis."""
        if self.cost_basis <= 0:
            return 0.0
        return self.total_pnl / self.cost_basis

    def mark_to_market(self, price: float) -> None:
        """Update position with current market price."""
        self.current_price = price
        if self.side == "BUY":
            self.unrealized_pnl = self.size * (price - self.entry_price)
        else:
            self.unrealized_pnl = self.size * (self.entry_price - price)
        self.updated_at = datetime.now(timezone.utc)


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio state."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capital: float = Field(default=0.0, description="Available capital (USDC).")
    total_value: float = Field(default=0.0, description="Capital + unrealized positions.")
    n_positions: int = Field(default=0, description="Number of open positions.")
    total_unrealized_pnl: float = Field(default=0.0)
    total_realized_pnl: float = Field(default=0.0)
    total_cost_basis: float = Field(default=0.0)
    max_position_value: float = Field(default=0.0, description="Largest position by value.")
    high_water_mark: float = Field(default=0.0, description="Historical peak total value.")
    current_drawdown: float = Field(default=0.0, description="Current drawdown from HWM.")

    @property
    def deployed_pct(self) -> float:
        """Fraction of capital deployed in positions."""
        if self.total_value <= 0:
            return 0.0
        return self.total_cost_basis / self.total_value


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------


class PositionManager:
    """Manages the portfolio of open positions and computes P&L.

    Responsibilities:
    - Track position entries from fill reports
    - Mark positions to market on each cycle
    - Compute portfolio-level P&L metrics
    - Reconcile with CLOB open orders / trades
    - Persist position state to ClickHouse

    Attributes:
        positions: Open positions keyed by condition_id.
        capital: Available USDC capital.
        high_water_mark: Historical peak portfolio value.
        total_realized_pnl: Cumulative realized P&L.
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.positions: dict[str, Position] = {}
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.high_water_mark = initial_capital
        self.total_realized_pnl = 0.0
        self._snapshots: list[PortfolioSnapshot] = []

    # ------------------------------------------------------------------
    # Position updates
    # ------------------------------------------------------------------

    def record_fill(self, result: OrderResult) -> Position:
        """Record a filled order as a position entry or update.

        Args:
            result: OrderResult from the execution engine.

        Returns:
            The created or updated Position.
        """
        req = result.request
        cid = req.condition_id

        fill_price = result.fill_price or req.price
        fill_size = result.filled_size or req.size

        if cid in self.positions:
            pos = self.positions[cid]
            # Update volume-weighted average entry price
            if pos.side == req.side:
                total_cost = pos.entry_price * pos.size + fill_price * fill_size
                pos.size += fill_size
                pos.entry_price = total_cost / pos.size if pos.size > 0 else fill_price
                pos.cost_basis += fill_price * fill_size
            else:
                # Reducing position (opposite side)
                close_size = min(fill_size, pos.size)
                if pos.side == "BUY":
                    realized = close_size * (fill_price - pos.entry_price)
                else:
                    realized = close_size * (pos.entry_price - fill_price)
                pos.realized_pnl += realized
                self.total_realized_pnl += realized
                self.capital += realized + close_size * pos.entry_price
                pos.size -= close_size
                pos.cost_basis = pos.entry_price * pos.size

                remaining = fill_size - close_size
                if remaining > 0 and pos.size <= 0:
                    # Flipped position
                    pos.side = req.side
                    pos.entry_price = fill_price
                    pos.size = remaining
                    pos.cost_basis = fill_price * remaining
                    pos.realized_pnl = 0.0

            pos.order_ids.append(result.order_id)
            pos.updated_at = datetime.now(timezone.utc)
        else:
            cost = fill_price * fill_size
            self.capital -= cost
            pos = Position(
                condition_id=cid,
                token_id=req.token_id,
                side=req.side,
                entry_price=fill_price,
                size=fill_size,
                cost_basis=cost,
                current_price=fill_price,
                order_ids=[result.order_id],
                signal_source=req.signal_source,
                edge_at_entry=req.edge,
            )
            self.positions[cid] = pos

        # Remove empty positions
        if pos.size <= 0.001:
            self.positions.pop(cid, None)

        logger.info(
            "fill_recorded",
            extra={
                "condition_id": cid,
                "side": req.side,
                "fill_price": fill_price,
                "fill_size": fill_size,
                "n_positions": len(self.positions),
                "capital": round(self.capital, 2),
            },
        )

        return pos

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update all positions with current market prices.

        Args:
            prices: Mapping of condition_id -> latest price.
        """
        for cid, pos in self.positions.items():
            if cid in prices:
                pos.mark_to_market(prices[cid])

    def close_position(self, condition_id: str, exit_price: float) -> float:
        """Close a position and realize P&L.

        Args:
            condition_id: Market to close.
            exit_price: Price at which position is closed.

        Returns:
            Realized P&L from the closure.
        """
        pos = self.positions.get(condition_id)
        if pos is None:
            return 0.0

        if pos.side == "BUY":
            realized = pos.size * (exit_price - pos.entry_price)
        else:
            realized = pos.size * (pos.entry_price - exit_price)

        pos.realized_pnl += realized
        self.total_realized_pnl += realized
        self.capital += realized + pos.size * pos.entry_price

        logger.info(
            "position_closed",
            extra={
                "condition_id": condition_id,
                "side": pos.side,
                "entry": pos.entry_price,
                "exit": exit_price,
                "size": pos.size,
                "realized_pnl": round(realized, 2),
            },
        )

        del self.positions[condition_id]
        return realized

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        """Take a point-in-time portfolio snapshot.

        Updates high water mark and returns comprehensive metrics.

        Returns:
            Current portfolio state.
        """
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_cost = sum(p.cost_basis for p in self.positions.values())
        max_pos_value = max(
            (p.market_value for p in self.positions.values()),
            default=0.0,
        )

        total_value = self.capital + sum(
            p.market_value for p in self.positions.values()
        )

        if total_value > self.high_water_mark:
            self.high_water_mark = total_value

        drawdown = 0.0
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - total_value) / self.high_water_mark

        snap = PortfolioSnapshot(
            capital=self.capital,
            total_value=total_value,
            n_positions=len(self.positions),
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=self.total_realized_pnl,
            total_cost_basis=total_cost,
            max_position_value=max_pos_value,
            high_water_mark=self.high_water_mark,
            current_drawdown=drawdown,
        )

        self._snapshots.append(snap)
        return snap

    def get_position(self, condition_id: str) -> Optional[Position]:
        """Get a specific position."""
        return self.positions.get(condition_id)

    def get_exposure(self, condition_id: str) -> float:
        """Get total exposure (cost basis) for a market.

        Returns 0.0 if no position exists.
        """
        pos = self.positions.get(condition_id)
        return pos.cost_basis if pos else 0.0

    @property
    def total_exposure(self) -> float:
        """Total cost basis across all positions."""
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def n_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    @property
    def equity_curve(self) -> list[float]:
        """Historical portfolio values from snapshots."""
        return [s.total_value for s in self._snapshots]

    def to_rows(self) -> list[list]:
        """Convert positions to ClickHouse-writable rows.

        Returns:
            List of rows for the execution_positions table.
        """
        now = datetime.now(timezone.utc)
        rows = []
        for pos in self.positions.values():
            rows.append([
                pos.condition_id,
                pos.token_id,
                pos.side,
                pos.entry_price,
                pos.size,
                pos.cost_basis,
                pos.current_price,
                pos.unrealized_pnl,
                pos.realized_pnl,
                pos.signal_source,
                pos.edge_at_entry,
                pos.opened_at,
                now,
            ])
        return rows
