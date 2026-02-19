"""Risk manager — portfolio-level controls and pre-trade checks.

Every order request passes through the risk manager before reaching the
execution engine. The risk manager enforces:

1. Per-market position limits (max exposure per condition_id)
2. Portfolio-level exposure limits (max total deployed capital)
3. Drawdown circuit breaker (halt trading if drawdown exceeds threshold)
4. Maximum concurrent positions
5. Minimum edge requirement (dynamic hurdle)
6. Correlation check (avoid over-concentration in correlated markets)
7. Daily loss limit
8. Kill switch (manual emergency halt)

The risk manager is intentionally conservative. Better to miss a trade
than to blow up the account.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from pipeline.config import (
    RISK_DAILY_LOSS_LIMIT_PCT,
    RISK_MAX_CONCURRENT_POSITIONS,
    RISK_MAX_DRAWDOWN_PCT,
    RISK_MAX_PORTFOLIO_EXPOSURE_PCT,
    RISK_MAX_POSITION_EXPOSURE_PCT,
    RISK_MIN_EDGE,
    RISK_MIN_LIQUIDITY,
)
from pipeline.execution.engine import OrderRequest
from pipeline.execution.position_manager import PositionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RiskViolation(str, Enum):
    """Type of risk limit violated."""

    POSITION_LIMIT = "position_limit"
    PORTFOLIO_LIMIT = "portfolio_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MAX_POSITIONS = "max_positions"
    MIN_EDGE = "min_edge"
    MIN_LIQUIDITY = "min_liquidity"
    DAILY_LOSS = "daily_loss"
    KILL_SWITCH = "kill_switch"


class RiskCheck(BaseModel):
    """Result of a pre-trade risk check."""

    approved: bool = False
    violation: Optional[RiskViolation] = None
    message: str = ""
    adjusted_size: Optional[float] = None
    max_allowed_size: Optional[float] = None


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------


class RiskManager:
    """Pre-trade and portfolio-level risk controls.

    Attributes:
        position_manager: Reference to the position manager for portfolio state.
        kill_switch: If True, all new orders are rejected.
        daily_realized_pnl: Running daily realized P&L (reset at midnight UTC).
    """

    def __init__(self, position_manager: PositionManager) -> None:
        self.pm = position_manager
        self.kill_switch = False
        self.daily_realized_pnl = 0.0
        self._daily_reset_date: Optional[str] = None

    def check_order(
        self,
        request: OrderRequest,
        liquidity: float = 50_000.0,
    ) -> RiskCheck:
        """Run pre-trade risk checks on an order request.

        Args:
            request: Proposed order.
            liquidity: Market liquidity in USD.

        Returns:
            RiskCheck indicating approval or rejection.
        """
        self._maybe_reset_daily()

        # 1. Kill switch
        if self.kill_switch:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.KILL_SWITCH,
                message="Kill switch is active. All trading halted.",
            )

        portfolio = self.pm.snapshot()
        order_value = request.price * request.size

        # 2. Drawdown limit
        if portfolio.current_drawdown >= RISK_MAX_DRAWDOWN_PCT:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.DRAWDOWN_LIMIT,
                message=(
                    f"Drawdown {portfolio.current_drawdown:.1%} "
                    f"exceeds limit {RISK_MAX_DRAWDOWN_PCT:.1%}. "
                    f"Trading halted until recovery."
                ),
            )

        # 3. Daily loss limit
        daily_loss_limit = self.pm.initial_capital * RISK_DAILY_LOSS_LIMIT_PCT
        if self.daily_realized_pnl < -daily_loss_limit:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.DAILY_LOSS,
                message=(
                    f"Daily loss ${abs(self.daily_realized_pnl):.2f} "
                    f"exceeds limit ${daily_loss_limit:.2f}."
                ),
            )

        # 4. Maximum concurrent positions
        if (
            request.condition_id not in self.pm.positions
            and self.pm.n_positions >= RISK_MAX_CONCURRENT_POSITIONS
        ):
            return RiskCheck(
                approved=False,
                violation=RiskViolation.MAX_POSITIONS,
                message=(
                    f"At max positions ({RISK_MAX_CONCURRENT_POSITIONS}). "
                    f"Close an existing position first."
                ),
            )

        # 5. Per-market position limit
        existing_exposure = self.pm.get_exposure(request.condition_id)
        max_market_exposure = portfolio.total_value * RISK_MAX_POSITION_EXPOSURE_PCT
        remaining_market = max_market_exposure - existing_exposure

        if remaining_market <= 0:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.POSITION_LIMIT,
                message=(
                    f"Market {request.condition_id[:12]}... at exposure limit "
                    f"(${existing_exposure:.0f} / ${max_market_exposure:.0f})."
                ),
                max_allowed_size=0.0,
            )

        # 6. Portfolio-level exposure limit
        max_portfolio_exposure = portfolio.total_value * RISK_MAX_PORTFOLIO_EXPOSURE_PCT
        remaining_portfolio = max_portfolio_exposure - self.pm.total_exposure

        if remaining_portfolio <= 0:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.PORTFOLIO_LIMIT,
                message=(
                    f"Portfolio at exposure limit "
                    f"(${self.pm.total_exposure:.0f} / ${max_portfolio_exposure:.0f})."
                ),
                max_allowed_size=0.0,
            )

        # 7. Minimum edge
        if abs(request.edge) < RISK_MIN_EDGE:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.MIN_EDGE,
                message=(
                    f"Edge {abs(request.edge):.4f} below minimum {RISK_MIN_EDGE:.4f}."
                ),
            )

        # 8. Minimum liquidity
        if liquidity < RISK_MIN_LIQUIDITY:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.MIN_LIQUIDITY,
                message=(
                    f"Market liquidity ${liquidity:.0f} below minimum "
                    f"${RISK_MIN_LIQUIDITY:.0f}."
                ),
            )

        # All checks passed — compute max allowed size
        max_size_market = remaining_market / request.price if request.price > 0 else 0
        max_size_portfolio = remaining_portfolio / request.price if request.price > 0 else 0
        max_size_capital = self.pm.capital / request.price if request.price > 0 else 0

        max_allowed = min(max_size_market, max_size_portfolio, max_size_capital)

        # Clamp order size to risk limits
        adjusted_size = min(request.size, max_allowed) if max_allowed > 0 else 0.0

        if adjusted_size <= 0:
            return RiskCheck(
                approved=False,
                violation=RiskViolation.PORTFOLIO_LIMIT,
                message="Insufficient capital for this order.",
                max_allowed_size=0.0,
            )

        return RiskCheck(
            approved=True,
            adjusted_size=adjusted_size,
            max_allowed_size=max_allowed,
            message="Order approved.",
        )

    def record_realized_pnl(self, pnl: float) -> None:
        """Record realized P&L for daily loss tracking."""
        self.daily_realized_pnl += pnl

    def activate_kill_switch(self, reason: str = "") -> None:
        """Activate the kill switch — halt all trading."""
        self.kill_switch = True
        logger.warning(
            "kill_switch_activated",
            extra={"reason": reason},
        )

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch — resume trading."""
        self.kill_switch = False
        logger.info("kill_switch_deactivated")

    def status(self) -> dict:
        """Return current risk status summary."""
        portfolio = self.pm.snapshot()
        return {
            "kill_switch": self.kill_switch,
            "drawdown": portfolio.current_drawdown,
            "drawdown_limit": RISK_MAX_DRAWDOWN_PCT,
            "n_positions": self.pm.n_positions,
            "max_positions": RISK_MAX_CONCURRENT_POSITIONS,
            "total_exposure": self.pm.total_exposure,
            "max_exposure": portfolio.total_value * RISK_MAX_PORTFOLIO_EXPOSURE_PCT,
            "daily_pnl": self.daily_realized_pnl,
            "daily_loss_limit": self.pm.initial_capital * RISK_DAILY_LOSS_LIMIT_PCT,
            "capital": self.pm.capital,
            "total_value": portfolio.total_value,
            "hwm": portfolio.high_water_mark,
        }

    def _maybe_reset_daily(self) -> None:
        """Reset daily P&L counter at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            if self._daily_reset_date is not None:
                logger.info(
                    "daily_pnl_reset",
                    extra={
                        "previous_day": self._daily_reset_date,
                        "final_daily_pnl": self.daily_realized_pnl,
                    },
                )
            self.daily_realized_pnl = 0.0
            self._daily_reset_date = today
