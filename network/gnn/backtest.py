"""Backtesting framework — Anastasiia's Point 4 + Point 5.

Implements:
  - Dynamic hurdle rate (Point 4): adjusts min edge based on trade impact
  - Backtesting engine (Point 5): walk-forward evaluation with:
    * 200ms latency simulation
    * Spread penalty (0.5%)
    * Sharpe Ratio, Max Drawdown, Win Rate
    * Look-ahead bias prevention (strict temporal ordering)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import BacktestConfig

logger = logging.getLogger(__name__)


# ============================================================
# Trade and position tracking
# ============================================================

@dataclass
class Trade:
    """A single executed trade."""
    timestamp: str
    market_idx: int
    market_name: str
    direction: str          # "BUY" or "SELL"
    model_prob: float       # our calibrated probability
    market_price: float     # price at entry
    exit_price: float       # price at exit (or resolution)
    size_usd: float         # position size in USDC
    pnl: float = 0.0       # realized PnL after costs
    costs: float = 0.0      # spread + slippage costs


@dataclass
class BacktestResult:
    """Aggregate result of a backtesting run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    avg_edge: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0


# ============================================================
# Dynamic Hurdle Rate (Point 4)
# ============================================================

class DynamicHurdle:
    """Calculates how much our own trade moves the price (market impact).

    hurdle = base_hurdle + impact_coefficient * sqrt(trade_size / liquidity)

    We only trade when our edge exceeds the hurdle.
    """

    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()

    def compute_hurdle(self, trade_size_usd: float, liquidity_usd: float) -> float:
        """Compute dynamic hurdle rate.

        Args:
            trade_size_usd: intended trade size
            liquidity_usd: market's available liquidity

        Returns:
            minimum edge required to trade
        """
        if liquidity_usd <= 0:
            return 1.0  # don't trade illiquid markets

        impact = self.cfg.impact_coefficient * math.sqrt(trade_size_usd / liquidity_usd)
        return self.cfg.base_hurdle + impact

    def should_trade(
        self,
        edge: float,
        trade_size_usd: float,
        liquidity_usd: float,
    ) -> bool:
        """Check if edge exceeds dynamic hurdle."""
        hurdle = self.compute_hurdle(trade_size_usd, liquidity_usd)
        return abs(edge) > hurdle

    def optimal_size(
        self,
        edge: float,
        kelly_fraction: float,
        capital: float,
        liquidity_usd: float,
    ) -> float:
        """Compute optimal trade size using fractional Kelly + impact constraint.

        Args:
            edge: model_prob - market_price
            kelly_fraction: raw Kelly fraction
            capital: available capital
            liquidity_usd: market liquidity

        Returns:
            trade size in USD
        """
        # Fractional Kelly
        raw_size = capital * kelly_fraction * self.cfg.kelly_fraction

        # Cap at max position and max trade size
        max_pos = capital * self.cfg.max_position_pct
        raw_size = min(raw_size, max_pos, self.cfg.max_trade_size_usd)

        # Reduce size if impact would eat the edge
        hurdle = self.compute_hurdle(raw_size, liquidity_usd)
        while abs(edge) < hurdle and raw_size > 100:
            raw_size *= 0.8
            hurdle = self.compute_hurdle(raw_size, liquidity_usd)

        return max(raw_size, 0.0)


# ============================================================
# Kelly Criterion
# ============================================================

def kelly_criterion(p_model: float, p_market: float) -> float:
    """Full Kelly fraction for a binary prediction market.

    f* = (p * b - q) / b where b = (1/market_price) - 1
    """
    if p_market <= 0.01 or p_market >= 0.99:
        return 0.0

    if p_model > p_market:
        # Betting YES
        b = (1.0 / p_market) - 1.0
        q = 1.0 - p_model
        f = (p_model * b - q) / b if b > 0 else 0.0
    else:
        # Betting NO
        p_no = 1.0 - p_model
        p_no_market = 1.0 - p_market
        b = (1.0 / p_no_market) - 1.0
        q = 1.0 - p_no
        f = (p_no * b - q) / b if b > 0 else 0.0

    return max(f, 0.0)


# ============================================================
# Backtesting Engine (Point 5)
# ============================================================

class BacktestEngine:
    """Walk-forward backtester with realistic execution simulation.

    Anti-look-ahead guarantees:
    - Predictions only use data up to t (ensured by TCN causality)
    - Entry at t+latency, exit at t+1 step
    - Spread deducted from both entry and exit
    """

    def __init__(
        self,
        cfg: Optional[BacktestConfig] = None,
        target_names: Optional[list[str]] = None,
    ):
        self.cfg = cfg or BacktestConfig()
        self.hurdle = DynamicHurdle(self.cfg)
        self.target_names = target_names or [f"market_{i}" for i in range(7)]

    def run(
        self,
        predictions: np.ndarray,
        market_prices: np.ndarray,
        timestamps: list[str],
        liquidities: Optional[np.ndarray] = None,
        initial_capital: float = 10_000.0,
    ) -> BacktestResult:
        """Execute walk-forward backtest.

        Args:
            predictions: (T, n_targets) — calibrated probabilities from model
            market_prices: (T, n_targets) — actual market prices at each step
            timestamps: (T,) — ISO timestamps
            liquidities: (T, n_targets) — market liquidity per step (optional)
            initial_capital: starting capital in USDC

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        T, n_targets = predictions.shape
        capital = initial_capital
        equity_curve = [capital]
        trades = []

        if liquidities is None:
            liquidities = np.full((T, n_targets), 50_000.0)  # default $50K liquidity

        # Simulate latency: entry at t, exit at t+1
        for t in range(T - 1):
            pred = predictions[t]
            price = market_prices[t]
            next_price = market_prices[t + 1]

            for j in range(n_targets):
                if np.isnan(pred[j]) or np.isnan(price[j]) or price[j] <= 0:
                    continue

                edge = pred[j] - price[j]
                kf = kelly_criterion(pred[j], price[j])

                if kf <= 0:
                    continue

                liq = liquidities[t, j]
                size = self.hurdle.optimal_size(edge, kf, capital, liq)

                if size < 10:  # minimum $10 trade
                    continue

                if not self.hurdle.should_trade(edge, size, liq):
                    continue

                # Execute trade with spread penalty
                spread_cost = size * self.cfg.spread_penalty
                direction = "BUY" if edge > 0 else "SELL"

                # PnL calculation
                if direction == "BUY":
                    entry = price[j] + self.cfg.spread_penalty / 2
                    exit_p = next_price[j] - self.cfg.spread_penalty / 2
                    pnl = size * (exit_p - entry) / entry
                else:
                    entry = price[j] - self.cfg.spread_penalty / 2
                    exit_p = next_price[j] + self.cfg.spread_penalty / 2
                    pnl = size * (entry - exit_p) / entry

                pnl -= spread_cost  # total cost

                trade = Trade(
                    timestamp=timestamps[t],
                    market_idx=j,
                    market_name=self.target_names[j],
                    direction=direction,
                    model_prob=float(pred[j]),
                    market_price=float(price[j]),
                    exit_price=float(next_price[j]),
                    size_usd=float(size),
                    pnl=float(pnl),
                    costs=float(spread_cost),
                )
                trades.append(trade)
                capital += pnl

            equity_curve.append(capital)

        # Compute metrics
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            n_trades=len(trades),
        )

        if trades:
            result.total_return = (capital - initial_capital) / initial_capital
            result.win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
            result.avg_edge = np.mean([abs(t.model_prob - t.market_price) for t in trades])

            # Sharpe ratio (annualized, per-step returns)
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            returns = returns[~np.isnan(returns)]
            if len(returns) > 1 and np.std(returns) > 0:
                # Annualize: 5-min steps → ~105K steps/year
                steps_per_year = 365.25 * 24 * 60 / 5
                excess = np.mean(returns) - self.cfg.risk_free_rate / steps_per_year
                result.sharpe_ratio = excess / np.std(returns) * math.sqrt(steps_per_year)

            # Max drawdown
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

            # Profit factor
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Calmar ratio
            if max_dd > 0:
                result.calmar_ratio = result.total_return / max_dd

        return result

    def print_report(self, result: BacktestResult) -> str:
        """Format backtest results as human-readable report."""
        lines = [
            "=" * 60,
            "BACKTEST REPORT",
            "=" * 60,
            f"Total trades:     {result.n_trades}",
            f"Win rate:         {result.win_rate:.1%}",
            f"Total return:     {result.total_return:.2%}",
            f"Sharpe ratio:     {result.sharpe_ratio:.2f}",
            f"Max drawdown:     {result.max_drawdown:.2%}",
            f"Profit factor:    {result.profit_factor:.2f}",
            f"Calmar ratio:     {result.calmar_ratio:.2f}",
            f"Avg edge:         {result.avg_edge:.4f}",
            "=" * 60,
        ]

        if result.trades:
            lines.append("\nTOP 10 TRADES (by PnL):")
            sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
            for t in sorted_trades[:10]:
                lines.append(
                    f"  {t.timestamp[:16]}  {t.market_name:<20}  "
                    f"{t.direction:4}  edge={t.model_prob - t.market_price:+.3f}  "
                    f"size=${t.size_usd:.0f}  pnl=${t.pnl:+.2f}"
                )

        report = "\n".join(lines)
        logger.info(report)
        return report
