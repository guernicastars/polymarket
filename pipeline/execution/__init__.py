"""Execution layer for the Polymarket pipeline.

This package bridges the gap between signal generation (composite signals,
GNN predictions, causal analysis) and actual market participation. It handles
order placement, position tracking, risk management, and portfolio-level
controls.

The execution layer is intentionally conservative: quarter-Kelly sizing,
hard drawdown stops, per-market and portfolio-level position limits, and
a kill switch for emergency shutdown. The goal is to survive first and
profit second.

Modules:
    engine          -- Order execution engine wrapping py-clob-client
    position_manager -- Open position tracking, P&L, and fill handling
    risk_manager    -- Portfolio risk limits, drawdown stops, correlation checks
"""

from pipeline.execution.engine import ExecutionEngine, OrderRequest, OrderResult, OrderStatus
from pipeline.execution.position_manager import Position, PositionManager, PortfolioSnapshot
from pipeline.execution.risk_manager import RiskManager, RiskCheck, RiskViolation

__all__ = [
    "ExecutionEngine",
    "OrderRequest",
    "OrderResult",
    "OrderStatus",
    "Position",
    "PositionManager",
    "PortfolioSnapshot",
    "RiskManager",
    "RiskCheck",
    "RiskViolation",
]
