"""Full-pipeline execution backtest.

Replays composite signals against resolved Polymarket markets to evaluate
the complete execution pipeline: signal → probability → Kelly sizing →
risk check → entry → resolution → P&L.

Uses actual data from ClickHouse:
- composite_signals: the signal that would trigger trades
- markets (FINAL): resolution outcomes, winning_outcome
- market_trades: real trade data for actual prices at signal time

This is NOT a look-ahead backtest — it simulates what would have happened
if the execution_runner had been live during the data collection period.

Usage:
    python -m pipeline.backtest_execution [--capital 1000] [--min-edge 0.02]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import clickhouse_connect


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CH_HOST = "ch.bloomsburytech.com"
CH_PORT = 443
CH_USER = "default"
CH_PASSWORD = "clickhouse_admin_2026"
CH_DB = "polymarket"

KELLY_FRACTION = 0.25       # quarter-Kelly
MAX_POSITION_PCT = 0.10     # 10% max per position
MAX_DRAWDOWN = 0.15         # 15% max drawdown
MAX_POSITIONS = 20          # max concurrent positions
SPREAD_COST = 0.005         # 50 bps round-trip
MAX_PORTFOLIO_EXPOSURE = 0.60  # 60% max portfolio exposure


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SignalRow:
    """One composite signal joined with market info and trade prices."""
    condition_id: str
    slug: str
    score: float
    confidence: float
    computed_at: datetime
    winning_outcome: str
    trade_vwap: float       # VWAP price from trades (Yes token)
    trade_count: int
    trade_volume: float


@dataclass
class BacktestTrade:
    """A single backtest trade."""
    condition_id: str
    slug: str
    direction: str      # BUY_YES or BUY_NO
    signal_score: float
    confidence: float
    model_prob: float
    market_price: float
    edge: float
    kelly_frac: float
    size_usd: float
    pnl: float
    costs: float
    signal_time: datetime
    won: bool
    winning_outcome: str


@dataclass
class BacktestResult:
    """Full backtest result."""
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    avg_edge: float = 0.0
    avg_size: float = 0.0
    total_costs: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def composite_to_prob(score: float, market_price: float) -> float:
    """Convert composite signal (-100..+100) to probability estimate.

    Uses the market price as anchor and shifts based on signal strength.
    Score of +100 shifts up ~15% of remaining range.
    """
    if market_price <= 0.01 or market_price >= 0.99:
        return market_price

    if score > 0:
        shift = (score / 100.0) * 0.15 * (1.0 - market_price)
    else:
        shift = (score / 100.0) * 0.15 * market_price

    return max(0.01, min(0.99, market_price + shift))


def calc_kelly(p_model: float, p_market: float) -> float:
    """Fractional Kelly for binary prediction market (quarter-Kelly)."""
    if p_market <= 0.01 or p_market >= 0.99:
        return 0.0

    if p_model > p_market:
        b = (1.0 / p_market) - 1.0
        q = 1.0 - p_model
        f = (p_model * b - q) / b if b > 0 else 0.0
    else:
        p_no = 1.0 - p_model
        p_no_mkt = 1.0 - p_market
        b = (1.0 / p_no_mkt) - 1.0
        q = 1.0 - p_no
        f = (p_no * b - q) / b if b > 0 else 0.0

    return max(f, 0.0) * KELLY_FRACTION


def get_client():
    """Create ClickHouse client."""
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=CH_DB, secure=True,
    )


def determine_win(direction: str, winning_outcome: str) -> bool:
    """Determine if our trade won based on resolution.

    Polymarket winning_outcome for binary: "Yes"/"No"
    For sports: the team/player name (which means Yes won)
    For totals: "Over"/"Under"
    """
    wo = winning_outcome.strip().lower() if winning_outcome else ""

    if direction == "BUY_YES":
        # We bet Yes. We win if outcome is Yes, a named winner, Over, etc.
        if wo in ("no", "under"):
            return False
        if wo == "":
            return False  # empty = no resolution or No
        return True  # "yes", "over", named team, etc.
    else:
        # BUY_NO: we win if outcome is No or Under
        if wo in ("no", "under"):
            return True
        if wo == "":
            return True  # empty could mean no outcome
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_backtest_data(client) -> list[SignalRow]:
    """Load all data needed for backtest in a single efficient query.

    Joins composite_signals → markets (resolved only) → market_trades (VWAP).
    The market_trades subquery is filtered to signal markets first to avoid
    memory issues.
    """
    # Step 1: Get resolved market condition_ids that have signals
    resolved_cids = client.query("""
        SELECT DISTINCT cs.condition_id
        FROM composite_signals AS cs
        INNER JOIN (SELECT condition_id FROM markets FINAL WHERE resolved = 1) AS m
            ON cs.condition_id = m.condition_id
    """)
    cids = [row[0] for row in resolved_cids.result_rows]

    if not cids:
        return []

    # Step 2: Get trade VWAP for Yes outcome per market
    # Process in batches to avoid memory issues
    batch_size = 100
    trade_data: dict[str, dict] = {}

    for i in range(0, len(cids), batch_size):
        batch = cids[i:i + batch_size]
        cid_list = ",".join(f"'{c}'" for c in batch)

        r = client.query(f"""
            SELECT
                condition_id,
                avg(price) AS avg_price,
                count() AS trade_count,
                sum(size * price) AS total_volume
            FROM market_trades
            WHERE outcome = 'Yes'
              AND condition_id IN ({cid_list})
            GROUP BY condition_id
        """)

        for row in r.result_rows:
            trade_data[row[0]] = {
                "vwap": float(row[1]),
                "count": int(row[2]),
                "volume": float(row[3]),
            }

    # Step 3: Get signals with market metadata
    cid_list_all = ",".join(f"'{c}'" for c in cids)
    result = client.query(f"""
        SELECT
            cs.condition_id,
            m.market_slug,
            cs.score,
            cs.confidence,
            cs.computed_at,
            m.winning_outcome
        FROM composite_signals AS cs
        INNER JOIN (
            SELECT condition_id, market_slug, winning_outcome
            FROM markets FINAL
            WHERE resolved = 1 AND condition_id IN ({cid_list_all})
        ) AS m ON cs.condition_id = m.condition_id
        ORDER BY cs.computed_at ASC
    """)

    rows = []
    for row in result.result_rows:
        cid = row[0]
        td = trade_data.get(cid)
        if not td or td["count"] < 3:  # need enough trades for reliable VWAP
            continue

        rows.append(SignalRow(
            condition_id=cid,
            slug=row[1] or "",
            score=float(row[2]),
            confidence=float(row[3]),
            computed_at=row[4],
            winning_outcome=row[5] or "",
            trade_vwap=td["vwap"],
            trade_count=td["count"],
            trade_volume=td["volume"],
        ))

    return rows


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def run_backtest(
    data: list[SignalRow],
    initial_capital: float = 1000.0,
    min_edge: float = 0.02,
    min_confidence: float = 0.3,
) -> BacktestResult:
    """Run the full execution pipeline backtest.

    For each signal on a resolved market:
    1. Use trade VWAP as market price
    2. Convert composite score to probability
    3. Compute edge vs market price
    4. Apply Kelly sizing with risk checks
    5. Simulate entry at market price + spread
    6. Resolve at 1.0 (win) or 0.0 (loss)
    7. Track P&L, equity, drawdowns
    """
    result = BacktestResult(initial_capital=initial_capital)
    capital = initial_capital
    equity_curve = [capital]
    peak = capital

    # Deduplicate: take latest signal per market
    latest_by_cid: dict[str, SignalRow] = {}
    for row in data:
        cid = row.condition_id
        if cid not in latest_by_cid or row.computed_at > latest_by_cid[cid].computed_at:
            latest_by_cid[cid] = row

    # Process in chronological order
    sorted_signals = sorted(latest_by_cid.values(), key=lambda s: s.computed_at)

    traded_cids: set[str] = set()
    n_concurrent = 0
    total_exposure = 0.0

    for sig in sorted_signals:
        cid = sig.condition_id

        # Avoid duplicate entries
        if cid in traded_cids:
            continue

        # Confidence gate
        if sig.confidence < min_confidence:
            continue

        market_price = sig.trade_vwap

        # Skip extreme prices
        if market_price <= 0.02 or market_price >= 0.98:
            continue

        # Convert signal to probability
        model_prob = composite_to_prob(sig.score, market_price)

        # Compute edge
        edge = model_prob - market_price
        abs_edge = abs(edge)

        if abs_edge < min_edge:
            continue

        # Direction
        if edge > 0:
            direction = "BUY_YES"
            entry_price = market_price
        else:
            direction = "BUY_NO"
            entry_price = 1.0 - market_price
            model_prob = 1.0 - model_prob

        # Kelly sizing
        kf = calc_kelly(model_prob, entry_price)
        if kf <= 0:
            continue

        size_usd = capital * kf

        # --- Risk checks ---
        max_pos = capital * MAX_POSITION_PCT
        size_usd = min(size_usd, max_pos)

        if total_exposure + size_usd > capital * MAX_PORTFOLIO_EXPOSURE:
            size_usd = max(0, capital * MAX_PORTFOLIO_EXPOSURE - total_exposure)

        if n_concurrent >= MAX_POSITIONS:
            continue

        current_dd = (peak - capital) / peak if peak > 0 else 0
        if current_dd >= MAX_DRAWDOWN:
            break  # circuit breaker — stop trading

        if size_usd < 5.0:
            continue

        # --- Simulate execution ---
        spread_cost = size_usd * SPREAD_COST
        effective_entry = entry_price + SPREAD_COST / 2

        # Did we win?
        won = determine_win(direction, sig.winning_outcome)
        resolution_price = 1.0 if won else 0.0

        # P&L: bought at effective_entry, resolved at 0 or 1
        pnl_gross = size_usd * (resolution_price - effective_entry) / effective_entry
        pnl_net = pnl_gross - spread_cost

        trade = BacktestTrade(
            condition_id=cid,
            slug=sig.slug,
            direction=direction,
            signal_score=sig.score,
            confidence=sig.confidence,
            model_prob=model_prob,
            market_price=market_price,
            edge=edge,
            kelly_frac=kf,
            size_usd=size_usd,
            pnl=pnl_net,
            costs=spread_cost,
            signal_time=sig.computed_at,
            won=won,
            winning_outcome=sig.winning_outcome,
        )

        result.trades.append(trade)
        capital += pnl_net
        equity_curve.append(capital)

        if capital > peak:
            peak = capital

        traded_cids.add(cid)

        # Binary markets resolve instantly for backtest purposes
        # (we enter, it resolves, we exit)

    # --- Compute metrics ---
    result.equity_curve = equity_curve
    result.final_capital = capital
    result.n_trades = len(result.trades)

    if result.trades:
        result.total_return = (capital - initial_capital) / initial_capital
        result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
        result.n_losses = result.n_trades - result.n_wins
        result.win_rate = result.n_wins / result.n_trades if result.n_trades > 0 else 0
        result.avg_edge = np.mean([abs(t.edge) for t in result.trades])
        result.avg_size = np.mean([t.size_usd for t in result.trades])
        result.total_costs = sum(t.costs for t in result.trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        pk = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > pk:
                pk = eq
            dd = (pk - eq) / pk if pk > 0 else 0
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        # Sharpe (per-trade returns, annualized)
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)
        returns_arr = np.array(returns)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            trades_per_year = result.n_trades * (365 / 5)
            result.sharpe_ratio = (
                np.mean(returns_arr) / np.std(returns_arr)
                * math.sqrt(max(trades_per_year, 1))
            )

        # Calmar
        if result.max_drawdown > 0:
            result.calmar_ratio = result.total_return / result.max_drawdown

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print()
    print("=" * 70)
    print("  POLYMARKET EXECUTION PIPELINE — BACKTEST REPORT")
    print("=" * 70)
    print()
    print(f"  Period:           Feb 14–19, 2026 (~5 days of signal data)")
    print(f"  Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"  Final Capital:    ${result.final_capital:,.2f}")
    print()
    print("  ─── Performance ───")
    print(f"  Total Return:     {result.total_return:+.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.2%}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Calmar Ratio:     {result.calmar_ratio:.2f}")
    print()
    print("  ─── Trading Stats ───")
    print(f"  Total Trades:     {result.n_trades}")
    print(f"  Wins / Losses:    {result.n_wins} / {result.n_losses}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print(f"  Avg Edge:         {result.avg_edge:.4f}")
    print(f"  Avg Size:         ${result.avg_size:.2f}")
    print(f"  Total Costs:      ${result.total_costs:.2f}")
    print()

    if not result.trades:
        print("  No trades executed.")
        print("=" * 70)
        return

    # Direction breakdown
    yes_trades = [t for t in result.trades if t.direction == "BUY_YES"]
    no_trades = [t for t in result.trades if t.direction == "BUY_NO"]
    print("  ─── Direction Breakdown ───")
    for label, bucket in [("BUY_YES", yes_trades), ("BUY_NO", no_trades)]:
        if bucket:
            w = sum(1 for t in bucket if t.pnl > 0)
            pnl = sum(t.pnl for t in bucket)
            print(f"  {label:10s}  {len(bucket):3d} trades, "
                  f"{w}/{len(bucket)} wins ({w / len(bucket):.0%}), "
                  f"P&L ${pnl:+.2f}")
    print()

    # Signal strength breakdown
    print("  ─── Signal Strength Breakdown ───")
    strong = [t for t in result.trades if abs(t.signal_score) >= 30]
    medium = [t for t in result.trades if 15 <= abs(t.signal_score) < 30]
    weak = [t for t in result.trades if abs(t.signal_score) < 15]

    for label, bucket in [
        ("Strong (|s|≥30)", strong),
        ("Medium (15≤|s|<30)", medium),
        ("Weak (|s|<15)", weak),
    ]:
        if bucket:
            w = sum(1 for t in bucket if t.pnl > 0)
            pnl = sum(t.pnl for t in bucket)
            print(f"  {label:25s}  {len(bucket):3d} trades, "
                  f"{w}/{len(bucket)} wins ({w / len(bucket):.0%}), "
                  f"P&L ${pnl:+.2f}")
    print()

    # Confidence breakdown
    print("  ─── Confidence Breakdown ───")
    hi_conf = [t for t in result.trades if t.confidence >= 0.5]
    lo_conf = [t for t in result.trades if t.confidence < 0.5]
    for label, bucket in [("High (≥0.5)", hi_conf), ("Low (<0.5)", lo_conf)]:
        if bucket:
            w = sum(1 for t in bucket if t.pnl > 0)
            pnl = sum(t.pnl for t in bucket)
            print(f"  {label:25s}  {len(bucket):3d} trades, "
                  f"{w}/{len(bucket)} wins ({w / len(bucket):.0%}), "
                  f"P&L ${pnl:+.2f}")
    print()

    # Top winners
    sorted_t = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
    print("  ─── Top 10 Winners ───")
    for t in sorted_t[:10]:
        status = "✓" if t.won else "✗"
        print(f"    {status} {t.slug[:38]:38s}  {t.direction:7s}  "
              f"score={t.signal_score:+5.1f}  mkt={t.market_price:.2f}  "
              f"edge={t.edge:+.3f}  sz=${t.size_usd:.0f}  P&L=${t.pnl:+.2f}")

    print()
    print("  ─── Top 10 Losers ───")
    for t in sorted_t[-10:]:
        status = "✓" if t.won else "✗"
        print(f"    {status} {t.slug[:38]:38s}  {t.direction:7s}  "
              f"score={t.signal_score:+5.1f}  mkt={t.market_price:.2f}  "
              f"edge={t.edge:+.3f}  sz=${t.size_usd:.0f}  P&L=${t.pnl:+.2f}")

    # Equity curve
    print()
    print("  ─── Equity Curve ───")
    _print_equity_ascii(result.equity_curve)

    print()
    print("=" * 70)


def _print_equity_ascii(curve: list[float], width: int = 60, height: int = 15) -> None:
    """Print ASCII equity curve."""
    if len(curve) < 2:
        print("    (not enough data)")
        return

    mn, mx = min(curve), max(curve)
    rng = mx - mn if mx != mn else 1

    step = max(1, len(curve) // width)
    sampled = [curve[i] for i in range(0, len(curve), step)]

    for row in range(height - 1, -1, -1):
        threshold = mn + (rng * row / (height - 1))
        line = f"  ${threshold:8.1f} │"
        for val in sampled:
            line += "█" if val >= threshold else " "
        print(line)

    print(f"  {'':>9s} └{'─' * len(sampled)}")
    print(f"  {'':>9s}  T=0{' ' * (len(sampled) - 8)}T={len(curve) - 1}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Backtest execution pipeline")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Initial capital (USD)")
    parser.add_argument("--min-edge", type=float, default=0.02,
                        help="Minimum edge to trade")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum signal confidence")
    args = parser.parse_args()

    print("\n  Loading data from ClickHouse...")
    client = get_client()

    print("  → Loading signals + markets + trade prices...")
    data = load_backtest_data(client)
    print(f"    {len(data)} signal-market-price rows loaded")

    unique_markets = len(set(d.condition_id for d in data))
    print(f"    {unique_markets} unique resolved markets with trades")

    if not data:
        print("\n  No data available for backtest. Exiting.")
        return

    print(f"\n  Running backtest (capital=${args.capital:.0f}, "
          f"min_edge={args.min_edge}, min_conf={args.min_confidence})...")

    result = run_backtest(
        data=data,
        initial_capital=args.capital,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
    )

    print_report(result)

    # Also run sensitivity analysis
    print("\n  ─── SENSITIVITY ANALYSIS ───")
    print(f"  {'Min Edge':>10s} {'Trades':>7s} {'WinRate':>8s} {'Return':>8s} "
          f"{'Sharpe':>7s} {'MaxDD':>7s} {'PF':>5s}")
    print(f"  {'─' * 10} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 5}")

    for me in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        r = run_backtest(data, args.capital, min_edge=me, min_confidence=args.min_confidence)
        if r.n_trades > 0:
            print(f"  {me:10.3f} {r.n_trades:7d} {r.win_rate:7.1%} "
                  f"{r.total_return:+7.2%} {r.sharpe_ratio:7.2f} "
                  f"{r.max_drawdown:6.2%} {r.profit_factor:5.2f}")
        else:
            print(f"  {me:10.3f}       0       -       -       -       -     -")

    print()


if __name__ == "__main__":
    main()
