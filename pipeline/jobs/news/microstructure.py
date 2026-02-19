"""Market microstructure engine — computes tick-level market quality metrics.

Reads from existing ClickHouse tables (orderbook_snapshots, market_trades, market_prices)
and writes enhanced microstructure snapshots for:
  - Spread dynamics (bid-ask, effective, realized)
  - Depth metrics (OBI, depth ratio, spoof detection)
  - Trade flow (volume delta, VWAP, large trade detection)
  - Toxicity (adverse selection, price impact)
  - Kyle's lambda (price impact per dollar traded)
  - Liquidity resilience (spread recovery, depth recovery)

Runs every 60 seconds for top markets, writes to market_microstructure table.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class MicrostructureEngine:
    """Computes market microstructure metrics from ClickHouse data."""

    def __init__(self, ch_client, writer, top_n: int = 100):
        """
        Args:
            ch_client: clickhouse_connect Client (for reads)
            writer: BatchedWriter (for writes)
            top_n: number of top markets to analyze
        """
        self.client = ch_client
        self.writer = writer
        self.top_n = top_n

    async def compute_all(self) -> int:
        """Compute microstructure for top N markets by volume.

        Returns:
            number of snapshots written
        """
        import asyncio

        # Get top markets by 24h volume
        markets = await asyncio.to_thread(self._get_top_markets)
        if not markets:
            logger.warning("No markets found for microstructure analysis")
            return 0

        count = 0
        now = datetime.utcnow()

        for condition_id, token_id in markets:
            try:
                snapshot = await asyncio.to_thread(
                    self._compute_snapshot, condition_id, token_id, now
                )
                if snapshot:
                    self.writer.add("market_microstructure", snapshot)
                    count += 1
            except Exception as e:
                logger.warning("Microstructure failed for %s: %s", condition_id, e)

        logger.info("Microstructure: computed %d/%d market snapshots", count, len(markets))
        return count

    def _get_top_markets(self) -> list[tuple[str, str]]:
        """Get top N markets by volume with their Yes-token IDs."""
        rows = self.client.query(f"""
            SELECT condition_id, token_ids[1] AS token_id
            FROM markets FINAL
            WHERE active = 1 AND closed = 0
              AND length(token_ids) > 0
            ORDER BY volume_24h DESC
            LIMIT {self.top_n}
        """).result_rows
        return [(r[0], r[1]) for r in rows]

    def _compute_snapshot(
        self, condition_id: str, token_id: str, now: datetime
    ) -> Optional[dict]:
        """Compute full microstructure snapshot for one market."""

        # --- Orderbook metrics ---
        ob = self._get_latest_orderbook(condition_id)
        if not ob:
            return None

        bid_prices, bid_sizes, ask_prices, ask_sizes = ob

        # Spread
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        bid_ask_spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0

        # Depth
        bid_depth_1 = bid_sizes[0] if bid_sizes else 0
        ask_depth_1 = ask_sizes[0] if ask_sizes else 0
        bid_depth_5 = sum(bid_sizes[:5]) if len(bid_sizes) >= 5 else sum(bid_sizes)
        ask_depth_5 = sum(ask_sizes[:5]) if len(ask_sizes) >= 5 else sum(ask_sizes)

        # OBI
        total_bid = sum(bid_sizes) if bid_sizes else 0
        total_ask = sum(ask_sizes) if ask_sizes else 0
        obi = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0

        # Depth ratio (spoof detection)
        depth_ratio = bid_depth_5 / bid_depth_1 if bid_depth_1 > 0 else 0

        # --- Trade flow metrics (5-min window) ---
        window_start = now - timedelta(minutes=5)
        trades = self._get_recent_trades(condition_id, window_start, now)

        buy_volume = sum(t[1] for t in trades if t[2] == "buy")
        sell_volume = sum(t[1] for t in trades if t[2] == "sell")
        trade_count = len(trades)
        large_trade_count = sum(1 for t in trades if t[0] * t[1] >= 1000)

        # VWAP
        total_notional = sum(t[0] * t[1] for t in trades)
        total_volume = sum(t[1] for t in trades)
        vwap = total_notional / total_volume if total_volume > 0 else 0

        # --- Kyle's lambda (price impact) ---
        kyle_lambda = self._estimate_kyle_lambda(trades)

        # --- Toxicity metrics ---
        toxic_flow = self._estimate_toxic_flow(trades, now)
        price_impact_1m = self._compute_price_impact(condition_id, now)

        # --- Effective spread ---
        effective_spread = self._compute_effective_spread(trades, best_bid, best_ask)

        # --- Spread recovery ---
        spread_after_trade = bid_ask_spread  # simplified
        depth_recovery_sec = 0.0  # requires time-series orderbook tracking

        return {
            "condition_id": condition_id,
            "bid_ask_spread": bid_ask_spread,
            "effective_spread": effective_spread,
            "realized_spread": max(effective_spread - price_impact_1m, 0),
            "bid_depth_1": bid_depth_1,
            "ask_depth_1": ask_depth_1,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "obi": obi,
            "depth_ratio": depth_ratio,
            "buy_volume_5m": buy_volume,
            "sell_volume_5m": sell_volume,
            "trade_count_5m": trade_count,
            "large_trade_count_5m": large_trade_count,
            "vwap_5m": vwap,
            "kyle_lambda": kyle_lambda,
            "toxic_flow_ratio": toxic_flow,
            "price_impact_1m": price_impact_1m,
            "spread_after_trade": spread_after_trade,
            "depth_recovery_sec": depth_recovery_sec,
            "snapshot_time": now,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_latest_orderbook(self, condition_id: str) -> Optional[tuple]:
        """Get latest orderbook snapshot."""
        rows = self.client.query("""
            SELECT bid_prices, bid_sizes, ask_prices, ask_sizes
            FROM orderbook_snapshots
            WHERE condition_id = {cid:String}
              AND outcome = 'Yes'
            ORDER BY snapshot_time DESC
            LIMIT 1
        """, parameters={"cid": condition_id}).result_rows

        if not rows:
            return None
        return rows[0]

    def _get_recent_trades(
        self, condition_id: str, start: datetime, end: datetime
    ) -> list[tuple]:
        """Get trades in window. Returns [(price, size, side), ...]."""
        rows = self.client.query("""
            SELECT price, size, side
            FROM market_trades
            WHERE condition_id = {cid:String}
              AND timestamp >= {start:DateTime64(3)}
              AND timestamp < {end:DateTime64(3)}
            ORDER BY timestamp
        """, parameters={"cid": condition_id, "start": start, "end": end}).result_rows
        return [(r[0], r[1], r[2]) for r in rows]

    def _estimate_kyle_lambda(self, trades: list[tuple]) -> float:
        """Estimate Kyle's lambda: price impact per dollar of signed order flow.

        lambda = Cov(delta_p, signed_volume) / Var(signed_volume)
        """
        if len(trades) < 5:
            return 0.0

        prices = [t[0] for t in trades]
        signed_vols = []
        for t in trades:
            sign = 1.0 if t[2] == "buy" else -1.0
            signed_vols.append(sign * t[0] * t[1])

        if len(prices) < 2:
            return 0.0

        # Price changes and signed volume
        dp = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        sv = signed_vols[:-1]

        if not dp or not sv:
            return 0.0

        mean_dp = sum(dp) / len(dp)
        mean_sv = sum(sv) / len(sv)

        cov = sum((d - mean_dp) * (s - mean_sv) for d, s in zip(dp, sv)) / len(dp)
        var_sv = sum((s - mean_sv) ** 2 for s in sv) / len(sv)

        return cov / var_sv if var_sv > 1e-10 else 0.0

    def _estimate_toxic_flow(self, trades: list[tuple], now: datetime) -> float:
        """Estimate toxic (informed) vs uninformed order flow ratio.

        Trades that move price significantly in their direction are "toxic".
        """
        if len(trades) < 3:
            return 0.0

        toxic_count = 0
        for i in range(len(trades) - 1):
            price_before = trades[i][0]
            price_after = trades[i + 1][0]
            side = trades[i][2]

            # A "toxic" trade moves price in its direction
            if side == "buy" and price_after > price_before:
                toxic_count += 1
            elif side == "sell" and price_after < price_before:
                toxic_count += 1

        return toxic_count / (len(trades) - 1) if len(trades) > 1 else 0.0

    def _compute_price_impact(self, condition_id: str, now: datetime) -> float:
        """Compute average 1-minute price impact after trades."""
        try:
            rows = self.client.query("""
                SELECT
                    avg(abs(p2.price - t.price)) AS avg_impact
                FROM market_trades t
                ASOF JOIN market_prices p2
                    ON t.condition_id = p2.condition_id
                    AND p2.timestamp >= t.timestamp + INTERVAL 60 SECOND
                WHERE t.condition_id = {cid:String}
                  AND t.timestamp >= now() - INTERVAL 30 MINUTE
                  AND t.size * t.price >= 500
            """, parameters={"cid": condition_id}).result_rows

            return rows[0][0] if rows and rows[0][0] else 0.0
        except Exception:
            return 0.0

    def _compute_effective_spread(
        self, trades: list[tuple], best_bid: float, best_ask: float
    ) -> float:
        """Effective spread: average distance from mid at execution."""
        if not trades or best_bid <= 0 or best_ask <= 0:
            return 0.0

        mid = (best_bid + best_ask) / 2
        spreads = [2 * abs(t[0] - mid) for t in trades]
        return sum(spreads) / len(spreads) if spreads else 0.0
