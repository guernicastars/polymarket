"""Transfer entropy and information flow analysis between markets.

Transfer entropy measures the directed flow of information from a source
to a target time series. Unlike Granger causality (which assumes linear
relationships), transfer entropy captures nonlinear dependencies.

A market with high outgoing transfer entropy is an "information source" --
it moves first and other markets follow. A market with high incoming
transfer entropy is a "derivative" market that mostly absorbs information
from elsewhere.

Uses a histogram-based estimator for computational efficiency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class InformationFlowAnalyzer:
    """Measure directed information flow between prediction markets.

    Computes transfer entropy between market price series using a
    histogram-based estimator. Identifies source and derivative markets
    in the information topology.

    Parameters
    ----------
    host : str, optional
        ClickHouse host. Defaults to config value.
    port : int, optional
        ClickHouse port. Defaults to config value.
    user : str, optional
        ClickHouse user. Defaults to config value.
    password : str, optional
        ClickHouse password. Defaults to config value.
    database : str, optional
        ClickHouse database. Defaults to config value.
    """

    def __init__(
        self,
        host: str = CLICKHOUSE_HOST,
        port: int = CLICKHOUSE_PORT,
        user: str = CLICKHOUSE_USER,
        password: str = CLICKHOUSE_PASSWORD,
        database: str = CLICKHOUSE_DATABASE,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._client: Client | None = None

    def _get_client(self) -> Client:
        """Lazily create and return a ClickHouse client."""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self._host,
                port=self._port,
                username=self._user,
                password=self._password,
                database=self._database,
                secure=True,
            )
        return self._client

    def _fetch_price_returns(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> dict[str, np.ndarray]:
        """Fetch price returns (log differences) for a set of tokens.

        Returns log-returns rather than raw prices, because transfer
        entropy is better estimated on stationary series.

        Parameters
        ----------
        token_ids : list[str]
            Token IDs to fetch.
        start_date : datetime
            Start of the observation window.
        end_date : datetime
            End of the observation window.
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from token_id to 1-D array of log-returns.
        """
        client = self._get_client()
        token_list = ", ".join(f"'{t}'" for t in token_ids)
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
            SELECT
                token_id,
                toStartOfInterval(timestamp, INTERVAL {resample_minutes} MINUTE) AS bucket,
                avg(price) AS avg_price
            FROM market_prices
            WHERE token_id IN ({token_list})
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
            GROUP BY token_id, bucket
            ORDER BY token_id, bucket
        """

        result = client.query(query)

        raw: dict[str, dict[datetime, float]] = {}
        for row in result.result_rows:
            tid = str(row[0])
            bucket = row[1]
            price = float(row[2])
            raw.setdefault(tid, {})[bucket] = price

        if not raw:
            return {}

        # Common time axis
        all_buckets: set[datetime] = set()
        for buckets in raw.values():
            all_buckets.update(buckets.keys())
        time_axis = sorted(all_buckets)

        returns: dict[str, np.ndarray] = {}
        for tid in token_ids:
            if tid not in raw:
                continue
            prices = []
            last_price = None
            for t in time_axis:
                if t in raw[tid]:
                    last_price = raw[tid][t]
                prices.append(last_price if last_price is not None else np.nan)
            arr = np.array(prices, dtype=np.float64)

            # Forward fill NaNs
            mask = np.isnan(arr)
            if mask.all():
                continue
            first_valid = int(np.argmax(~mask))
            arr[:first_valid] = arr[first_valid]
            remaining = np.isnan(arr)
            if remaining.any():
                idx = np.where(~remaining, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]

            # Log-returns (handle zeros by clamping)
            arr_safe = np.maximum(arr, 1e-10)
            log_returns = np.diff(np.log(arr_safe))
            if len(log_returns) > 0:
                returns[tid] = log_returns

        return returns

    @staticmethod
    def compute_transfer_entropy(
        source_series: np.ndarray,
        target_series: np.ndarray,
        lag: int = 1,
        bins: int = 10,
    ) -> float:
        """Estimate transfer entropy from source to target.

        Transfer entropy T(X -> Y) measures the reduction in uncertainty
        about Y's future given X's past, beyond what Y's own past provides:

            T(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

        Uses a histogram-based estimator: discretize both series into bins,
        then estimate conditional entropies from joint frequency tables.

        Parameters
        ----------
        source_series : np.ndarray
            Candidate information source (1-D).
        target_series : np.ndarray
            Information target (1-D).
        lag : int
            Lag for the source series (default 1).
        bins : int
            Number of histogram bins for discretization (default 10).

        Returns
        -------
        float
            Estimated transfer entropy in nats. Non-negative; higher values
            indicate stronger directional information flow.
        """
        n = min(len(source_series), len(target_series))
        if n < lag + 2:
            return 0.0

        # Align series: target_future, target_past, source_past
        target_future = target_series[lag:n]
        target_past = target_series[:n - lag]
        source_past = source_series[:n - lag]

        # Discretize into equal-width bins
        def _digitize(arr: np.ndarray, n_bins: int) -> np.ndarray:
            """Bin array into n_bins equal-width intervals."""
            mn, mx = np.min(arr), np.max(arr)
            if mx - mn < 1e-12:
                return np.zeros(len(arr), dtype=int)
            edges = np.linspace(mn, mx, n_bins + 1)
            # np.digitize returns 1..n_bins; shift to 0..n_bins-1
            d = np.digitize(arr, edges[1:-1])
            return d

        tf_d = _digitize(target_future, bins)
        tp_d = _digitize(target_past, bins)
        sp_d = _digitize(source_past, bins)

        # Joint counts -> probabilities
        # P(target_future, target_past, source_past)
        n_obs = len(tf_d)

        # Build 3D histogram manually for efficiency
        joint_3d: dict[tuple[int, int, int], int] = {}
        for i in range(n_obs):
            key = (tf_d[i], tp_d[i], sp_d[i])
            joint_3d[key] = joint_3d.get(key, 0) + 1

        # Marginals needed:
        # P(target_future, target_past)
        joint_tf_tp: dict[tuple[int, int], int] = {}
        # P(target_past, source_past)
        joint_tp_sp: dict[tuple[int, int], int] = {}
        # P(target_past)
        marginal_tp: dict[int, int] = {}

        for (tf, tp, sp), count in joint_3d.items():
            key_tf_tp = (tf, tp)
            joint_tf_tp[key_tf_tp] = joint_tf_tp.get(key_tf_tp, 0) + count
            key_tp_sp = (tp, sp)
            joint_tp_sp[key_tp_sp] = joint_tp_sp.get(key_tp_sp, 0) + count
            marginal_tp[tp] = marginal_tp.get(tp, 0) + count

        # Transfer entropy:
        # T(X -> Y) = sum P(y_t, y_{t-1}, x_{t-lag}) *
        #             log[ P(y_t | y_{t-1}, x_{t-lag}) / P(y_t | y_{t-1}) ]
        #
        # = sum P(y_t, y_{t-1}, x_{t-lag}) *
        #   log[ P(y_t, y_{t-1}, x_{t-lag}) * P(y_{t-1}) /
        #        (P(y_{t-1}, x_{t-lag}) * P(y_t, y_{t-1})) ]

        te = 0.0
        for (tf, tp, sp), count_3 in joint_3d.items():
            p_3 = count_3 / n_obs
            p_tf_tp = joint_tf_tp.get((tf, tp), 0) / n_obs
            p_tp_sp = joint_tp_sp.get((tp, sp), 0) / n_obs
            p_tp = marginal_tp.get(tp, 0) / n_obs

            if p_tf_tp > 0 and p_tp_sp > 0 and p_tp > 0:
                ratio = (p_3 * p_tp) / (p_tp_sp * p_tf_tp)
                if ratio > 0:
                    te += p_3 * np.log(ratio)

        # Transfer entropy should be non-negative; small negative values
        # are numerical artifacts
        return max(0.0, float(te))

    def compute_flow_matrix(
        self,
        token_ids: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        lag: int = 1,
        bins: int = 10,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Compute a pairwise transfer entropy matrix.

        Parameters
        ----------
        token_ids : list[str]
            Token IDs to analyze.
        start_date : datetime, optional
            Start of window. Defaults to 7 days ago.
        end_date : datetime, optional
            End of window. Defaults to now.
        lag : int
            Lag for transfer entropy estimation.
        bins : int
            Number of histogram bins.
        resample_minutes : int
            Time bin width for price resampling.

        Returns
        -------
        dict
            Keys:
            - ``token_ids`` (list[str]): ordered tokens (matrix axes).
            - ``te_matrix`` (list[list[float]]): NxN transfer entropy matrix.
              ``te_matrix[i][j]`` = TE from token i to token j.
            - ``max_te`` (float): maximum transfer entropy observed.
        """
        now = datetime.now(timezone.utc)
        if end_date is None:
            end_date = now
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        returns = self._fetch_price_returns(
            token_ids, start_date, end_date, resample_minutes
        )

        available = [t for t in token_ids if t in returns]
        n = len(available)

        if n < 2:
            logger.warning(
                "flow_matrix_insufficient",
                extra={"available": n},
            )
            return {
                "token_ids": available,
                "te_matrix": [],
                "max_te": 0.0,
            }

        te_matrix = [[0.0] * n for _ in range(n)]
        max_te = 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                te = self.compute_transfer_entropy(
                    returns[available[i]], returns[available[j]], lag, bins
                )
                te_matrix[i][j] = round(te, 6)
                if te > max_te:
                    max_te = te

        logger.info(
            "flow_matrix_complete",
            extra={
                "markets": n,
                "max_te": round(max_te, 6),
            },
        )

        return {
            "token_ids": available,
            "te_matrix": te_matrix,
            "max_te": round(max_te, 6),
        }

    def identify_source_markets(
        self,
        flow_matrix: dict[str, Any],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Identify markets with the highest outgoing information flow.

        Parameters
        ----------
        flow_matrix : dict
            Output of ``compute_flow_matrix``.
        top_n : int
            Number of top sources to return.

        Returns
        -------
        list[dict]
            Sorted by ``total_outflow`` descending. Each dict has:
            - ``token_id`` (str)
            - ``total_outflow`` (float): sum of outgoing TE.
            - ``avg_outflow`` (float): average outgoing TE.
            - ``max_outflow`` (float): maximum outgoing TE to any single target.
        """
        token_ids = flow_matrix["token_ids"]
        te_matrix = flow_matrix["te_matrix"]
        n = len(token_ids)

        sources: list[dict[str, Any]] = []
        for i in range(n):
            outflows = [te_matrix[i][j] for j in range(n) if j != i]
            if not outflows:
                continue
            sources.append({
                "token_id": token_ids[i],
                "total_outflow": round(sum(outflows), 6),
                "avg_outflow": round(float(np.mean(outflows)), 6),
                "max_outflow": round(max(outflows), 6),
            })

        sources.sort(key=lambda x: x["total_outflow"], reverse=True)
        return sources[:top_n]

    def identify_derivative_markets(
        self,
        flow_matrix: dict[str, Any],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Identify markets with the highest incoming information flow.

        Parameters
        ----------
        flow_matrix : dict
            Output of ``compute_flow_matrix``.
        top_n : int
            Number of top derivative markets to return.

        Returns
        -------
        list[dict]
            Sorted by ``total_inflow`` descending. Each dict has:
            - ``token_id`` (str)
            - ``total_inflow`` (float): sum of incoming TE.
            - ``avg_inflow`` (float): average incoming TE.
            - ``max_inflow`` (float): maximum incoming TE from any single source.
        """
        token_ids = flow_matrix["token_ids"]
        te_matrix = flow_matrix["te_matrix"]
        n = len(token_ids)

        derivatives: list[dict[str, Any]] = []
        for j in range(n):
            inflows = [te_matrix[i][j] for i in range(n) if i != j]
            if not inflows:
                continue
            derivatives.append({
                "token_id": token_ids[j],
                "total_inflow": round(sum(inflows), 6),
                "avg_inflow": round(float(np.mean(inflows)), 6),
                "max_inflow": round(max(inflows), 6),
            })

        derivatives.sort(key=lambda x: x["total_inflow"], reverse=True)
        return derivatives[:top_n]

    def net_information_flow(
        self,
        flow_matrix: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Compute net information flow per market (outflow - inflow).

        Positive net flow means the market is a net information producer.
        Negative net flow means it is a net information consumer.

        Parameters
        ----------
        flow_matrix : dict
            Output of ``compute_flow_matrix``.

        Returns
        -------
        list[dict]
            Sorted by ``net_flow`` descending. Each dict has:
            - ``token_id`` (str)
            - ``total_outflow`` (float)
            - ``total_inflow`` (float)
            - ``net_flow`` (float): outflow - inflow.
            - ``role`` (str): "source" if net_flow > 0, "derivative" otherwise.
        """
        token_ids = flow_matrix["token_ids"]
        te_matrix = flow_matrix["te_matrix"]
        n = len(token_ids)

        flows: list[dict[str, Any]] = []
        for i in range(n):
            outflow = sum(te_matrix[i][j] for j in range(n) if j != i)
            inflow = sum(te_matrix[j][i] for j in range(n) if j != i)
            net = outflow - inflow
            flows.append({
                "token_id": token_ids[i],
                "total_outflow": round(outflow, 6),
                "total_inflow": round(inflow, 6),
                "net_flow": round(net, 6),
                "role": "source" if net > 0 else "derivative",
            })

        flows.sort(key=lambda x: x["net_flow"], reverse=True)
        return flows
