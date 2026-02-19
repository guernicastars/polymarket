"""Granger causality analysis between prediction market price series.

Tests whether the past values of one market's price series contain
information useful for predicting another market's future prices, beyond
what the target's own past provides. This is the first step toward causal
discovery: Granger causality is not true causation, but markets that
Granger-cause others are candidates for information leaders.

Uses statsmodels' grangercausalitytests under the hood, querying price
data from ClickHouse's market_prices table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client
from statsmodels.tsa.stattools import grangercausalitytests

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class GrangerCausalityAnalyzer:
    """Pairwise Granger causality testing for market price series.

    Fetches aligned price time series from ClickHouse and computes
    Granger causality F-statistics and p-values for all pairs in a panel
    of markets.

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

    def fetch_price_series(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> dict[str, np.ndarray]:
        """Fetch and resample price series for given tokens from ClickHouse.

        Prices are averaged into fixed-width time bins so that all series
        share a common time axis, which is required for Granger tests.

        Parameters
        ----------
        token_ids : list[str]
            Token IDs to fetch prices for.
        start_date : datetime
            Start of the observation window (inclusive).
        end_date : datetime
            End of the observation window (inclusive).
        resample_minutes : int
            Width of each time bin in minutes (default 60 = hourly).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from token_id to a 1-D array of average prices per bin.
            All arrays have the same length; missing bins are forward-filled.
        """
        client = self._get_client()

        # Use ClickHouse toStartOfInterval for efficient resampling
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

        # Organize into {token_id: {bucket: price}}
        raw: dict[str, dict[datetime, float]] = {}
        for row in result.result_rows:
            tid = str(row[0])
            bucket = row[1]
            price = float(row[2])
            raw.setdefault(tid, {})[bucket] = price

        if not raw:
            logger.warning(
                "granger_no_data",
                extra={"token_count": len(token_ids)},
            )
            return {}

        # Build a common time axis from the union of all buckets
        all_buckets: set[datetime] = set()
        for buckets in raw.values():
            all_buckets.update(buckets.keys())
        time_axis = sorted(all_buckets)

        if len(time_axis) < 3:
            logger.warning(
                "granger_insufficient_buckets",
                extra={"buckets": len(time_axis)},
            )
            return {}

        # Align all series to the common time axis with forward-fill
        series: dict[str, np.ndarray] = {}
        for tid in token_ids:
            if tid not in raw:
                continue
            prices = []
            last_price = None
            for t in time_axis:
                if t in raw[tid]:
                    last_price = raw[tid][t]
                if last_price is not None:
                    prices.append(last_price)
                else:
                    prices.append(np.nan)
            arr = np.array(prices, dtype=np.float64)
            # Drop leading NaNs then forward-fill remaining
            first_valid = np.argmax(~np.isnan(arr))
            arr[:first_valid] = arr[first_valid]
            # Forward fill any remaining NaNs
            mask = np.isnan(arr)
            if mask.any():
                idx = np.where(~mask, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
            series[tid] = arr

        logger.info(
            "granger_series_fetched",
            extra={
                "tokens": len(series),
                "time_steps": len(time_axis),
            },
        )
        return series

    def pairwise_granger(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        max_lag: int = 10,
    ) -> dict[str, Any]:
        """Test whether series_a Granger-causes series_b.

        Parameters
        ----------
        series_a : np.ndarray
            Candidate causal series (1-D).
        series_b : np.ndarray
            Target series (1-D).
        max_lag : int
            Maximum lag to test (default 10).

        Returns
        -------
        dict
            Keys:
            - ``best_lag`` (int): lag with the smallest p-value.
            - ``best_f_stat`` (float): F-statistic at best lag.
            - ``best_p_value`` (float): p-value at best lag.
            - ``all_lags`` (dict): ``{lag: {"f_stat": ..., "p_value": ...}}``.
            - ``significant`` (bool): True if best p-value < 0.05.
        """
        min_length = max_lag + 3  # need enough observations
        if len(series_a) < min_length or len(series_b) < min_length:
            return {
                "best_lag": 0,
                "best_f_stat": 0.0,
                "best_p_value": 1.0,
                "all_lags": {},
                "significant": False,
            }

        # statsmodels wants a 2-column array: [target, predictor]
        data = np.column_stack([series_b, series_a])

        # Clip max_lag to at most (n / 3) to avoid degeneracy
        effective_max_lag = min(max_lag, len(series_a) // 3)
        if effective_max_lag < 1:
            effective_max_lag = 1

        try:
            results = grangercausalitytests(data, maxlag=effective_max_lag, verbose=False)
        except Exception as exc:
            logger.warning("granger_test_error", extra={"error": str(exc)})
            return {
                "best_lag": 0,
                "best_f_stat": 0.0,
                "best_p_value": 1.0,
                "all_lags": {},
                "significant": False,
            }

        all_lags: dict[int, dict[str, float]] = {}
        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag, (tests, _) in results.items():
            # Use the ssr_ftest (standard F-test)
            f_stat = float(tests["ssr_ftest"][0])
            p_value = float(tests["ssr_ftest"][1])
            all_lags[lag] = {"f_stat": round(f_stat, 4), "p_value": round(p_value, 6)}
            if p_value < best_p:
                best_p = p_value
                best_f = f_stat
                best_lag = lag

        return {
            "best_lag": best_lag,
            "best_f_stat": round(best_f, 4),
            "best_p_value": round(best_p, 6),
            "all_lags": all_lags,
            "significant": best_p < 0.05,
        }

    def compute_granger_matrix(
        self,
        token_ids: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_lag: int = 10,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Compute a pairwise Granger causality matrix for a panel of markets.

        Parameters
        ----------
        token_ids : list[str]
            Token IDs to include in the analysis.
        start_date : datetime, optional
            Start of window. Defaults to 7 days ago.
        end_date : datetime, optional
            End of window. Defaults to now.
        max_lag : int
            Maximum lag for Granger tests.
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict
            Keys:
            - ``token_ids`` (list[str]): ordered token IDs (matrix axes).
            - ``f_stats`` (list[list[float]]): NxN matrix of F-statistics.
              ``f_stats[i][j]`` = F-stat for "token i Granger-causes token j".
            - ``p_values`` (list[list[float]]): NxN matrix of p-values.
            - ``best_lags`` (list[list[int]]): NxN matrix of best lags.
            - ``n_significant`` (int): count of significant pairs (p < 0.05).
        """
        now = datetime.now(timezone.utc)
        if end_date is None:
            end_date = now
        if start_date is None:
            from datetime import timedelta
            start_date = end_date - timedelta(days=7)

        series = self.fetch_price_series(
            token_ids, start_date, end_date, resample_minutes
        )

        # Filter to tokens that have data
        available = [t for t in token_ids if t in series]
        n = len(available)

        if n < 2:
            logger.warning(
                "granger_matrix_insufficient",
                extra={"available": n, "requested": len(token_ids)},
            )
            return {
                "token_ids": available,
                "f_stats": [],
                "p_values": [],
                "best_lags": [],
                "n_significant": 0,
            }

        f_stats = [[0.0] * n for _ in range(n)]
        p_values = [[1.0] * n for _ in range(n)]
        best_lags = [[0] * n for _ in range(n)]
        n_significant = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result = self.pairwise_granger(
                    series[available[i]], series[available[j]], max_lag
                )
                f_stats[i][j] = result["best_f_stat"]
                p_values[i][j] = result["best_p_value"]
                best_lags[i][j] = result["best_lag"]
                if result["significant"]:
                    n_significant += 1

        logger.info(
            "granger_matrix_complete",
            extra={
                "markets": n,
                "pairs_tested": n * (n - 1),
                "significant_pairs": n_significant,
            },
        )

        return {
            "token_ids": available,
            "f_stats": f_stats,
            "p_values": p_values,
            "best_lags": best_lags,
            "n_significant": n_significant,
        }

    def find_leading_markets(
        self,
        matrix: dict[str, Any],
        threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Identify markets that Granger-cause many others.

        Parameters
        ----------
        matrix : dict
            Output of ``compute_granger_matrix``.
        threshold : float
            p-value threshold for significance (default 0.05).

        Returns
        -------
        list[dict]
            Sorted by ``caused_count`` descending. Each dict has:
            - ``token_id`` (str)
            - ``caused_count`` (int): number of markets this token
              Granger-causes.
            - ``caused_by_count`` (int): number of markets that
              Granger-cause this token.
            - ``net_influence`` (int): caused_count - caused_by_count.
            - ``targets`` (list[str]): token IDs this market leads.
        """
        token_ids = matrix["token_ids"]
        p_values = matrix["p_values"]
        n = len(token_ids)

        leaders: list[dict[str, Any]] = []

        for i in range(n):
            targets = []
            caused_by: list[str] = []
            for j in range(n):
                if i == j:
                    continue
                if p_values[i][j] < threshold:
                    targets.append(token_ids[j])
                if p_values[j][i] < threshold:
                    caused_by.append(token_ids[j])

            leaders.append({
                "token_id": token_ids[i],
                "caused_count": len(targets),
                "caused_by_count": len(caused_by),
                "net_influence": len(targets) - len(caused_by),
                "targets": targets,
            })

        leaders.sort(key=lambda x: x["caused_count"], reverse=True)
        return leaders
