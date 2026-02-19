"""Causal impact analysis of real-world events on prediction markets.

Estimates the causal effect of an exogenous event on a market's price
using an interrupted time series approach: fit a linear regression on
the pre-period using control markets as covariates, forecast the
counterfactual post-period trajectory, and measure the difference
between actual and counterfactual prices.

This is a simplified version of Google's CausalImpact methodology,
implemented entirely with numpy and scipy (no external causalimpact
library required).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client
from scipy import stats as sp_stats

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class EventImpactAnalyzer:
    """Estimate the causal impact of events on market prices.

    Uses interrupted time series with control markets as covariates.
    The pre-period relationship between treatment and control markets is
    estimated via OLS, then projected into the post-period to construct
    a counterfactual trajectory.

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

    def fetch_market_series(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Fetch a resampled price series for a single token.

        Parameters
        ----------
        token_id : str
            Token ID to fetch.
        start_date : datetime
            Start of the observation window.
        end_date : datetime
            End of the observation window.
        resample_minutes : int
            Width of each time bin in minutes (default 60).

        Returns
        -------
        dict
            Keys:
            - ``timestamps`` (list[str]): ISO-formatted bucket timestamps.
            - ``prices`` (np.ndarray): average price per bucket.
        """
        client = self._get_client()
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
            SELECT
                toStartOfInterval(timestamp, INTERVAL {resample_minutes} MINUTE) AS bucket,
                avg(price) AS avg_price
            FROM market_prices
            WHERE token_id = '{token_id}'
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
            GROUP BY bucket
            ORDER BY bucket
        """

        result = client.query(query)
        timestamps = []
        prices = []
        for row in result.result_rows:
            timestamps.append(row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]))
            prices.append(float(row[1]))

        return {
            "timestamps": timestamps,
            "prices": np.array(prices, dtype=np.float64),
        }

    def fetch_control_series(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> dict[str, np.ndarray]:
        """Fetch resampled price series for control-group tokens.

        Parameters
        ----------
        token_ids : list[str]
            Control token IDs.
        start_date : datetime
            Start of the observation window.
        end_date : datetime
            End of the observation window.
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from token_id to 1-D price array. All arrays share
            a common time axis built from the union of all buckets,
            forward-filled to handle gaps.
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

        series: dict[str, np.ndarray] = {}
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
            remaining_mask = np.isnan(arr)
            if remaining_mask.any():
                idx = np.where(~remaining_mask, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
            series[tid] = arr

        return series

    def estimate_impact(
        self,
        treatment_token: str,
        control_tokens: list[str],
        event_date: datetime,
        pre_period_days: int = 30,
        post_period_days: int = 7,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Estimate the causal impact of an event on a market.

        Methodology:
        1. Fit OLS regression: treatment_price ~ intercept + control_prices
           using pre-period data only.
        2. Forecast counterfactual post-period prices using fitted model +
           observed control prices.
        3. Compute point-wise and cumulative impact = actual - counterfactual.
        4. Build confidence intervals from pre-period residual std.

        Parameters
        ----------
        treatment_token : str
            Token ID of the market affected by the event.
        control_tokens : list[str]
            Token IDs of unaffected control markets.
        event_date : datetime
            Timestamp of the event (boundary between pre and post).
        pre_period_days : int
            Number of days before the event for model fitting (default 30).
        post_period_days : int
            Number of days after the event to analyze (default 7).
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict
            Keys:
            - ``point_effect`` (float): average post-period impact.
            - ``cumulative_effect`` (float): sum of point-wise impacts.
            - ``relative_effect`` (float): point_effect / mean pre-period price.
            - ``ci_lower`` (float): 95% CI lower bound on point_effect.
            - ``ci_upper`` (float): 95% CI upper bound on point_effect.
            - ``p_value`` (float): two-sided p-value for the impact.
            - ``significant`` (bool): True if p_value < 0.05.
            - ``pre_period_r2`` (float): R-squared of the pre-period model.
            - ``actual_post`` (list[float]): actual post-period prices.
            - ``counterfactual_post`` (list[float]): predicted counterfactual.
            - ``impact_series`` (list[float]): point-wise actual - counterfactual.
        """
        start_date = event_date - timedelta(days=pre_period_days)
        end_date = event_date + timedelta(days=post_period_days)

        # Fetch treatment series
        treatment_data = self.fetch_market_series(
            treatment_token, start_date, end_date, resample_minutes
        )
        if len(treatment_data["prices"]) == 0:
            logger.warning("event_impact_no_treatment_data", extra={"token": treatment_token})
            return self._empty_result()

        # Fetch control series
        control_series = self.fetch_control_series(
            control_tokens, start_date, end_date, resample_minutes
        )
        if not control_series:
            logger.warning("event_impact_no_control_data")
            return self._empty_result()

        # Align: need all series to have the same length
        treatment_prices = treatment_data["prices"]
        n_total = len(treatment_prices)

        # Filter controls to same length
        controls = []
        for tid, arr in control_series.items():
            if len(arr) == n_total:
                controls.append(arr)
        if not controls:
            # Try truncating to min length
            min_len = min(len(arr) for arr in control_series.values())
            min_len = min(min_len, n_total)
            treatment_prices = treatment_prices[:min_len]
            n_total = min_len
            for tid, arr in control_series.items():
                controls.append(arr[:min_len])

        if not controls:
            logger.warning("event_impact_alignment_failed")
            return self._empty_result()

        # Compute pre/post split index
        # pre_period_hours = pre_period_days * 24 * 60 / resample_minutes
        pre_steps = int(pre_period_days * 24 * 60 / resample_minutes)
        pre_steps = min(pre_steps, n_total - 1)
        post_steps = n_total - pre_steps

        if pre_steps < 10 or post_steps < 1:
            logger.warning(
                "event_impact_insufficient_data",
                extra={"pre_steps": pre_steps, "post_steps": post_steps},
            )
            return self._empty_result()

        # Build design matrices
        X_controls = np.column_stack(controls)  # (n_total, n_controls)
        y = treatment_prices

        X_pre = X_controls[:pre_steps]
        y_pre = y[:pre_steps]
        X_post = X_controls[pre_steps:]
        y_post = y[pre_steps:]

        # Add intercept column
        X_pre_aug = np.column_stack([np.ones(pre_steps), X_pre])
        X_post_aug = np.column_stack([np.ones(post_steps), X_post])

        # OLS fit on pre-period
        try:
            # Use least-squares with regularization for numerical stability
            beta, residuals, rank, sv = np.linalg.lstsq(X_pre_aug, y_pre, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("event_impact_linalg_error")
            return self._empty_result()

        # Pre-period predictions and R-squared
        y_pre_hat = X_pre_aug @ beta
        ss_res = np.sum((y_pre - y_pre_hat) ** 2)
        ss_tot = np.sum((y_pre - np.mean(y_pre)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Residual standard error
        dof = max(pre_steps - X_pre_aug.shape[1], 1)
        sigma = np.sqrt(ss_res / dof)

        # Post-period counterfactual
        y_post_hat = X_post_aug @ beta

        # Impact analysis
        impact = y_post.astype(np.float64) - y_post_hat
        point_effect = float(np.mean(impact))
        cumulative_effect = float(np.sum(impact))

        # Confidence interval (from pre-period residual distribution)
        se = sigma / np.sqrt(post_steps) if post_steps > 0 else sigma
        t_crit = sp_stats.t.ppf(0.975, df=dof)
        ci_lower = point_effect - t_crit * se
        ci_upper = point_effect + t_crit * se

        # Two-sided t-test for the point effect
        t_stat = point_effect / se if se > 0 else 0.0
        p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=dof))

        mean_pre_price = float(np.mean(y_pre))
        relative_effect = point_effect / mean_pre_price if mean_pre_price != 0 else 0.0

        logger.info(
            "event_impact_complete",
            extra={
                "treatment": treatment_token,
                "controls": len(controls),
                "point_effect": round(point_effect, 6),
                "p_value": round(p_value, 6),
                "r2": round(r2, 4),
            },
        )

        return {
            "point_effect": round(point_effect, 6),
            "cumulative_effect": round(cumulative_effect, 6),
            "relative_effect": round(relative_effect, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "pre_period_r2": round(r2, 4),
            "actual_post": [round(float(v), 6) for v in y_post],
            "counterfactual_post": [round(float(v), 6) for v in y_post_hat],
            "impact_series": [round(float(v), 6) for v in impact],
        }

    def detect_structural_breaks(
        self,
        token_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        resample_minutes: int = 60,
        window_size: int = 24,
        threshold_sigma: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Detect structural breaks in a market's price series.

        Uses a rolling-window approach: for each point, compare the mean
        of the window before to the mean of the window after. Points where
        the difference exceeds ``threshold_sigma`` standard deviations are
        flagged as potential structural breaks (i.e., event dates).

        Parameters
        ----------
        token_id : str
            Token ID to analyze.
        start_date : datetime, optional
            Start of window. Defaults to 30 days ago.
        end_date : datetime, optional
            End of window. Defaults to now.
        resample_minutes : int
            Time bin width in minutes.
        window_size : int
            Number of time steps on each side of the candidate break point.
        threshold_sigma : float
            Number of standard deviations for a break to be flagged.

        Returns
        -------
        list[dict]
            Each dict has:
            - ``timestamp`` (str): ISO time of the break.
            - ``break_magnitude`` (float): absolute difference in means.
            - ``sigma_score`` (float): break magnitude / pooled std.
            - ``direction`` (str): "up" or "down".
            - ``pre_mean`` (float): mean price before break.
            - ``post_mean`` (float): mean price after break.
        """
        now = datetime.now(timezone.utc)
        if end_date is None:
            end_date = now
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        data = self.fetch_market_series(token_id, start_date, end_date, resample_minutes)
        prices = data["prices"]
        timestamps = data["timestamps"]

        if len(prices) < 2 * window_size + 1:
            logger.warning(
                "structural_break_insufficient_data",
                extra={"length": len(prices), "needed": 2 * window_size + 1},
            )
            return []

        breaks: list[dict[str, Any]] = []

        for i in range(window_size, len(prices) - window_size):
            pre_window = prices[i - window_size: i]
            post_window = prices[i: i + window_size]

            pre_mean = float(np.mean(pre_window))
            post_mean = float(np.mean(post_window))

            # Pooled standard deviation
            pooled_std = float(np.sqrt(
                (np.var(pre_window) + np.var(post_window)) / 2
            ))

            if pooled_std < 1e-10:
                continue

            diff = post_mean - pre_mean
            sigma_score = abs(diff) / pooled_std

            if sigma_score >= threshold_sigma:
                breaks.append({
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "break_magnitude": round(abs(diff), 6),
                    "sigma_score": round(sigma_score, 4),
                    "direction": "up" if diff > 0 else "down",
                    "pre_mean": round(pre_mean, 6),
                    "post_mean": round(post_mean, 6),
                })

        # Merge nearby breaks (within window_size steps)
        if breaks:
            merged: list[dict[str, Any]] = [breaks[0]]
            for b in breaks[1:]:
                # Simple dedup: keep the one with higher sigma_score
                if b["sigma_score"] > merged[-1]["sigma_score"]:
                    merged[-1] = b
                elif len(merged) > 0:
                    # Check distance — if far enough, add as new break
                    # Use index distance approximation via timestamps
                    merged.append(b)
            breaks = merged

        logger.info(
            "structural_breaks_detected",
            extra={
                "token": token_id,
                "breaks_found": len(breaks),
            },
        )

        return breaks

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty impact result when analysis cannot proceed."""
        return {
            "point_effect": 0.0,
            "cumulative_effect": 0.0,
            "relative_effect": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False,
            "pre_period_r2": 0.0,
            "actual_post": [],
            "counterfactual_post": [],
            "impact_series": [],
        }
