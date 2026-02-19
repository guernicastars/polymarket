"""Counterfactual analysis for prediction markets.

Answers questions like "What would this market's price be if event X
hadn't occurred?" using two complementary approaches:

1. **Synthetic control method**: constructs a weighted combination of
   donor markets that matches the treatment market's pre-event trajectory,
   then extrapolates into the post-event period as a counterfactual.

2. **Historical analogy**: searches for similar past events and estimates
   the average impact to predict "what if" scenarios.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import clickhouse_connect
import numpy as np
from clickhouse_connect.driver.client import Client
from scipy.optimize import minimize

from pipeline.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)

logger = logging.getLogger(__name__)


class CounterfactualAnalyzer:
    """Counterfactual reasoning for prediction market prices.

    Implements the synthetic control method and historical analogy
    approach for estimating what a market's price trajectory would
    have been under alternative scenarios.

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

    def _fetch_aligned_series(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resample_minutes: int = 60,
    ) -> tuple[list[str], np.ndarray, list[str]]:
        """Fetch aligned price series for multiple tokens.

        Returns
        -------
        tuple
            (available_token_ids, price_matrix, timestamp_labels)
            where price_matrix is (T, N) and timestamps are ISO strings.
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
            return [], np.array([]), []

        # Common time axis
        all_buckets: set[datetime] = set()
        for buckets in raw.values():
            all_buckets.update(buckets.keys())
        time_axis = sorted(all_buckets)
        timestamps = [
            t.isoformat() if hasattr(t, "isoformat") else str(t)
            for t in time_axis
        ]

        available: list[str] = []
        columns: list[np.ndarray] = []

        for tid in token_ids:
            if tid not in raw:
                continue
            prices = []
            last = None
            for t in time_axis:
                if t in raw[tid]:
                    last = raw[tid][t]
                prices.append(last if last is not None else np.nan)
            arr = np.array(prices, dtype=np.float64)

            # Forward fill NaNs
            mask = np.isnan(arr)
            if mask.all():
                continue
            fv = int(np.argmax(~mask))
            arr[:fv] = arr[fv]
            rem = np.isnan(arr)
            if rem.any():
                idx = np.where(~rem, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]

            available.append(tid)
            columns.append(arr)

        if not columns:
            return [], np.array([]), []

        matrix = np.column_stack(columns)  # (T, N)
        return available, matrix, timestamps

    def synthetic_control(
        self,
        treatment_token: str,
        donor_tokens: list[str],
        event_date: datetime,
        pre_period_days: int = 60,
        post_period_days: int = 14,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Estimate counterfactual using the synthetic control method.

        Finds a convex combination of donor markets that best matches
        the treatment market's pre-event trajectory. The same weights
        are applied in the post-event period to construct a counterfactual
        "what would have happened without the event."

        Parameters
        ----------
        treatment_token : str
            Token ID of the treatment market.
        donor_tokens : list[str]
            Token IDs of the donor pool (unaffected markets).
        event_date : datetime
            Timestamp of the event that divides pre and post periods.
        pre_period_days : int
            Number of days before the event for weight fitting.
        post_period_days : int
            Number of days after the event for counterfactual projection.
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict
            Keys:
            - ``weights`` (dict): donor token -> weight in synthetic control.
            - ``pre_fit_rmse`` (float): RMSE of synthetic control in pre-period.
            - ``actual_post`` (list[float]): actual treatment prices post-event.
            - ``counterfactual_post`` (list[float]): synthetic control prices post-event.
            - ``impact_series`` (list[float]): actual - counterfactual per time step.
            - ``average_effect`` (float): mean of impact_series.
            - ``cumulative_effect`` (float): sum of impact_series.
            - ``timestamps_post`` (list[str]): timestamps for post-period.
        """
        start_date = event_date - timedelta(days=pre_period_days)
        end_date = event_date + timedelta(days=post_period_days)

        all_tokens = [treatment_token] + donor_tokens
        available, matrix, timestamps = self._fetch_aligned_series(
            all_tokens, start_date, end_date, resample_minutes
        )

        if treatment_token not in available:
            logger.warning(
                "synthetic_control_no_treatment",
                extra={"token": treatment_token},
            )
            return self._empty_sc_result()

        treat_idx = available.index(treatment_token)
        donor_indices = [i for i, t in enumerate(available) if t != treatment_token]

        if not donor_indices:
            logger.warning("synthetic_control_no_donors")
            return self._empty_sc_result()

        T = matrix.shape[0]
        pre_steps = int(pre_period_days * 24 * 60 / resample_minutes)
        pre_steps = min(pre_steps, T - 1)
        post_start = pre_steps

        if pre_steps < 5:
            logger.warning(
                "synthetic_control_insufficient_pre",
                extra={"pre_steps": pre_steps},
            )
            return self._empty_sc_result()

        # Pre-period data
        y_pre = matrix[:pre_steps, treat_idx]
        X_pre = matrix[:pre_steps][:, donor_indices]

        # Post-period data
        y_post = matrix[post_start:, treat_idx]
        X_post = matrix[post_start:][:, donor_indices]
        timestamps_post = timestamps[post_start:]

        n_donors = len(donor_indices)

        # Optimize weights: minimize ||y_pre - X_pre @ w||^2
        # subject to: w >= 0, sum(w) = 1  (convex combination)
        def objective(w: np.ndarray) -> float:
            synthetic = X_pre @ w
            return float(np.sum((y_pre - synthetic) ** 2))

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0.0, 1.0)] * n_donors
        w0 = np.ones(n_donors) / n_donors  # Equal weights initial guess

        try:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            weights = result.x
        except Exception as exc:
            logger.warning("synthetic_control_optimization_failed", extra={"error": str(exc)})
            weights = w0

        # Compute synthetic control series
        synthetic_pre = X_pre @ weights
        synthetic_post = X_post @ weights

        # Fit quality
        pre_rmse = float(np.sqrt(np.mean((y_pre - synthetic_pre) ** 2)))

        # Impact
        impact = y_post - synthetic_post
        avg_effect = float(np.mean(impact)) if len(impact) > 0 else 0.0
        cum_effect = float(np.sum(impact)) if len(impact) > 0 else 0.0

        # Build weights dict
        donor_available = [available[i] for i in donor_indices]
        weights_dict = {
            tid: round(float(w), 6)
            for tid, w in zip(donor_available, weights)
            if w > 0.001  # Only include non-trivial weights
        }

        logger.info(
            "synthetic_control_complete",
            extra={
                "treatment": treatment_token,
                "donors": len(donor_indices),
                "active_donors": len(weights_dict),
                "pre_rmse": round(pre_rmse, 6),
                "avg_effect": round(avg_effect, 6),
            },
        )

        return {
            "weights": weights_dict,
            "pre_fit_rmse": round(pre_rmse, 6),
            "actual_post": [round(float(v), 6) for v in y_post],
            "counterfactual_post": [round(float(v), 6) for v in synthetic_post],
            "impact_series": [round(float(v), 6) for v in impact],
            "average_effect": round(avg_effect, 6),
            "cumulative_effect": round(cum_effect, 6),
            "timestamps_post": timestamps_post,
        }

    def what_if(
        self,
        token_id: str,
        hypothetical_event: str,
        related_tokens: list[str],
        lookback_days: int = 90,
        resample_minutes: int = 60,
    ) -> dict[str, Any]:
        """Estimate the impact of a hypothetical event using historical analogies.

        Searches the price history of the target market for past structural
        breaks (significant price changes), measures their magnitude and
        direction, and uses the average as an estimate for the hypothetical
        event's impact.

        Parameters
        ----------
        token_id : str
            Token ID of the market to analyze.
        hypothetical_event : str
            Description of the hypothetical event (for labeling).
        related_tokens : list[str]
            Related token IDs to check for correlated past events.
        lookback_days : int
            Number of days to search for historical analogies.
        resample_minutes : int
            Time bin width in minutes.

        Returns
        -------
        dict
            Keys:
            - ``hypothetical_event`` (str): event description.
            - ``estimated_impact`` (float): average impact from analogies.
            - ``impact_std`` (float): standard deviation of historical impacts.
            - ``n_analogies`` (int): number of historical analogies found.
            - ``analogies`` (list[dict]): individual analogy events with details.
            - ``current_price`` (float): latest observed price.
            - ``projected_price`` (float): current_price + estimated_impact.
        """
        from pipeline.causal.event_impact import EventImpactAnalyzer

        impact_analyzer = EventImpactAnalyzer(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
        )

        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=lookback_days)

        # Find structural breaks in the target token's history
        breaks = impact_analyzer.detect_structural_breaks(
            token_id,
            start_date=start_date,
            end_date=now,
            resample_minutes=resample_minutes,
            window_size=24,
            threshold_sigma=2.5,
        )

        # Also check related tokens for correlated events
        related_breaks: list[dict[str, Any]] = []
        for rt in related_tokens[:5]:  # Limit to avoid excessive queries
            rt_breaks = impact_analyzer.detect_structural_breaks(
                rt,
                start_date=start_date,
                end_date=now,
                resample_minutes=resample_minutes,
                window_size=24,
                threshold_sigma=2.5,
            )
            for b in rt_breaks:
                b["source_token"] = rt
                related_breaks.append(b)

        # Collect all analogies
        analogies: list[dict[str, Any]] = []

        for b in breaks:
            analogies.append({
                "source_token": token_id,
                "timestamp": b["timestamp"],
                "magnitude": b["break_magnitude"],
                "direction": b["direction"],
                "sigma_score": b["sigma_score"],
            })

        for b in related_breaks:
            analogies.append({
                "source_token": b.get("source_token", "unknown"),
                "timestamp": b["timestamp"],
                "magnitude": b["break_magnitude"],
                "direction": b["direction"],
                "sigma_score": b["sigma_score"],
            })

        # Estimate impact from analogies
        if analogies:
            impacts = []
            for a in analogies:
                sign = 1.0 if a["direction"] == "up" else -1.0
                impacts.append(sign * a["magnitude"])
            estimated_impact = float(np.mean(impacts))
            impact_std = float(np.std(impacts))
        else:
            estimated_impact = 0.0
            impact_std = 0.0

        # Fetch current price
        client = self._get_client()
        current_result = client.query(f"""
            SELECT price
            FROM market_prices
            WHERE token_id = '{token_id}'
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        current_price = 0.0
        if current_result.result_rows:
            current_price = float(current_result.result_rows[0][0])

        projected_price = current_price + estimated_impact
        # Clamp to valid price range [0, 1] for prediction markets
        projected_price = max(0.0, min(1.0, projected_price))

        logger.info(
            "what_if_analysis",
            extra={
                "token": token_id,
                "event": hypothetical_event,
                "analogies": len(analogies),
                "estimated_impact": round(estimated_impact, 6),
                "current_price": round(current_price, 6),
            },
        )

        return {
            "hypothetical_event": hypothetical_event,
            "estimated_impact": round(estimated_impact, 6),
            "impact_std": round(impact_std, 6),
            "n_analogies": len(analogies),
            "analogies": analogies,
            "current_price": round(current_price, 6),
            "projected_price": round(projected_price, 6),
        }

    @staticmethod
    def _empty_sc_result() -> dict[str, Any]:
        """Return an empty synthetic control result."""
        return {
            "weights": {},
            "pre_fit_rmse": 0.0,
            "actual_post": [],
            "counterfactual_post": [],
            "impact_series": [],
            "average_effect": 0.0,
            "cumulative_effect": 0.0,
            "timestamps_post": [],
        }
