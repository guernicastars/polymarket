"""Linear probe framework for testing concept separability in embedding space.

Core evaluation mechanism: if a linear classifier can predict a known concept
(category, winning outcome, volatility regime) from the embedding alone,
the concept is linearly separable — evidence that the embedding disentangles it.

Uses cross-validation + permutation tests for statistical rigor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from .config import ProbeConfig

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single linear probe evaluation."""

    concept_name: str
    task_type: str  # "classification" or "regression"
    accuracy: float  # mean CV accuracy (or R^2 for regression)
    accuracy_std: float  # std across CV folds
    baseline_accuracy: float  # majority class (clf) or mean-prediction R^2=0 (reg)
    p_value: float  # from permutation test
    is_significant: bool  # p < alpha
    coefficients: np.ndarray  # (latent_dim,) — probe weight direction
    cv_scores: list[float]  # per-fold scores
    confusion_matrix: Optional[np.ndarray] = None  # for classification


class LinearProbe:
    """Train linear classifiers/regressors on frozen embeddings.

    Tests whether concepts are linearly separable in embedding space.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.cfg = config or ProbeConfig()

    def probe_classification(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        concept_name: str,
    ) -> ProbeResult:
        """Test if a categorical concept is linearly separable.

        Uses LogisticRegression with L2 regularization, StratifiedKFold CV,
        and permutation test for statistical significance.
        """
        # Filter out invalid labels (-1 or NaN)
        valid = labels >= 0
        if valid.sum() < 20:
            logger.warning("Skipping %s: only %d valid samples", concept_name, valid.sum())
            return self._empty_result(concept_name, "classification", embeddings.shape[1])

        X = embeddings[valid]
        y = labels[valid]

        n_classes = len(np.unique(y))
        if n_classes < 2:
            logger.warning("Skipping %s: only %d class", concept_name, n_classes)
            return self._empty_result(concept_name, "classification", embeddings.shape[1])

        # Baseline: majority class
        baseline = np.max(np.bincount(y.astype(int))) / len(y)

        # Cross-validation
        n_folds = min(self.cfg.cv_folds, min(np.bincount(y.astype(int))))
        n_folds = max(n_folds, 2)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        clf = LogisticRegression(max_iter=self.cfg.max_iter, solver="lbfgs")

        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))

        # Fit on full data for coefficients + confusion matrix
        clf.fit(X, y)
        coefs = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_.mean(axis=0)
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)

        # Permutation test
        p_value = self._permutation_test(X, y, mean_acc, "classification")

        return ProbeResult(
            concept_name=concept_name,
            task_type="classification",
            accuracy=mean_acc,
            accuracy_std=std_acc,
            baseline_accuracy=baseline,
            p_value=p_value,
            is_significant=p_value < self.cfg.alpha,
            coefficients=coefs,
            cv_scores=scores.tolist(),
            confusion_matrix=cm,
        )

    def probe_regression(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        concept_name: str,
    ) -> ProbeResult:
        """Test if a continuous concept is linearly predictable.

        Uses Ridge regression with CV, permutation test for significance.
        """
        valid = np.isfinite(targets)
        if valid.sum() < 20:
            logger.warning("Skipping %s: only %d valid samples", concept_name, valid.sum())
            return self._empty_result(concept_name, "regression", embeddings.shape[1])

        X = embeddings[valid]
        y = targets[valid]

        # Check variance
        if np.std(y) < 1e-8:
            logger.warning("Skipping %s: zero variance in targets", concept_name)
            return self._empty_result(concept_name, "regression", embeddings.shape[1])

        n_folds = min(self.cfg.cv_folds, len(X))
        n_folds = max(n_folds, 2)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        reg = Ridge(alpha=1.0)
        scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
        mean_r2 = float(np.mean(scores))
        std_r2 = float(np.std(scores))

        # Fit on full data for coefficients
        reg.fit(X, y)
        coefs = reg.coef_

        # Permutation test
        p_value = self._permutation_test(X, y, mean_r2, "regression")

        return ProbeResult(
            concept_name=concept_name,
            task_type="regression",
            accuracy=mean_r2,
            accuracy_std=std_r2,
            baseline_accuracy=0.0,  # R^2 baseline = predict mean = 0
            p_value=p_value,
            is_significant=p_value < self.cfg.alpha,
            coefficients=coefs,
            cv_scores=scores.tolist(),
        )

    def run_all_probes(
        self,
        embeddings: np.ndarray,
        labels: list[dict],
        feature_names: list[str],
    ) -> list[ProbeResult]:
        """Run all configured probes and return sorted results."""
        logger.info("Running linear probes (%d permutations each)...", self.cfg.n_permutation_tests)
        results = []

        # Extract label arrays
        n = len(labels)

        # --- Classification probes ---

        # 1. Winning outcome (binary: Yes=1, No=0)
        outcome_binary = np.array([lb["outcome_binary"] for lb in labels], dtype=np.int32)
        results.append(self.probe_classification(embeddings, outcome_binary, "winning_outcome"))

        # 2. Category
        categories = [lb["category"] for lb in labels]
        if len(set(categories)) >= 2:
            le = LabelEncoder()
            cat_encoded = le.fit_transform(categories)
            results.append(self.probe_classification(embeddings, cat_encoded, "category"))

        # 3. Duration bucket
        dur_buckets = [lb["duration_bucket"] for lb in labels]
        le_dur = LabelEncoder()
        dur_encoded = le_dur.fit_transform(dur_buckets)
        results.append(self.probe_classification(embeddings, dur_encoded, "duration_bucket"))

        # 4. Volume bucket
        vol_buckets = [lb["volume_bucket"] for lb in labels]
        le_vol = LabelEncoder()
        vol_encoded = le_vol.fit_transform(vol_buckets)
        results.append(self.probe_classification(embeddings, vol_encoded, "volume_bucket"))

        # 5. Volatility regime
        vol_regimes = [lb["volatility_regime"] for lb in labels]
        le_vr = LabelEncoder()
        vr_encoded = le_vr.fit_transform(vol_regimes)
        results.append(self.probe_classification(embeddings, vr_encoded, "volatility_regime"))

        # Sort by significance then accuracy
        results.sort(key=lambda r: (-int(r.is_significant), -r.accuracy))
        return results

    def _permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        true_score: float,
        task_type: str,
    ) -> float:
        """Compute p-value via label permutation.

        Shuffle y N times, refit and score each time.
        p-value = fraction of permuted scores >= true score.
        """
        rng = np.random.RandomState(42)
        n_permutations = self.cfg.n_permutation_tests
        count_ge = 0

        for perm_i in range(n_permutations):
            if perm_i % 50 == 0:
                logger.info("  permutation %d/%d", perm_i, n_permutations)
            y_perm = rng.permutation(y)

            if task_type == "classification":
                n_classes = len(np.unique(y_perm))
                if n_classes < 2:
                    continue
                n_folds = min(self.cfg.cv_folds, min(np.bincount(y_perm.astype(int))))
                n_folds = max(n_folds, 2)
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                clf = LogisticRegression(max_iter=self.cfg.max_iter, solver="lbfgs")
                scores = cross_val_score(clf, X, y_perm, cv=cv, scoring="accuracy")
            else:
                n_folds = min(self.cfg.cv_folds, len(X))
                n_folds = max(n_folds, 2)
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                reg = Ridge(alpha=1.0)
                scores = cross_val_score(reg, X, y_perm, cv=cv, scoring="r2")

            if np.mean(scores) >= true_score:
                count_ge += 1

        return (count_ge + 1) / (n_permutations + 1)

    def run_temporal_probes(
        self,
        embeddings: np.ndarray,
        sequences: list[np.ndarray],
        labels: list[dict],
    ) -> list[ProbeResult]:
        """Run temporal-specific probes that test for time-series dynamics.

        These probe for concepts that only temporal data can capture:
        - trajectory_shape: monotonic / V-recovery / mean-reverting / volatile
        - momentum_profile: accelerating / decelerating / steady
        - volume_pattern: front-loaded / back-loaded / uniform

        Args:
            embeddings: (N, d) frozen embeddings
            sequences: list of (n_bars, 12) arrays (normalized) — one per market
            labels: list of label dicts (for filtering)
        """
        logger.info("Running temporal probes (%d permutations each)...", self.cfg.n_permutation_tests)
        results = []

        n = len(sequences)
        if n < 20:
            logger.warning("Too few sequences for temporal probes: %d", n)
            return results

        # --- 1. Trajectory shape ---
        # Classify price path shape based on log-returns (column 0)
        traj_labels = np.array([self._classify_trajectory(s) for s in sequences])
        n_classes = len(np.unique(traj_labels[traj_labels >= 0]))
        if n_classes >= 2:
            results.append(self.probe_classification(embeddings, traj_labels, "trajectory_shape"))

        # --- 2. Momentum profile ---
        # Classify acceleration pattern: is momentum building or fading?
        momentum_labels = np.array([self._classify_momentum(s) for s in sequences])
        n_classes = len(np.unique(momentum_labels[momentum_labels >= 0]))
        if n_classes >= 2:
            results.append(self.probe_classification(embeddings, momentum_labels, "momentum_profile"))

        # --- 3. Volume pattern ---
        # Classify volume distribution across market lifetime
        vol_labels = np.array([self._classify_volume_pattern(s) for s in sequences])
        n_classes = len(np.unique(vol_labels[vol_labels >= 0]))
        if n_classes >= 2:
            results.append(self.probe_classification(embeddings, vol_labels, "volume_pattern"))

        results.sort(key=lambda r: (-int(r.is_significant), -r.accuracy))
        return results

    @staticmethod
    def _classify_trajectory(seq: np.ndarray) -> int:
        """Classify price path shape from log-returns (column 0).

        0 = monotonic_up: cumulative returns steadily positive
        1 = monotonic_down: cumulative returns steadily negative
        2 = v_recovery: drops then recovers (min in first 60%, ends above midpoint)
        3 = mean_reverting: oscillates around zero (low cumulative, high variance)
        """
        if len(seq) < 8:
            return -1
        log_rets = seq[:, 0]  # already z-scored, but sign matters
        cum_rets = np.cumsum(log_rets)
        final = cum_rets[-1]
        mid = len(cum_rets) // 2
        min_idx = np.argmin(cum_rets)
        max_idx = np.argmax(cum_rets)

        # Mean-reverting: final cumulative return near zero relative to variation
        ret_range = np.max(cum_rets) - np.min(cum_rets)
        if ret_range > 0 and abs(final) / ret_range < 0.2:
            return 3  # mean_reverting

        # V-recovery: minimum in first 60%, ends positive
        if min_idx < len(cum_rets) * 0.6 and final > 0 and cum_rets[min_idx] < -0.3 * ret_range:
            return 2  # v_recovery

        # Monotonic up or down
        if final > 0:
            return 0  # monotonic_up
        return 1  # monotonic_down

    @staticmethod
    def _classify_momentum(seq: np.ndarray) -> int:
        """Classify momentum profile from log-returns.

        0 = accelerating: 2nd-half absolute returns > 1st-half
        1 = decelerating: 1st-half absolute returns > 2nd-half
        2 = steady: roughly equal
        """
        if len(seq) < 8:
            return -1
        log_rets = seq[:, 0]
        mid = len(log_rets) // 2
        first_half_energy = np.mean(np.abs(log_rets[:mid]))
        second_half_energy = np.mean(np.abs(log_rets[mid:]))

        total = first_half_energy + second_half_energy
        if total < 1e-8:
            return 2  # steady (near-zero returns)

        ratio = second_half_energy / max(first_half_energy, 1e-8)
        if ratio > 1.5:
            return 0  # accelerating
        elif ratio < 0.67:
            return 1  # decelerating
        return 2  # steady

    @staticmethod
    def _classify_volume_pattern(seq: np.ndarray) -> int:
        """Classify volume distribution from volume delta (column 6).

        0 = front_loaded: higher volume in first third
        1 = back_loaded: higher volume in last third
        2 = uniform: roughly even distribution
        """
        if len(seq) < 6:
            return -1
        vol_delta = seq[:, 6]  # volume delta feature
        n = len(vol_delta)
        third = n // 3

        first = np.mean(np.abs(vol_delta[:third]))
        last = np.mean(np.abs(vol_delta[-third:]))

        total = first + last
        if total < 1e-8:
            return 2  # uniform

        ratio = first / max(last, 1e-8)
        if ratio > 1.5:
            return 0  # front_loaded
        elif ratio < 0.67:
            return 1  # back_loaded
        return 2  # uniform

    def _empty_result(self, name: str, task_type: str, latent_dim: int) -> ProbeResult:
        return ProbeResult(
            concept_name=name,
            task_type=task_type,
            accuracy=0.0,
            accuracy_std=0.0,
            baseline_accuracy=0.0,
            p_value=1.0,
            is_significant=False,
            coefficients=np.zeros(latent_dim),
            cv_scores=[],
        )
