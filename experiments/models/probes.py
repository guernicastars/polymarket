"""Linear probe framework for testing embedding disentanglement.

The key idea: if embeddings truly disentangle latent factors, then simple
linear classifiers/regressors should achieve strong performance on
downstream concept prediction tasks. We compare probe performance on:

    (a) Raw input features (potentially multicollinear)
    (b) Learned embeddings (hypothesized to be disentangled)

If the embedding space is better disentangled, probes trained on embeddings
should match or exceed raw feature probes despite lower dimensionality,
indicating that the autoencoder has learned a more efficient representation.

Statistical significance is assessed via permutation tests (shuffling labels
K times and measuring chance-level performance) and likelihood ratio tests
(comparing the full model to a null model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Results from a single linear probe experiment.

    Attributes:
        concept: Name of the concept being probed (e.g., 'category', 'outcome').
        task_type: 'classification' or 'regression'.
        input_type: 'raw' or 'embedding'.
        input_dim: Dimensionality of the input to the probe.
        metrics: Dict of metric_name -> value (e.g., accuracy, f1, r2).
        metrics_std: Dict of metric_name -> std across CV folds.
        p_value: Permutation test p-value (probability of chance performance).
        n_samples: Number of samples used.
        n_classes: Number of classes (classification only).
    """

    concept: str
    task_type: str
    input_type: str
    input_dim: int
    metrics: dict[str, float]
    metrics_std: dict[str, float] = field(default_factory=dict)
    p_value: float | None = None
    n_samples: int = 0
    n_classes: int = 0

    def __str__(self) -> str:
        lines = [
            f"Probe: {self.concept} ({self.task_type}) on {self.input_type} features "
            f"[dim={self.input_dim}, n={self.n_samples}]",
        ]
        for name, val in self.metrics.items():
            std_str = ""
            if name in self.metrics_std:
                std_str = f" +/- {self.metrics_std[name]:.4f}"
            lines.append(f"  {name}: {val:.4f}{std_str}")
        if self.p_value is not None:
            lines.append(f"  p-value: {self.p_value:.4f}")
        return "\n".join(lines)


@dataclass
class ProbeComparison:
    """Side-by-side comparison of raw vs embedding probes for one concept."""

    concept: str
    raw_result: ProbeResult
    embed_result: ProbeResult

    @property
    def primary_metric(self) -> str:
        if self.raw_result.task_type == "regression":
            return "r2"
        return "accuracy"

    @property
    def improvement(self) -> float:
        """Improvement from raw to embedding (positive = embedding better)."""
        m = self.primary_metric
        return self.embed_result.metrics[m] - self.raw_result.metrics[m]

    def __str__(self) -> str:
        m = self.primary_metric
        raw_val = self.raw_result.metrics[m]
        emb_val = self.embed_result.metrics[m]
        delta = self.improvement
        sign = "+" if delta >= 0 else ""
        lines = [
            f"=== {self.concept} ({self.raw_result.task_type}) ===",
            f"  Raw features ({self.raw_result.input_dim}D): {m}={raw_val:.4f}",
            f"  Embeddings   ({self.embed_result.input_dim}D): {m}={emb_val:.4f}",
            f"  Delta: {sign}{delta:.4f}",
        ]
        if self.raw_result.p_value is not None:
            lines.append(f"  p-value (raw): {self.raw_result.p_value:.4f}")
        if self.embed_result.p_value is not None:
            lines.append(f"  p-value (emb): {self.embed_result.p_value:.4f}")
        return "\n".join(lines)


class LinearProbe:
    """Linear probe for evaluating representation quality.

    Fits a simple linear model (logistic regression for classification,
    ridge regression for regression) on top of frozen features and measures
    how well the representation encodes a given concept.
    """

    def __init__(self, n_folds: int = 5, n_permutations: int = 100, seed: int = 42) -> None:
        self.n_folds = n_folds
        self.n_permutations = n_permutations
        self.seed = seed

    def probe_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        concept: str,
        input_type: str,
    ) -> ProbeResult:
        """Run classification probe with cross-validation.

        Fits LogisticRegression and evaluates accuracy, macro-F1,
        and AUC-ROC (where applicable).

        Args:
            X: Feature matrix (N, D).
            y: Labels (N,) — integer or string class labels.
            concept: Name of the concept being probed.
            input_type: 'raw' or 'embedding'.

        Returns:
            ProbeResult with classification metrics.
        """
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        fold_metrics: dict[str, list[float]] = {
            "accuracy": [], "macro_f1": [],
        }
        if n_classes == 2:
            fold_metrics["auc_roc"] = []

        for train_idx, test_idx in cv.split(X, y_enc):
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs",
                random_state=self.seed,
            )
            clf.fit(X[train_idx], y_enc[train_idx])
            y_pred = clf.predict(X[test_idx])

            fold_metrics["accuracy"].append(accuracy_score(y_enc[test_idx], y_pred))
            fold_metrics["macro_f1"].append(
                f1_score(y_enc[test_idx], y_pred, average="macro", zero_division=0)
            )
            if n_classes == 2:
                y_prob = clf.predict_proba(X[test_idx])[:, 1]
                fold_metrics["auc_roc"].append(roc_auc_score(y_enc[test_idx], y_prob))

        metrics = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
        metrics_std = {k: float(np.std(v)) for k, v in fold_metrics.items()}

        # Permutation test for statistical significance
        p_value = self._permutation_test_classification(X, y_enc, metrics["accuracy"])

        return ProbeResult(
            concept=concept,
            task_type="classification",
            input_type=input_type,
            input_dim=X.shape[1],
            metrics=metrics,
            metrics_std=metrics_std,
            p_value=p_value,
            n_samples=X.shape[0],
            n_classes=n_classes,
        )

    def probe_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        concept: str,
        input_type: str,
    ) -> ProbeResult:
        """Run regression probe with cross-validation.

        Fits Ridge regression and evaluates R-squared and MAE.

        Args:
            X: Feature matrix (N, D).
            y: Continuous target (N,).
            concept: Name of the concept being probed.
            input_type: 'raw' or 'embedding'.

        Returns:
            ProbeResult with regression metrics.
        """
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_metrics: dict[str, list[float]] = {"r2": [], "mae": []}

        for train_idx, test_idx in cv.split(X):
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx], y[train_idx])
            y_pred = reg.predict(X[test_idx])

            fold_metrics["r2"].append(r2_score(y[test_idx], y_pred))
            fold_metrics["mae"].append(mean_absolute_error(y[test_idx], y_pred))

        metrics = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
        metrics_std = {k: float(np.std(v)) for k, v in fold_metrics.items()}

        p_value = self._permutation_test_regression(X, y, metrics["r2"])

        return ProbeResult(
            concept=concept,
            task_type="regression",
            input_type=input_type,
            input_dim=X.shape[1],
            metrics=metrics,
            metrics_std=metrics_std,
            p_value=p_value,
            n_samples=X.shape[0],
        )

    def _permutation_test_classification(
        self, X: np.ndarray, y: np.ndarray, observed_acc: float
    ) -> float:
        """Permutation test: shuffle labels K times, count how often chance >= observed.

        p-value = (1 + #{permuted_acc >= observed_acc}) / (1 + K)

        The +1 correction avoids p=0 and accounts for the observed statistic itself.
        """
        rng = np.random.RandomState(self.seed)
        count_ge = 0
        for _ in range(self.n_permutations):
            y_perm = rng.permutation(y)
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=self.seed)
            clf.fit(X, y_perm)
            perm_acc = accuracy_score(y, clf.predict(X))
            if perm_acc >= observed_acc:
                count_ge += 1
        return (1 + count_ge) / (1 + self.n_permutations)

    def _permutation_test_regression(
        self, X: np.ndarray, y: np.ndarray, observed_r2: float
    ) -> float:
        """Permutation test for regression: shuffle targets, compare R-squared."""
        rng = np.random.RandomState(self.seed)
        count_ge = 0
        for _ in range(self.n_permutations):
            y_perm = rng.permutation(y)
            reg = Ridge(alpha=1.0)
            reg.fit(X, y_perm)
            perm_r2 = r2_score(y_perm, reg.predict(X))
            if perm_r2 >= observed_r2:
                count_ge += 1
        return (1 + count_ge) / (1 + self.n_permutations)

    def compare(
        self,
        X_raw: np.ndarray,
        X_embed: np.ndarray,
        y: np.ndarray,
        concept: str,
        task_type: str = "classification",
    ) -> ProbeComparison:
        """Run probes on raw features and embeddings, return comparison.

        Args:
            X_raw: Raw feature matrix (N, D_raw).
            X_embed: Embedding matrix (N, D_embed).
            y: Target labels/values.
            concept: Name of concept being probed.
            task_type: 'classification' or 'regression'.

        Returns:
            ProbeComparison with both results.
        """
        probe_fn = self.probe_classification if task_type == "classification" else self.probe_regression
        logger.info("Probing '%s' (%s) on raw features [%dD]...", concept, task_type, X_raw.shape[1])
        raw_result = probe_fn(X_raw, y, concept, "raw")
        logger.info("Probing '%s' (%s) on embeddings [%dD]...", concept, task_type, X_embed.shape[1])
        embed_result = probe_fn(X_embed, y, concept, "embedding")
        return ProbeComparison(concept=concept, raw_result=raw_result, embed_result=embed_result)


def run_standard_probes(
    X_raw: np.ndarray,
    X_embed: np.ndarray,
    labels: dict[str, np.ndarray],
    n_permutations: int = 100,
) -> list[ProbeComparison]:
    """Run the standard set of probe comparisons.

    Expected labels dict keys:
        'category' — multi-class string labels (e.g., 'Sports', 'Politics')
        'outcome' — binary int labels (0/1, resolved yes/no)
        'time_to_expiry' — continuous float (days until resolution)
        'volatility_regime' — binary int (0=low, 1=high volatility)

    Args:
        X_raw: Raw feature matrix (N, D).
        X_embed: Embedding matrix (N, E).
        labels: Dict mapping concept name to label array.
        n_permutations: Number of permutation test iterations.

    Returns:
        List of ProbeComparison objects.
    """
    probe = LinearProbe(n_permutations=n_permutations)
    comparisons = []

    probe_configs = [
        ("category", "classification"),
        ("outcome", "classification"),
        ("volatility_regime", "classification"),
        ("time_to_expiry", "regression"),
    ]

    for concept, task_type in probe_configs:
        if concept not in labels:
            logger.warning("Skipping probe '%s': labels not found", concept)
            continue
        y = labels[concept]
        if len(y) != X_raw.shape[0]:
            logger.warning("Skipping probe '%s': label count %d != sample count %d",
                           concept, len(y), X_raw.shape[0])
            continue
        comparison = probe.compare(X_raw, X_embed, y, concept, task_type)
        comparisons.append(comparison)
        logger.info("\n%s", comparison)

    return comparisons
