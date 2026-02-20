"""Statistical validation for the multicollinearity hypothesis.

Core hypothesis: autoencoder embeddings reduce multicollinearity compared
to raw features. We measure this via:

1. **Variance Inflation Factor (VIF)**: For feature j in matrix X,

       VIF_j = 1 / (1 - R_j^2)

   where R_j^2 is the R-squared from regressing feature j on all other features.
   VIF > 5 suggests moderate multicollinearity; VIF > 10 is severe.

2. **Condition Number**: kappa(X) = sigma_max / sigma_min (ratio of largest
   to smallest singular value). kappa > 30 indicates ill-conditioning.

3. **Orthogonality Test**: For embedding matrix Z with columns z_1,...,z_d,
   compute the cosine similarity matrix:

       C_{ij} = (z_i . z_j) / (||z_i|| * ||z_j||)

   Perfectly disentangled embeddings have C = I (identity matrix).
   We report mean and max off-diagonal |C_{ij}|.

4. **Predictive Power (Wald test)**: For each embedding dimension, test
   whether it carries statistically significant predictive information
   for a target variable via logistic regression coefficient z-tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression, LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class VIFResult:
    """Variance Inflation Factor results for a feature matrix.

    Attributes:
        vif_values: VIF for each feature/dimension.
        feature_names: Names of features (or indices).
        mean_vif: Mean VIF across all features.
        max_vif: Maximum VIF.
        n_severe: Count of features with VIF > 10.
        n_moderate: Count of features with VIF > 5.
    """

    vif_values: np.ndarray
    feature_names: list[str]
    mean_vif: float
    max_vif: float
    n_severe: int
    n_moderate: int

    def __str__(self) -> str:
        lines = [
            f"VIF Analysis ({len(self.vif_values)} features):",
            f"  Mean VIF: {self.mean_vif:.2f}",
            f"  Max VIF:  {self.max_vif:.2f}",
            f"  Severe (>10): {self.n_severe}",
            f"  Moderate (>5): {self.n_moderate}",
        ]
        if self.n_severe > 0:
            top_idx = np.argsort(self.vif_values)[-min(5, self.n_severe):][::-1]
            lines.append("  Top offenders:")
            for i in top_idx:
                lines.append(f"    {self.feature_names[i]}: {self.vif_values[i]:.2f}")
        return "\n".join(lines)


@dataclass
class OrthogonalityResult:
    """Results from embedding orthogonality analysis.

    Attributes:
        cosine_sim_matrix: Full pairwise cosine similarity matrix (D x D).
        mean_off_diagonal: Mean |cosine similarity| for off-diagonal entries.
        max_off_diagonal: Max |cosine similarity| for off-diagonal entries.
        n_correlated_pairs: Number of pairs with |cos_sim| > threshold.
    """

    cosine_sim_matrix: np.ndarray
    mean_off_diagonal: float
    max_off_diagonal: float
    n_correlated_pairs: int
    threshold: float = 0.3

    def __str__(self) -> str:
        d = self.cosine_sim_matrix.shape[0]
        total_pairs = d * (d - 1) // 2
        return (
            f"Orthogonality ({d} dimensions):\n"
            f"  Mean |cos_sim| off-diagonal: {self.mean_off_diagonal:.4f}\n"
            f"  Max  |cos_sim| off-diagonal: {self.max_off_diagonal:.4f}\n"
            f"  Correlated pairs (>{self.threshold}): "
            f"{self.n_correlated_pairs}/{total_pairs}"
        )


@dataclass
class PredictivePowerResult:
    """Results from per-dimension predictive power analysis.

    For each embedding dimension, reports the Wald test z-statistic
    and p-value from logistic regression, indicating whether that
    dimension carries statistically significant signal for the target.
    """

    dimension_names: list[str]
    coefficients: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    n_significant: int
    alpha: float = 0.05

    def __str__(self) -> str:
        lines = [
            f"Predictive Power ({len(self.dimension_names)} dims, alpha={self.alpha}):",
            f"  Significant dimensions: {self.n_significant}/{len(self.dimension_names)}",
        ]
        sig_idx = np.where(self.p_values < self.alpha)[0]
        if len(sig_idx) > 0:
            sorted_sig = sig_idx[np.argsort(self.p_values[sig_idx])]
            for i in sorted_sig[:10]:
                lines.append(
                    f"    {self.dimension_names[i]}: "
                    f"coef={self.coefficients[i]:.4f}, "
                    f"z={self.z_scores[i]:.2f}, "
                    f"p={self.p_values[i]:.4e}"
                )
        return "\n".join(lines)


@dataclass
class MulticollinearityComparison:
    """Side-by-side comparison of multicollinearity in raw vs embedding space."""

    raw_vif: VIFResult
    embed_vif: VIFResult
    raw_condition: float
    embed_condition: float
    raw_max_corr: float
    embed_max_corr: float

    def __str__(self) -> str:
        return (
            f"Multicollinearity Comparison:\n"
            f"{'Metric':<30} {'Raw':>10} {'Embedding':>10} {'Reduction':>10}\n"
            f"{'-'*62}\n"
            f"{'Mean VIF':<30} {self.raw_vif.mean_vif:>10.2f} "
            f"{self.embed_vif.mean_vif:>10.2f} "
            f"{(1 - self.embed_vif.mean_vif / max(self.raw_vif.mean_vif, 1e-9)) * 100:>9.1f}%\n"
            f"{'Max VIF':<30} {self.raw_vif.max_vif:>10.2f} "
            f"{self.embed_vif.max_vif:>10.2f} "
            f"{(1 - self.embed_vif.max_vif / max(self.raw_vif.max_vif, 1e-9)) * 100:>9.1f}%\n"
            f"{'Condition Number':<30} {self.raw_condition:>10.1f} "
            f"{self.embed_condition:>10.1f} "
            f"{(1 - self.embed_condition / max(self.raw_condition, 1e-9)) * 100:>9.1f}%\n"
            f"{'Max Pairwise Correlation':<30} {self.raw_max_corr:>10.4f} "
            f"{self.embed_max_corr:>10.4f} "
            f"{(1 - self.embed_max_corr / max(self.raw_max_corr, 1e-9)) * 100:>9.1f}%\n"
            f"{'Severe VIF features (>10)':<30} {self.raw_vif.n_severe:>10d} "
            f"{self.embed_vif.n_severe:>10d}\n"
        )


def compute_vif(
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> VIFResult:
    """Compute Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R_j^2)

    where R_j^2 is the coefficient of determination from regressing
    feature j on all remaining features.

    Args:
        X: Feature matrix of shape (N, D).
        feature_names: Optional names for each feature.

    Returns:
        VIFResult with per-feature VIF values and summary statistics.
    """
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"dim_{i}" for i in range(d)]

    vif_values = np.zeros(d)
    for j in range(d):
        X_other = np.delete(X, j, axis=1)
        y_j = X[:, j]

        reg = LinearRegression()
        reg.fit(X_other, y_j)
        r_squared = reg.score(X_other, y_j)

        vif_values[j] = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else np.inf

    return VIFResult(
        vif_values=vif_values,
        feature_names=feature_names,
        mean_vif=float(np.mean(vif_values[np.isfinite(vif_values)])),
        max_vif=float(np.max(vif_values)),
        n_severe=int(np.sum(vif_values > 10)),
        n_moderate=int(np.sum(vif_values > 5)),
    )


def compute_condition_number(X: np.ndarray) -> float:
    """Compute the condition number of feature matrix X.

    kappa(X) = sigma_max / sigma_min

    where sigma are the singular values of X. A condition number > 30
    indicates substantial multicollinearity.

    Args:
        X: Feature matrix of shape (N, D).

    Returns:
        Condition number (float).
    """
    singular_values = np.linalg.svd(X, compute_uv=False)
    if singular_values[-1] < 1e-10:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix.

    Args:
        X: Feature matrix of shape (N, D).

    Returns:
        Correlation matrix of shape (D, D).
    """
    return np.corrcoef(X.T)


def test_orthogonality(
    embeddings: np.ndarray,
    threshold: float = 0.3,
) -> OrthogonalityResult:
    """Test orthogonality of embedding dimensions via cosine similarity.

    For a perfectly disentangled embedding, each dimension should encode
    an independent factor, yielding near-zero cosine similarity between
    dimension vectors across the dataset.

    Cosine similarity between dimensions i and j:

        C_{ij} = (z_i . z_j) / (||z_i|| ||z_j||)

    where z_i is the i-th column of the embedding matrix (all samples'
    values for dimension i).

    Args:
        embeddings: Embedding matrix of shape (N, D).
        threshold: Cosine similarity threshold for "correlated" pairs.

    Returns:
        OrthogonalityResult with cosine similarity analysis.
    """
    # Normalize each dimension (column) to unit length
    norms = np.linalg.norm(embeddings, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms

    # Cosine similarity matrix between dimensions
    cos_sim = normalized.T @ normalized / embeddings.shape[0]

    # Off-diagonal analysis
    d = cos_sim.shape[0]
    mask = ~np.eye(d, dtype=bool)
    off_diag = np.abs(cos_sim[mask])

    return OrthogonalityResult(
        cosine_sim_matrix=cos_sim,
        mean_off_diagonal=float(np.mean(off_diag)),
        max_off_diagonal=float(np.max(off_diag)) if len(off_diag) > 0 else 0.0,
        n_correlated_pairs=int(np.sum(off_diag > threshold)),
        threshold=threshold,
    )


def test_predictive_power(
    embeddings: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> PredictivePowerResult:
    """Test predictive power of each embedding dimension via Wald test.

    Fits a logistic regression y ~ Z and performs a Wald test on each
    coefficient to determine which embedding dimensions carry significant
    predictive information.

    The Wald statistic for coefficient beta_j:

        W_j = beta_j / SE(beta_j)

    follows approximately N(0,1) under H0: beta_j = 0.

    Args:
        embeddings: Embedding matrix (N, D).
        y: Binary target labels (N,).
        alpha: Significance level for the Wald test.

    Returns:
        PredictivePowerResult with per-dimension test statistics.
    """
    d = embeddings.shape[1]
    dim_names = [f"dim_{i}" for i in range(d)]

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(embeddings, y)
    coefficients = clf.coef_.flatten()

    # Compute standard errors via the Hessian (Fisher information)
    # For logistic regression: Var(beta) = (X^T W X)^{-1}
    # where W = diag(p_i * (1 - p_i))
    probs = clf.predict_proba(embeddings)[:, 1]
    W = probs * (1 - probs)
    X_w = embeddings * np.sqrt(W)[:, np.newaxis]
    fisher_info = X_w.T @ X_w
    try:
        cov_matrix = np.linalg.inv(fisher_info + 1e-8 * np.eye(d))
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_errors = np.ones(d) * np.inf

    z_scores = coefficients / np.maximum(std_errors, 1e-10)
    p_values = 2 * (1 - sp_stats.norm.cdf(np.abs(z_scores)))
    n_significant = int(np.sum(p_values < alpha))

    return PredictivePowerResult(
        dimension_names=dim_names,
        coefficients=coefficients,
        z_scores=z_scores,
        p_values=p_values,
        n_significant=n_significant,
        alpha=alpha,
    )


def compare_multicollinearity(
    X_raw: np.ndarray,
    X_embed: np.ndarray,
    raw_feature_names: list[str] | None = None,
) -> MulticollinearityComparison:
    """Compare multicollinearity between raw features and embeddings.

    This is the main hypothesis test: if the autoencoder successfully
    disentangles the input features, the embedding space should exhibit:

    - Lower mean and max VIF
    - Lower condition number
    - Lower maximum pairwise correlation

    Args:
        X_raw: Raw feature matrix (N, D_raw).
        X_embed: Embedding matrix (N, D_embed).
        raw_feature_names: Optional names for raw features.

    Returns:
        MulticollinearityComparison with side-by-side analysis.
    """
    logger.info("Computing VIF for raw features (%d dims)...", X_raw.shape[1])
    raw_vif = compute_vif(X_raw, raw_feature_names)

    logger.info("Computing VIF for embeddings (%d dims)...", X_embed.shape[1])
    embed_names = [f"emb_{i}" for i in range(X_embed.shape[1])]
    embed_vif = compute_vif(X_embed, embed_names)

    raw_cond = compute_condition_number(X_raw)
    embed_cond = compute_condition_number(X_embed)

    raw_corr = compute_correlation_matrix(X_raw)
    np.fill_diagonal(raw_corr, 0)
    raw_max_corr = float(np.max(np.abs(raw_corr)))

    embed_corr = compute_correlation_matrix(X_embed)
    np.fill_diagonal(embed_corr, 0)
    embed_max_corr = float(np.max(np.abs(embed_corr)))

    comparison = MulticollinearityComparison(
        raw_vif=raw_vif,
        embed_vif=embed_vif,
        raw_condition=raw_cond,
        embed_condition=embed_cond,
        raw_max_corr=raw_max_corr,
        embed_max_corr=embed_max_corr,
    )
    logger.info("\n%s", comparison)
    return comparison
