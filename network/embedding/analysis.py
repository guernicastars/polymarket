"""Disentanglement analysis: PCA, novel directions, orthogonality, clustering.

Answers: is the embedding space well-structured? Do latent dimensions encode
separable concepts? Are there novel factors the model discovered that don't
map to any predefined input feature?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from .config import ProbeConfig
from .probes import ProbeResult

logger = logging.getLogger(__name__)


@dataclass
class NovelDirection:
    """A discovered direction in embedding space with interpretable correlation."""

    direction_idx: int  # PCA component index
    variance_explained: float
    top_correlations: list[tuple[str, float]]  # (feature_name, pearson_r)
    max_correlation: float  # max |r| with any input feature
    description: str  # auto-generated plain language


class DisentanglementAnalyzer:
    """Analyze the structure of the learned embedding space."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.cfg = config or ProbeConfig()

    def pca_analysis(
        self,
        embeddings: np.ndarray,
        feature_names: list[str],
        raw_features: np.ndarray,
    ) -> dict:
        """PCA on embeddings: variance explained, correlation with input features.

        Returns dict with:
          - components: (n_components, latent_dim)
          - explained_variance_ratio: (n_components,)
          - cumulative_variance: (n_components,)
          - component_feature_correlations: (n_components, n_input_features)
        """
        n_components = min(self.cfg.n_pca_components, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(embeddings)

        # Correlate each PCA component with each input feature
        n_feat = raw_features.shape[1]
        correlations = np.zeros((n_components, n_feat))

        for i in range(n_components):
            for j in range(n_feat):
                feat_std = np.std(raw_features[:, j])
                comp_std = np.std(projected[:, i])
                if feat_std > 1e-8 and comp_std > 1e-8:
                    correlations[i, j] = np.corrcoef(projected[:, i], raw_features[:, j])[0, 1]

        return {
            "components": pca.components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "component_feature_correlations": correlations,
            "projected": projected,
            "feature_names": feature_names,
        }

    def search_novel_directions(
        self,
        embeddings: np.ndarray,
        feature_names: list[str],
        raw_features: np.ndarray,
    ) -> list[NovelDirection]:
        """Search for PCA directions that DON'T correlate with known input features.

        Directions with low max-|correlation| to any input feature are novel —
        the model discovered latent structure not present in the raw features.
        """
        pca_result = self.pca_analysis(embeddings, feature_names, raw_features)
        correlations = pca_result["component_feature_correlations"]
        explained = pca_result["explained_variance_ratio"]

        novel_directions = []
        for i in range(correlations.shape[0]):
            abs_corrs = np.abs(correlations[i])
            max_corr = float(np.max(abs_corrs))

            # Novel: max correlation below threshold
            if max_corr < self.cfg.min_correlation_threshold:
                # Build description
                sorted_indices = np.argsort(-abs_corrs)[:3]
                top_corrs = [(feature_names[j], float(correlations[i, j])) for j in sorted_indices]

                desc = (
                    f"Novel direction (PC{i}): max|r|={max_corr:.2f} with any input feature. "
                    f"Weakly correlates with {top_corrs[0][0]} (r={top_corrs[0][1]:.2f})"
                )

                novel_directions.append(NovelDirection(
                    direction_idx=i,
                    variance_explained=float(explained[i]),
                    top_correlations=top_corrs,
                    max_correlation=max_corr,
                    description=desc,
                ))

        # Sort by variance explained (most important novel directions first)
        novel_directions.sort(key=lambda d: -d.variance_explained)
        return novel_directions

    def orthogonality_test(
        self,
        probe_results: list[ProbeResult],
    ) -> np.ndarray:
        """Test orthogonality between probe weight vectors.

        If probes for different concepts have orthogonal weights, the
        concepts are stored in independent directions — strong disentanglement.

        Returns (n_probes, n_probes) cosine similarity matrix.
        """
        vectors = [pr.coefficients for pr in probe_results if pr.coefficients is not None and pr.coefficients.sum() != 0]
        n = len(vectors)
        if n < 2:
            return np.eye(n)

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                norm_i = np.linalg.norm(vectors[i])
                norm_j = np.linalg.norm(vectors[j])
                if norm_i > 0 and norm_j > 0:
                    sim_matrix[i, j] = np.dot(vectors[i], vectors[j]) / (norm_i * norm_j)
                else:
                    sim_matrix[i, j] = 0.0

        return sim_matrix

    def temporal_stability(
        self,
        embeddings: np.ndarray,
        labels: list[dict],
        resolution_dates: list,
    ) -> dict:
        """Test whether probe accuracy is stable across time.

        Split markets by resolution date into early/late halves.
        Train probe on early, test on late (and vice versa).
        """
        from .probes import LinearProbe

        # Sort by resolution date
        n = len(labels)
        mid = n // 2

        indices = list(range(n))
        if resolution_dates:
            indices.sort(key=lambda i: resolution_dates[i] if resolution_dates[i] else 0)

        early_idx = indices[:mid]
        late_idx = indices[mid:]

        # Binary outcome probe: early → late
        outcome_binary = np.array([labels[i]["outcome_binary"] for i in range(n)], dtype=np.int32)

        probe = LinearProbe(self.cfg)
        results = {}

        for name, train_idx, test_idx in [
            ("early_to_late", early_idx, late_idx),
            ("late_to_early", late_idx, early_idx),
        ]:
            train_valid = [i for i in train_idx if outcome_binary[i] >= 0]
            test_valid = [i for i in test_idx if outcome_binary[i] >= 0]

            if len(train_valid) < 10 or len(test_valid) < 10:
                results[name] = {"accuracy": 0.0, "n_train": len(train_valid), "n_test": len(test_valid)}
                continue

            X_train = embeddings[train_valid]
            y_train = outcome_binary[train_valid]
            X_test = embeddings[test_valid]
            y_test = outcome_binary[test_valid]

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=100, solver="lbfgs")
            clf.fit(X_train, y_train)
            acc = float(clf.score(X_test, y_test))

            results[name] = {"accuracy": acc, "n_train": len(train_valid), "n_test": len(test_valid)}

        return results

    def cluster_validation(
        self,
        embeddings: np.ndarray,
        labels: list[dict],
    ) -> dict:
        """Validate that similar markets cluster in embedding space.

        Metrics: silhouette score by category, k-NN accuracy.
        """
        results = {}

        # Silhouette by category
        categories = [lb["category"] for lb in labels]
        if len(set(categories)) >= 2:
            le = LabelEncoder()
            cat_labels = le.fit_transform(categories)
            try:
                sil = float(silhouette_score(embeddings, cat_labels))
                results["silhouette_category"] = sil
            except Exception:
                results["silhouette_category"] = 0.0

        # Silhouette by outcome
        outcomes = np.array([lb["outcome_binary"] for lb in labels])
        valid = outcomes >= 0
        if valid.sum() >= 20:
            try:
                sil = float(silhouette_score(embeddings[valid], outcomes[valid]))
                results["silhouette_outcome"] = sil
            except Exception:
                results["silhouette_outcome"] = 0.0

        # k-NN accuracy for category
        if len(set(categories)) >= 2:
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = cross_val_score(knn, embeddings, cat_labels, cv=3, scoring="accuracy")
            results["knn_category_accuracy"] = float(np.mean(scores))

        return results

    def tsne_coordinates(
        self,
        embeddings: np.ndarray,
        perplexity: float = 30.0,
    ) -> np.ndarray:
        """Compute t-SNE 2D projection for visualization.

        Returns (N, 2) coordinates.
        """
        perplexity = min(perplexity, max(5.0, len(embeddings) / 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(embeddings)
