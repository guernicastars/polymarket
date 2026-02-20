"""Feature interpretation: trace embedding activations back to input features.

For each significant probe direction, computes:
  1. Correlation-based attribution (fast, robust)
  2. Jacobian-based attribution (precise, requires autograd)
  3. Plain-language description of what the direction means
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .probes import ProbeResult

logger = logging.getLogger(__name__)


@dataclass
class InterpretedDirection:
    """A fully interpreted latent direction."""

    name: str
    description: str  # plain-language
    direction_vector: np.ndarray  # (latent_dim,)
    input_attributions: dict[str, float]  # feature_name -> attribution score
    top_positive_markets: list[str]  # condition_ids scoring highest along direction
    top_negative_markets: list[str]  # condition_ids scoring lowest
    probe_accuracy: Optional[float] = None


class FeatureInterpreter:
    """Trace embedding activations back to input features.

    Two methods:
      - Correlation-based (default): fast, interpretable, no autograd
      - Jacobian-based: precise but slower, uses autograd
    """

    def __init__(self, model: nn.Module, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names

    def attribute_direction_correlation(
        self,
        direction: np.ndarray,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
    ) -> dict[str, float]:
        """Attribute a latent direction to input features via correlation.

        Method: compute dot product of each embedding with the direction vector,
        then correlate this scalar projection with each input feature.
        """
        # Project embeddings onto direction
        projections = embeddings @ direction  # (N,)

        attributions = {}
        for j, fname in enumerate(self.feature_names):
            feat = raw_features[:, j]
            if np.std(feat) < 1e-8 or np.std(projections) < 1e-8:
                attributions[fname] = 0.0
            else:
                r = np.corrcoef(projections, feat)[0, 1]
                attributions[fname] = float(r)

        return attributions

    def attribute_direction_jacobian(
        self,
        direction: np.ndarray,
        normalized_features: np.ndarray,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Attribute a latent direction to input features via Jacobian.

        Computes d(encoder(x))/dx for each sample, projects each row onto
        the direction vector, and averages across all samples.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

        n = len(normalized_features)
        all_attr = np.zeros(len(self.feature_names), dtype=np.float64)
        count = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.tensor(
                normalized_features[start:end], dtype=torch.float32, device=device
            ).requires_grad_(True)

            # Forward through encoder
            if hasattr(self.model, "encode"):
                # Standard AE
                z = self.model.encode(x)
            else:
                continue

            # Project onto direction: scalar = z @ direction
            proj = z @ direction_t  # (batch,)
            proj_sum = proj.sum()

            # Backprop to get gradient w.r.t. input
            proj_sum.backward()
            if x.grad is not None:
                # Average gradient across batch gives per-feature attribution
                grad = x.grad.detach().cpu().numpy()  # (batch, input_dim)
                all_attr += grad.mean(axis=0) * (end - start)
                count += end - start

        if count > 0:
            all_attr /= count

        return {fname: float(all_attr[j]) for j, fname in enumerate(self.feature_names)}

    def interpret_probe(
        self,
        probe_result: ProbeResult,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        condition_ids: list[str],
        use_jacobian: bool = False,
        normalized_features: Optional[np.ndarray] = None,
    ) -> InterpretedDirection:
        """Full interpretation of a single probe direction."""
        direction = probe_result.coefficients

        # Attribution
        if use_jacobian and normalized_features is not None:
            attributions = self.attribute_direction_jacobian(direction, normalized_features)
        else:
            attributions = self.attribute_direction_correlation(direction, embeddings, raw_features)

        # Top markets along direction
        projections = embeddings @ direction
        sorted_idx = np.argsort(projections)
        top_pos = [condition_ids[i] for i in sorted_idx[-5:]][::-1]
        top_neg = [condition_ids[i] for i in sorted_idx[:5]]

        # Plain-language description
        description = self._generate_description(probe_result, attributions)

        return InterpretedDirection(
            name=probe_result.concept_name,
            description=description,
            direction_vector=direction,
            input_attributions=attributions,
            top_positive_markets=top_pos,
            top_negative_markets=top_neg,
            probe_accuracy=probe_result.accuracy,
        )

    def interpret_all(
        self,
        probe_results: list[ProbeResult],
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        condition_ids: list[str],
    ) -> list[InterpretedDirection]:
        """Interpret all significant probes."""
        interpretations = []
        for pr in probe_results:
            if pr.is_significant:
                interp = self.interpret_probe(pr, embeddings, raw_features, condition_ids)
                interpretations.append(interp)
        return interpretations

    def generate_report(
        self,
        interpretations: list[InterpretedDirection],
        pca_result: Optional[dict] = None,
        novel_directions: Optional[list] = None,
    ) -> str:
        """Generate human-readable plain-text report."""
        lines = [
            "=" * 70,
            "EMBEDDING INTERPRETATION REPORT",
            "=" * 70,
            "",
        ]

        if pca_result:
            cum_var = pca_result.get("cumulative_variance", [])
            if len(cum_var) > 0:
                lines.append(f"PCA: {cum_var[-1]:.1%} variance captured in {len(cum_var)} components")
                for k in [5, 10, 20]:
                    if k - 1 < len(cum_var):
                        lines.append(f"  First {k} components: {cum_var[k-1]:.1%}")
                lines.append("")

        if novel_directions:
            lines.append(f"NOVEL DIRECTIONS ({len(novel_directions)} found):")
            for nd in novel_directions[:5]:
                lines.append(f"  PC{nd.direction_idx}: var_explained={nd.variance_explained:.3f}, "
                             f"max|r|={nd.max_correlation:.2f}")
                lines.append(f"    {nd.description}")
            lines.append("")

        lines.append("INTERPRETED PROBE DIRECTIONS:")
        lines.append("-" * 70)

        for interp in interpretations:
            lines.append(f"\n  [{interp.name}]  accuracy={interp.probe_accuracy:.3f}")

            # Top 5 attributed features
            sorted_attr = sorted(interp.input_attributions.items(), key=lambda x: -abs(x[1]))[:5]
            lines.append("  Top attributed input features:")
            for fname, score in sorted_attr:
                lines.append(f"    {fname:<25} r={score:+.3f}")

            lines.append(f"  Description: {interp.description}")

            lines.append(f"  Top positive markets: {', '.join(interp.top_positive_markets[:3])}")
            lines.append(f"  Top negative markets: {', '.join(interp.top_negative_markets[:3])}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def _generate_description(
        self,
        probe_result: ProbeResult,
        attributions: dict[str, float],
    ) -> str:
        """Generate a plain-language description of a probe direction."""
        sorted_attr = sorted(attributions.items(), key=lambda x: -abs(x[1]))

        # Top 3 features
        top = sorted_attr[:3]
        parts = []
        for fname, score in top:
            if abs(score) > 0.1:
                direction = "positively" if score > 0 else "negatively"
                parts.append(f"{fname} ({direction}, r={score:+.2f})")

        if not parts:
            return f"Direction for '{probe_result.concept_name}' with weak input correlations"

        feat_str = ", ".join(parts)
        return (
            f"This direction captures {probe_result.concept_name} "
            f"(accuracy={probe_result.accuracy:.2f}) primarily through {feat_str}"
        )
