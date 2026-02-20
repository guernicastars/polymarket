"""Beta-Bernoulli conjugate Bayesian combiner for prediction markets.

Uses the market price to initialize a Beta prior, then updates with
evidence from multiple signal sources expressed as likelihood ratios.

Key design principle: CONSERVATIVE. Only departs from market price when
multiple independent signals agree with high confidence. Markets are
efficient most of the time.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .evidence import EvidenceUpdate

logger = logging.getLogger(__name__)


@dataclass
class BetaPosterior:
    """Beta distribution for a single market's event probability."""

    alpha: float = 1.0
    beta: float = 1.0
    last_updated: Optional[datetime] = None
    n_updates: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    @property
    def concentration(self) -> float:
        return self.alpha + self.beta

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval using Beta quantiles."""
        try:
            from scipy.stats import beta as beta_dist

            tail = (1 - level) / 2
            lo = beta_dist.ppf(tail, self.alpha, self.beta)
            hi = beta_dist.ppf(1 - tail, self.alpha, self.beta)
            return (float(lo), float(hi))
        except ImportError:
            # Fallback: approximate with normal
            std = math.sqrt(self.variance)
            return (
                max(0.0, self.mean - 1.96 * std),
                min(1.0, self.mean + 1.96 * std),
            )


@dataclass
class BayesianPrediction:
    """Full output of the Bayesian combiner for one market."""

    condition_id: str
    posterior_mean: float
    posterior_alpha: float
    posterior_beta: float
    credible_lo: float
    credible_hi: float
    market_price: float
    edge: float
    confidence: float
    n_evidence_sources: int
    evidence_agreement: float
    kelly_fraction: float
    direction: str
    evidence_detail: str = "{}"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class BayesianCombiner:
    """Beta-Bernoulli Bayesian combiner for prediction markets.

    Architecture:
      1. Market price → Beta prior (informative, kappa concentration)
      2. Each signal → likelihood ratio (Bayes factor)
      3. Sequential conjugate update
      4. Time decay toward market price
      5. Deviation cap (max 7.5% from market)
    """

    def __init__(
        self,
        prior_strength: float = 20.0,
        max_single_update: float = 5.0,
        decay_halflife_hours: float = 4.0,
        min_evidence_sources: int = 2,
        market_efficiency: float = 0.85,
        min_edge: float = 0.02,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.prior_strength = prior_strength
        self.max_single_update = max_single_update
        self.decay_halflife_hours = decay_halflife_hours
        self.min_evidence_sources = min_evidence_sources
        self.market_efficiency = market_efficiency
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction

    def initialize_prior(
        self, condition_id: str, market_price: float
    ) -> BetaPosterior:
        """Create Beta prior from market price.

        Beta(p * kappa, (1-p) * kappa) has mean = p and concentration = kappa.
        """
        p = max(0.01, min(0.99, market_price))
        alpha = p * self.prior_strength
        beta = (1.0 - p) * self.prior_strength
        return BetaPosterior(
            alpha=alpha,
            beta=beta,
            last_updated=datetime.now(timezone.utc),
            n_updates=0,
        )

    def update(
        self,
        condition_id: str,
        evidence: list[EvidenceUpdate],
        current_market_price: float,
        existing_posterior: Optional[BetaPosterior] = None,
    ) -> BayesianPrediction:
        """Run full Bayesian update cycle for one market.

        Steps:
          1. Retrieve or initialize posterior
          2. Decay toward current market price
          3. Apply each evidence update as likelihood ratio
          4. Cap deviation from market price
          5. Compute CI and Kelly sizing
        """
        now = datetime.now(timezone.utc)
        p_mkt = max(0.01, min(0.99, current_market_price))

        # Step 1: Get or create posterior
        if existing_posterior is not None:
            posterior = existing_posterior
        else:
            posterior = self.initialize_prior(condition_id, p_mkt)

        # Step 2: Decay toward market price
        if posterior.last_updated is not None:
            hours_elapsed = (
                now - posterior.last_updated
            ).total_seconds() / 3600.0
            posterior = self._decay_toward_market(posterior, p_mkt, hours_elapsed)

        # Step 3: Apply evidence updates
        evidence_detail = {}
        directions = []
        for ev in evidence:
            if ev.likelihood_ratio == 1.0:
                continue  # Uninformative, skip
            posterior = self._apply_evidence(posterior, ev)
            direction = "YES" if ev.likelihood_ratio > 1.0 else "NO"
            directions.append(direction)
            evidence_detail[ev.source] = {
                "K": round(ev.likelihood_ratio, 4),
                "w": round(ev.weight, 3),
                "dir": direction,
            }

        # Step 4: Cap deviation
        posterior = self._check_deviation_cap(posterior, p_mkt)

        # Step 5: Renormalize concentration if too high
        max_conc = self.prior_strength * 5
        if posterior.concentration > max_conc:
            scale = max_conc / posterior.concentration
            posterior.alpha *= scale
            posterior.beta *= scale

        posterior.last_updated = now
        posterior.n_updates += 1

        # Compute outputs
        post_mean = posterior.mean
        ci_lo, ci_hi = posterior.credible_interval(0.95)
        edge = post_mean - p_mkt
        n_sources = len([e for e in evidence if e.likelihood_ratio != 1.0])

        # Evidence agreement: fraction of sources agreeing on direction
        if directions:
            majority = max(
                directions.count("YES"), directions.count("NO")
            )
            agreement = majority / len(directions)
        else:
            agreement = 0.0

        # Confidence: normalized inverse variance
        confidence = min(1.0, posterior.concentration / (self.prior_strength * 3))

        # Kelly criterion
        kelly = self._kelly_from_posterior(post_mean, p_mkt)

        # Direction (only if enough evidence sources and edge exceeds minimum)
        if (
            edge > self.min_edge
            and n_sources >= self.min_evidence_sources
        ):
            direction = "BUY"
        elif (
            edge < -self.min_edge
            and n_sources >= self.min_evidence_sources
        ):
            direction = "SELL"
        else:
            direction = "HOLD"

        return BayesianPrediction(
            condition_id=condition_id,
            posterior_mean=post_mean,
            posterior_alpha=posterior.alpha,
            posterior_beta=posterior.beta,
            credible_lo=ci_lo,
            credible_hi=ci_hi,
            market_price=p_mkt,
            edge=edge,
            confidence=confidence,
            n_evidence_sources=n_sources,
            evidence_agreement=agreement,
            kelly_fraction=kelly,
            direction=direction,
            evidence_detail=json.dumps(evidence_detail),
            timestamp=now,
        )

    def _decay_toward_market(
        self,
        posterior: BetaPosterior,
        market_price: float,
        hours_elapsed: float,
    ) -> BetaPosterior:
        """Exponential decay of posterior toward market price.

        Over time, if no new evidence arrives, the posterior relaxes
        back toward the market price. Implements the assumption that
        market prices are informationally efficient on average.
        """
        if hours_elapsed <= 0:
            return posterior

        decay_factor = 0.5 ** (hours_elapsed / self.decay_halflife_hours)

        # Decay concentration toward prior_strength
        current_conc = posterior.concentration
        target_conc = self.prior_strength
        new_conc = target_conc + (current_conc - target_conc) * decay_factor

        # Shift mean toward market price
        current_mean = posterior.mean
        new_mean = market_price + (current_mean - market_price) * decay_factor
        new_mean = max(0.01, min(0.99, new_mean))

        new_conc = max(2.0, new_conc)  # Minimum concentration
        return BetaPosterior(
            alpha=new_mean * new_conc,
            beta=(1.0 - new_mean) * new_conc,
            last_updated=posterior.last_updated,
            n_updates=posterior.n_updates,
        )

    def _apply_evidence(
        self, posterior: BetaPosterior, evidence: EvidenceUpdate
    ) -> BetaPosterior:
        """Apply a single evidence update via likelihood ratio.

        For K > 1 (evidence favors YES): alpha' = alpha * K^w
        For K < 1 (evidence favors NO): beta' = beta / K^w
        """
        k = evidence.likelihood_ratio
        w = evidence.weight

        # Cap the Bayes factor
        k = max(1.0 / self.max_single_update, min(k, self.max_single_update))

        # Attenuate: K_eff = K^w (w < 1 shrinks the update)
        k_eff = k ** w

        if k_eff >= 1.0:
            return BetaPosterior(
                alpha=posterior.alpha * k_eff,
                beta=posterior.beta,
                last_updated=posterior.last_updated,
                n_updates=posterior.n_updates,
            )
        else:
            return BetaPosterior(
                alpha=posterior.alpha,
                beta=posterior.beta / k_eff,
                last_updated=posterior.last_updated,
                n_updates=posterior.n_updates,
            )

    def _check_deviation_cap(
        self, posterior: BetaPosterior, market_price: float
    ) -> BetaPosterior:
        """Prevent posterior from deviating too far from market price."""
        max_deviation = (1.0 - self.market_efficiency) * 0.5
        post_mean = posterior.mean

        if abs(post_mean - market_price) > max_deviation:
            sign = 1.0 if post_mean > market_price else -1.0
            capped_mean = market_price + sign * max_deviation
            capped_mean = max(0.01, min(0.99, capped_mean))
            conc = posterior.concentration
            return BetaPosterior(
                alpha=capped_mean * conc,
                beta=(1.0 - capped_mean) * conc,
                last_updated=posterior.last_updated,
                n_updates=posterior.n_updates,
            )
        return posterior

    def _kelly_from_posterior(self, prob: float, market_price: float) -> float:
        """Kelly criterion from posterior probability."""
        edge = prob - market_price
        if abs(edge) < self.min_edge:
            return 0.0

        if edge > 0:
            # Buying YES: odds = (1/p_mkt) - 1
            b = (1.0 / max(market_price, 0.01)) - 1.0
            q = 1.0 - prob
            f = (prob * b - q) / b
        else:
            # Buying NO: odds = (1/(1-p_mkt)) - 1
            p_no = 1.0 - prob
            p_no_mkt = 1.0 - market_price
            b = (1.0 / max(p_no_mkt, 0.01)) - 1.0
            q = 1.0 - p_no
            f = (p_no * b - q) / b

        return max(0.0, f * self.kelly_fraction)
