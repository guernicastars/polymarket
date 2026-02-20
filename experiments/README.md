# Disentangled Feature Analysis via Neural Embeddings: A Polymarket Testbed

## Abstract

High-dimensional feature spaces constructed from financial market data exhibit
severe multicollinearity: price momentum, volume trends, volatility measures,
and orderbook imbalances are mechanistically entangled. Classical statistical
tests (VIF, Wald tests, linear regression coefficients) become unreliable in
this regime -- variance inflation factors diverge, coefficient signs flip, and
feature importance rankings become unstable under minor perturbations. We
propose that neural network embeddings -- specifically variational autoencoders
trained on resolved Polymarket prediction contracts -- learn a compressed
representation in which the latent directions are approximately orthogonal and
semantically meaningful. By comparing multicollinearity metrics, linear probe
accuracy, and statistical significance between raw feature space and the learned
embedding space, we provide empirical evidence that embeddings produce
disentangled representations amenable to classical interpretability methods.

## Hypothesis

> In high-dimensional market feature spaces where multicollinearity renders
> classical feature importance analysis unreliable, neural network embeddings
> learn disentangled representations in which (1) variance inflation factors
> drop below critical thresholds, (2) semantic concepts become linearly
> separable, and (3) standard statistical tests recover valid, interpretable
> feature attributions.

**Null hypothesis (H0):** Embedding dimensions exhibit comparable or worse
multicollinearity than raw features, and linear probes on embeddings show no
improvement in concept separability.

**Alternative hypothesis (H1):** Embedding dimensions achieve VIF < 10 (vs.
VIF >> 10 in raw space), condition numbers improve by at least one order of
magnitude, and linear probes on embeddings outperform raw-feature probes by a
statistically significant margin (p < 0.05).

## Background

### The Multicollinearity Problem

In prediction market data, features are not independent. Consider a market
approaching resolution:

- **Price momentum** (1h, 4h, 24h returns) captures directional movement, but
  short windows are nested within longer windows, creating mechanical
  correlation.
- **Volume metrics** (trade count, USDC volume, volume ratios) all increase as
  resolution approaches, producing shared temporal trends.
- **Orderbook features** (bid/ask spread, depth imbalance, OBI) reflect the
  same underlying liquidity dynamics that drive price.
- **Volatility measures** (realized volatility, high-low range) are
  mechanistically linked to momentum and volume.

The result: a feature matrix with condition numbers exceeding 10^4, VIF values
regularly above 50, and pairwise correlations forming dense blocks. In this
regime:

1. **OLS coefficients are unstable.** Small perturbations in data produce large
   swings in estimated coefficients (Belsley et al., 1980).
2. **Significance tests fail.** Inflated standard errors mask true effects;
   the Wald test loses power.
3. **Feature importance is arbitrary.** SHAP values, permutation importance,
   and gradient-based attribution all inherit the degeneracy -- they cannot
   distinguish genuinely important features from correlated proxies.

### Why Embeddings Help

A neural network autoencoder, trained to reconstruct the input feature vector
through a bottleneck, must learn a compressed code that preserves information.
The key insight is structural:

- **Information compression forces factorization.** The bottleneck cannot
  simply copy correlated features; it must learn shared structure.
- **Reconstruction loss encourages coverage.** Every input feature must be
  recoverable, so the code must capture all independent axes of variation.
- **VAE regularization promotes disentanglement.** The KL divergence term
  penalizes deviation from an isotropic Gaussian prior, encouraging orthogonal
  latent dimensions (Higgins et al., 2017).

The result is an embedding space where:
- Dimensions corresponding to "momentum" are separated from "volume."
- "Liquidity" factors are orthogonal to "sentiment" factors.
- Classical tests (VIF, correlation, linear regression) become valid again.

## Core Argument

```
  Raw Features (entangled)           Embedding Space (disentangled)
  ┌─────────────────────┐           ┌─────────────────────────────┐
  │ momentum_1h  ─┐     │           │  z_1 (momentum)     [VIF<5]│
  │ momentum_4h  ─┤─┐   │  train   │  z_2 (volume)       [VIF<5]│
  │ momentum_24h ─┘ │   │  ────►   │  z_3 (liquidity)    [VIF<5]│
  │ volume_1h   ────┘   │  VAE     │  z_4 (sentiment)    [VIF<5]│
  │ volume_4h   ────┐   │           │  ...                       │
  │ spread      ────┤   │           │  z_64                      │
  │ obi         ────┘   │           └───────────┬─────────────────┘
  │ volatility  ───┐    │                       │
  │ trade_size  ───┘    │                       ▼
  │        VIF >> 10    │           ┌─────────────────────────────┐
  └─────────────────────┘           │  Linear Probes:             │
                                    │   "Is this a crypto market?"│
                                    │   → z_7 direction, p<0.001  │
                                    │  Statistical Tests:         │
                                    │   VIF reduced 10x           │
                                    │   Condition number: 10^4→10 │
                                    │  Feature Attribution:       │
                                    │   z_4 ← spread + obi (0.8)  │
                                    │   Interpretable mapping     │
                                    └─────────────────────────────┘
```

The pipeline:

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ ClickHouse│───►│ Feature  │───►│  VAE     │───►│  Probes  │───►│  Report  │
  │ resolved  │    │ Engineer │    │ Training │    │ & Stats  │    │ & Viz    │
  │ contracts │    │ (X, y)   │    │ (Z = enc)│    │ (compare)│    │ (verdict)│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
       data/           data/          models/         models/         results/
     extract.py      features.py    train.py       probes.py       figures/
                                                  statistics.py    reports/
```

## Methodology

### Phase 1: Data Extraction

Query ClickHouse Cloud for all resolved Polymarket contracts. For each market
with sufficient data (>= 50 price observations, >= 20 trades), extract a
feature vector spanning:

| Feature Group   | Features                                          | Count |
|-----------------|---------------------------------------------------|-------|
| Price dynamics  | Returns (1h, 4h, 12h, 24h, 48h, 168h)            | 6     |
| Volatility      | Realized vol (4h, 12h, 24h, 48h, 168h)            | 5     |
| Volume          | Trade count, USDC volume, volume ratios (1h-48h)  | ~8    |
| Orderbook       | Bid-ask spread, depth, OBI                         | 3     |
| Market metadata | Category (encoded), time to expiry, liquidity      | ~4    |
| Composite       | Composite signal score (if available)              | 1     |

Total: approximately 25-30 features per market snapshot.

Snapshots taken at fixed horizons before resolution (1h, 4h, 12h, 24h, 48h,
7d, 30d) to capture the approach dynamics. Temporal train/val/test split
(70/15/15) by resolution date to prevent future leakage.

### Phase 2: Autoencoder Training

Train a Variational Autoencoder:
- **Encoder:** Input (n_features) -> 256 -> 128 -> 64 (embedding dim)
- **Decoder:** 64 -> 128 -> 256 -> Input (n_features)
- **Loss:** MSE reconstruction + beta * KL divergence
- **Regularization:** Dropout (0.1), early stopping (patience=20)

The embedding dimension (64) is deliberately larger than the expected number of
independent factors (~8-12) to avoid over-compression. The model will learn to
use only the dimensions that carry information; unused dimensions will collapse
to the prior.

### Phase 3: Disentanglement Analysis

**Multicollinearity comparison (the key test):**
- Compute VIF for all raw features. Expected: many VIF > 10, some > 50.
- Compute VIF for all embedding dimensions. Expected: VIF < 10 for most.
- Compare condition numbers: raw feature matrix vs. embedding matrix.
- Correlation heatmaps: dense blocks in raw space vs. sparse in embedding.

**Linear probes (concept separability):**
- For each concept (category, outcome, volatility regime, time bucket):
  - Train linear classifier on raw features -> accuracy_raw
  - Train linear classifier on embeddings -> accuracy_embed
  - If accuracy_embed >= accuracy_raw: information preserved
  - If probe uses fewer embedding dims: information concentrated

**Statistical tests:**
- Wald test on logistic regression coefficients (raw vs. embedding)
- Likelihood ratio tests for nested models
- Permutation tests (1000 iterations) for significance

### Phase 4: Interpretation and Transfer

Map embedding directions back to input features via gradient attribution.
Identify which raw features contribute to each learned factor. This produces
an interpretable factor model: "embedding dimension 3 = 0.6 * spread + 0.3 *
obi + 0.1 * volume_ratio" -- a discovered liquidity factor.

Discuss transferability to art market data (auction prices, artist features,
medium, provenance) where the same multicollinearity problems arise but
labeled data is scarcer.

## How to Run

### Prerequisites

```bash
# From the polymarket root directory
cd experiments

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure ClickHouse credentials are configured
# (reads from ../pipeline/.env automatically)
```

### Full Pipeline

```bash
# Run the complete experiment end-to-end
python run_experiment.py

# With custom config
python run_experiment.py --config custom_config.yaml

# Skip data extraction (use cached data)
python run_experiment.py --skip-extract

# Skip model training (use saved checkpoint)
python run_experiment.py --skip-train

# Skip both (analysis only)
python run_experiment.py --skip-extract --skip-train
```

### Individual Steps

```bash
# Step 1: Extract data from ClickHouse
python -m data.extract

# Step 2: Train autoencoder
python -m models.train

# Step 3: Run probes and statistical tests
python -m models.probes
python -m models.statistics

# Step 4: Generate visualizations
python -m models.visualize
```

### Outputs

All outputs are written to `results/`:

```
results/
├── data/
│   ├── features.npz          — Feature matrix X, labels y, metadata
│   └── metadata.json         — Feature names, market IDs, split indices
├── checkpoints/
│   └── vae_best.pt           — Best model checkpoint
├── figures/
│   ├── correlation_raw.png   — Raw feature correlation heatmap
│   ├── correlation_embed.png — Embedding correlation heatmap
│   ├── vif_comparison.png    — VIF bar chart (raw vs. embedding)
│   ├── tsne_outcome.png      — t-SNE colored by outcome
│   ├── tsne_category.png     — t-SNE colored by category
│   ├── probe_accuracy.png    — Probe accuracy comparison
│   └── attribution.png       — Feature attribution heatmap
└── reports/
    └── experiment_report.json — Full metrics, verdicts, timestamps
```

## Expected Outputs and Success Criteria

| Metric                  | Raw Features (expected)  | Embeddings (expected) | Verdict if true          |
|-------------------------|--------------------------|-----------------------|--------------------------|
| Max VIF                 | > 50                     | < 10                  | H1: multicollinearity resolved |
| Condition number        | > 10,000                 | < 100                 | H1: numerically stable   |
| Mean pairwise |corr|    | > 0.5                    | < 0.2                 | H1: decorrelated         |
| Probe accuracy (cat.)   | ~60%                     | >= 60%                | Information preserved    |
| Probe accuracy (outcome)| ~55%                     | >= 55%                | Predictive content kept  |
| Significant dimensions  | Unstable                 | Consistent            | H1: reliable attribution |

**The hypothesis holds if:**
1. VIF drops below 10 for >80% of embedding dimensions
2. Linear probes on embeddings match or exceed raw-feature probes
3. Statistical tests on embedding coefficients yield stable, significant results

**The hypothesis fails if:**
1. Embeddings show comparable multicollinearity to raw features
2. Probe accuracy drops substantially (information lost in compression)
3. No interpretable mapping from embedding directions to input features

## Connection to Art Market Transfer

The ultimate application is art market analytics, where:
- Features (artist reputation, medium, size, provenance, auction house) are
  deeply entangled
- Labeled data is scarce (resolved auctions with known hammer prices)
- Classical hedonic regression suffers from severe multicollinearity

Polymarket serves as an ideal testbed because:
- Binary resolution provides clean labels (Yes/No outcome)
- High data volume (45K+ markets, continuous price/volume data)
- Feature structure mirrors art market entanglement patterns
- If embeddings disentangle prediction market features, the same architecture
  can be applied to auction data with confidence

## References

- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics:
  Identifying Influential Data and Sources of Collinearity.* Wiley.
- Higgins, I., et al. (2017). "beta-VAE: Learning Basic Visual Concepts with
  a Constrained Variational Framework." *ICLR 2017.*
- Kingma, D. P. & Welling, M. (2014). "Auto-Encoding Variational Bayes."
  *ICLR 2014.*
- Kim, H. & Mnih, A. (2018). "Disentangling by Factorising." *ICML 2018.*
- Locatello, F., et al. (2019). "Challenging Common Assumptions in the
  Unsupervised Learning of Disentangled Representations." *ICML 2019.*
- Rajan, R. & Zingales, L. (1995). "What Do We Know about Capital Structure?"
  *Journal of Finance, 50(5).*

---

*Experiment infrastructure for the Polymarket Signals project. See
[polymarket/CLAUDE.md](../CLAUDE.md) for full system documentation.*
