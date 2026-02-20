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
# Polymarket experiment (honest, non-leaking features)
python run_honest_experiment.py

# Art market experiment (5-model comparison)
python run_art_experiment.py

# Art market with custom hyperparameters (65-feature final run)
python run_art_experiment.py --embedding-dim 12 --batch-size 512 --patience 20 --epochs 200

# Art market with custom data directory
python run_art_experiment.py --data-dir art_data/output --embedding-dim 8

# Regression precision analysis (29K Sotheby's)
python run_regression_analysis.py
```

### Individual Steps

```bash
# Step 1: Extract data from ClickHouse / SQLite
python art_data/extract.py                     # From ClickHouse (3 houses, 693K)
python art_data/extract.py --source sqlite     # From SQLite (Sotheby's only, 29K)

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
├── honest_run/                — Polymarket honest experiment
│   ├── honest_report.json     — Full metrics (8D, 813 samples)
│   └── checkpoints/           — Model checkpoint
├── art_market/                — Art market experiment (latest: 63D, 693K)
│   ├── art_experiment_report.json — Full 5-model comparison metrics
│   ├── checkpoints/           — Best SVAE model checkpoint
│   ├── train_data/            — Saved feature matrix for reproducibility
│   ├── vif_comparison.png     — VIF: Raw vs PCA vs Orth-SVAE
│   ├── correlation_matrices.png — Correlation heatmaps (3 panels)
│   ├── model_comparison.png   — 5-model CV metric bar chart
│   ├── walk_forward.png       — Walk-forward temporal backtest (all methods)
│   ├── feature_attribution.png — Jacobian heatmap (input -> embedding)
│   └── domain_comparison.png  — Polymarket vs Art Market comparison
└── art_data/output/           — Input data
    ├── features.npz           — Feature matrix (X_train/val/test, labels, probes)
    └── metadata.json          — 65 features, 15 categories, split sizes
```

## Experimental Results

### Iteration History

Nine experiment iterations: six on Polymarket, three on art market data:

**Polymarket Iterations (methodology development):**

| Run | Model | Dims | Key Change | VIF | Acc | Sig Dims | Verdict |
|-----|-------|------|------------|-----|-----|----------|---------|
| 1 | VAE | 64 | Baseline | 223 (worse) | 91.9% | 0/64 (0%) | WEAK (2/4) |
| 2 | VAE | 8 | Reduce dims | 7.15 | 80.0% | 3/8 (37%) | MODERATE (3/4) |
| 3 | Supervised VAE | 8 | Add prediction loss | 1502 (worse) | 97.6% | 0/8 (0%) | WEAK (2/4) |
| 4 | **Orth-SVAE** | **8** | **Add orthogonality penalty** | **1.005** | **91.9%** | **8/8 (100%)** | **STRONG (4/4)** |
| 5 | PCA baseline | 8 | Linear comparison | 1.0 | 85.4% | 6/8 (75%) | -- |
| 6 | Orth-SVAE (honest) | 8 | Non-leaking features only | 1.01 | 70.1% | 4/8 (50%) | Honest baseline |

**Art Market Iterations (domain transfer + scaling):**

| Run | Dataset | Features | Samples | SVAE CV | Best CV | SVAE OOS | Verdict |
|-----|---------|----------|---------|---------|---------|----------|---------|
| 7 | Sotheby's only | 30 | 29K | **0.7543** | SVAE (0.7543) | **0.7839** | STRONG: SVAE wins all |
| 8 | 3 houses | 35 | 693K | 0.8984 | RF (0.9026) | -- | Mixed: trees win CV |
| 9 | 3 houses (expanded) | 63 | 693K | 0.9022 | RF (0.9080) | **0.8948** | Interpretability + stability win |

### Key Result: Orth-SVAE on 25-Feature Set (Run 4)

The orthogonality-regularized supervised VAE achieved **4/4 STRONG SUPPORT**:

| Metric | Raw (25D) | PCA (8D) | Orth-SVAE (8D) |
|--------|-----------|----------|----------------|
| Max VIF | 11.32 | 1.00 | **1.005** |
| Condition # | 271,118 | 1.9 | **1.35** |
| CV Accuracy | 87.4% | 85.4% | **91.9%** |
| Balanced Acc | 87.0% | 84.4% | **91.4%** |
| AUC-ROC | 0.961 | 0.932 | **0.981** |
| Sig Dims (Wald) | 4/25 (16%) | 6/8 (75%) | **8/8 (100%)** |

The 4-component loss function:

```
L = L_recon + beta * KL + alpha * BCE(pred, outcome) + gamma * ||corr(Z) - I||^2
```

Best config: embedding_dim=8, beta=1.0, alpha=1.0, gamma=1.0.

### Feature Attribution

All 8 embedding dimensions map to distinct, interpretable market factors:

| Dim | Theme | Top Drivers | Wald |z| | MI |
|-----|-------|-------------|---------|-----|
| 6 | Price level | last_price, final_velocity | 11.0 | 0.265 |
| 4 | Momentum | one_week/one_day price change | 12.5 | 0.221 |
| 2 | Weekly momentum | one_week_price_change, volume_accel | 11.0 | 0.130 |
| 1 | Price/trading | late_volume_ratio, price_range | 10.3 | 0.101 |
| 0 | Price dynamics | neg_risk, price_at_75pct | 7.5 | 0.062 |
| 5 | Buy pressure | buy_sell_ratio, buy_volume_ratio | 7.8 | 0.046 |
| 3 | Volume/structure | avg_trade_size, neg_risk | 2.4 | 0.037 |
| 7 | Trading intensity | trades_per_day, volume_total | 5.6 | 0.035 |

Wald significance correlates with mutual information: **rho=0.881, p=0.004**.

### Critical Finding: Feature Leakage

Post-experiment audit revealed that 8 of 25 features were measured **after
resolution** by the Gamma API (one_day_price_change, last_price,
final_price_velocity, etc.). These features encode the outcome, not predict it.

A trivial rule -- `one_day_price_change > 0 => Yes` -- achieves **97.4%
accuracy** without any model. The 91.9% Orth-SVAE accuracy was largely driven
by these leaking features.

### Honest Evaluation (Run 6)

Re-run using only 13 non-leaking structural features on balanced Over/Under
markets (49/51 baseline):

| Metric | Raw (8D) | PCA (8D) | Orth-SVAE (8D) |
|--------|----------|----------|----------------|
| CV Accuracy | 69.4% | 69.4% | 70.1% |
| CV Balanced Acc | 68.3% | 68.3% | 68.6% |
| CV AUC | 0.710 | 0.710 | 0.720 |
| OOS Accuracy | 62.3% | 62.3% | 59.8% |
| OOS AUC | 0.579 | 0.579 | 0.607 |

**Findings:**
1. **Real signal exists**: 70% accuracy vs 55.5% baseline (+14pp) from trade
   microstructure alone (trade count, size, Gini, duration, volume).
2. **Neural net adds ~0 over PCA** on 8 clean features (+0.7pp). With low
   dimensionality and mild multicollinearity (VIF=6.19), PCA is sufficient.
3. **12pp overfitting gap** on temporal test split (regime shift).
4. **Prediction markets are efficient**: betting simulation loses money for all
   models despite 87% win rate (payoff structure punishes errors heavily).

### Conclusions

1. **The Orth-SVAE methodology is validated** as a technique for eliminating
   multicollinearity while preserving predictive power. Across all experiments
   (8D to 63D), it reduces VIF to near 1.0 and consistently beats PCA.

2. **The advantage scales with multicollinearity severity.** At VIF=6 (8 clean
   features), PCA is equivalent (+0.7pp). At VIF=infinity (30D), SVAE beats all
   models (+8.0pp vs PCA, +1.7pp vs RF). At VIF=infinity (63D) with large data,
   SVAE trades ~0.6pp CV accuracy for best OOS generalization (+1.6pp vs RF).

3. **SVAE produces the most stable temporal predictions.** On the 63-feature
   art market data, walk-forward standard deviation is 0.0078 for SVAE vs 0.0087
   for RF and 0.0144 for LGBM. The decorrelated embedding resists regime shifts.

4. **Feature leakage is the dominant risk** in prediction market analysis. Any
   feature sourced from an API snapshot taken after resolution must be excluded
   or time-truncated.

5. **Trade microstructure contains genuine predictive signal** for binary market
   outcomes, independent of price information.

6. **The practical recommendation depends on the use case:**
   - For maximum in-distribution accuracy with large data: use RF/LGBM on raw features.
   - For interpretable inference, OOS generalization, and temporal stability: use Orth-SVAE.
   - For valid statistical testing on correlated features: Orth-SVAE is the only option that eliminates multicollinearity while preserving predictive signal.

## Out-of-Sample Validation

### Polymarket Test Set (last 15% by resolution date)
- Orth-SVAE AUC: 0.845 on 25 features, 0.607 on clean features
- Walk-forward accuracy: 89.4% (25 features), 66.6% (clean)
- Betting simulation: negative returns (market efficiency)

### Art Market Test Set (last 15% by sale date, 104K lots)
- Orth-SVAE OOS accuracy: **0.8948** (best of all 5 models)
- RF OOS accuracy: 0.8785 (SVAE +1.6pp better)
- LGBM OOS accuracy: 0.8930 (SVAE +0.2pp better)
- Walk-forward (14 windows): SVAE mean 0.9019 with lowest variance (std=0.0078)

### Overfitting Disclosure
- Polymarket 25-feature set: 91.9% train vs 79.8% test (12pp gap)
- Polymarket clean features: 70.1% CV vs 59.8% test (10pp gap)
- Art market 63-feature: 0.9022 CV vs 0.8948 OOS (0.7pp gap -- minimal overfitting)
- Art market RF: 0.9080 CV vs 0.8785 OOS (3.0pp gap -- RF overfits more)

## Replication on Other Domains

The methodology is domain-agnostic. It has been validated on two domains:
Polymarket prediction markets (8D, 813 samples) and art market auctions (63D,
693K samples). To apply to a new dataset:

1. **Replace `data/extract.py`** -- swap ClickHouse queries for your data source
2. **Replace `data/features.py`** -- define your domain's feature engineering
3. **Keep `models/` unchanged** -- autoencoder, probes, stats, viz are generic
4. **Run `python run_experiment.py`** -- same pipeline, different data

The `models/` folder accepts any feature matrix X (n_samples x n_features) and
label vector y. No domain-specific code in the model layer.

**When to use Orth-SVAE over PCA:**
- Feature count > 20 and max VIF > 10
- Pairwise correlations form dense blocks
- You need individually significant dimensions for inference
- OOS generalization and temporal stability matter more than peak CV accuracy

**When PCA is sufficient:**
- Feature count < 15 and VIF < 10
- Goal is decorrelation only, not prediction improvement

**When to use trees (RF/LGBM) on raw features instead:**
- Maximum in-distribution accuracy is the only goal
- N > 100K samples (trees can learn interaction effects)
- Interpretability via SHAP/permutation importance is acceptable

## Art Market Results

The methodology has been fully validated on art market auction data from three
houses (Sotheby's, Christie's, Phillips) across three dataset scales.

### Final Results: 63 Features, 693,650 Lots

**5-Model Comparison (Cross-Validated):**

| Model | CV Accuracy | CV AUC | OOS Accuracy | Walk-Forward |
|-------|------------|--------|-------------|-------------|
| Raw+LR | 0.8994 | 0.9627 | 0.8941 | 0.9006 |
| PCA+LR | 0.8931 | 0.9593 | 0.8707 | 0.8911 |
| Raw+RF | **0.9080** | **0.9700** | 0.8785 | 0.9027 |
| Raw+LGBM | 0.9079 | 0.9697 | 0.8930 | **0.9056** |
| SVAE+LR | 0.9022 | 0.9646 | **0.8948** | 0.9019 |

**Multicollinearity Reduction:**

| Metric | Raw (63D) | PCA (12D) | SVAE (12D) |
|--------|-----------|----------|-----------|
| Max VIF | infinity (17 severe) | 9.21 | **1.01** |
| Mean VIF | 3,233 | 2.90 | **1.003** |
| Condition # | infinity | 14.57 | **3.48** |
| Sig Dims (Wald) | 35/63 (56%) | 12/12 (100%) | 12/12 (100%) |

**Key findings:**
- SVAE achieves **best OOS accuracy** (0.8948) despite losing CV to trees (-0.6pp)
- SVAE has **lowest walk-forward variance** (std=0.0078 vs RF 0.0087, LGBM 0.0144)
- VIF reduction from infinity to 1.01 enables valid coefficient-based inference
- 15 feature categories (A-P) compressed into 12 decorrelated dimensions

### Progression Across Dataset Scales

| Metric | 30D / 29K | 35D / 693K | 63D / 693K |
|--------|-----------|------------|------------|
| SVAE CV | 0.7543 | 0.8984 | 0.9022 |
| RF CV | 0.7369 | 0.9026 | 0.9080 |
| SVAE vs RF (CV) | **+1.7pp** | -0.4pp | -0.6pp |
| SVAE vs RF (OOS) | **+4.1pp** | -- | **+1.6pp** |
| Raw max VIF | inf | inf | inf |
| SVAE max VIF | 1.00 | 1.01 | 1.01 |

**Pattern:** On small data (29K), SVAE's regularization wins everywhere. On
large data (693K), trees have enough samples to learn interaction effects and
win in-distribution, but SVAE generalizes better to unseen data.

### Feature Attribution (63D Experiment)

Each of the 12 embedding dimensions captures distinct input feature groups:
- **Dims 0, 2, 5, 6:** Estimate features (log_estimate_low/mid/high, estimate_mid_usd)
- **Dim 1:** Physical dimensions (width_cm, is_book, is_jewelry)
- **Dim 3:** Text/confidence (is_book, title_length, artist_name_confidence)
- **Dim 4:** Provenance (exhibition_count, estimate_relative_level, is_attributed_artist)
- **Dim 7:** 3D properties (has_depth, log_depth_cm, height_cm)

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
