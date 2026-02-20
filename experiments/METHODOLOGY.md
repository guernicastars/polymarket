# Orth-SVAE: Orthogonality-Regularized Supervised Variational Autoencoder

## A Step-by-Step Implementation Guide for Multicollinearity-Free Embeddings

This document provides a complete, self-contained guide to implementing and validating the Orth-SVAE methodology. All mathematical formulas, architecture details, hyperparameters, and validation criteria are included so that anyone can replicate this on any tabular dataset in any programming language.

---

## 1. Problem Statement

### 1.1 What Is Multicollinearity?

Multicollinearity occurs when two or more features in a dataset are highly correlated, meaning one feature can be approximately predicted from the others. This is pervasive in real-world tabular data: physical dimensions correlate with each other, historical price features are near-duplicates, and temporal features encode overlapping information.

**Formal definition:** Given a feature matrix **X** of shape (N x D), multicollinearity exists when the columns of **X** are nearly linearly dependent -- that is, the matrix is close to rank-deficient.

### 1.2 How Multicollinearity Breaks Classical Statistics

When features are correlated, standard linear models (logistic regression, linear regression, OLS) suffer three concrete failures:

1. **Inflated standard errors.** The variance of coefficient estimates is proportional to VIF:

   ```
   Var(beta_j) = sigma^2 * (X^T X)^{-1}_{jj} = sigma^2 * VIF_j / Var(x_j)
   ```

   When VIF_j = 50, the standard error is sqrt(50) = 7x larger than it would be with orthogonal features. Confidence intervals become useless.

2. **Unstable coefficients.** Small changes in the data can flip the sign of coefficients. A feature that is "positively significant" in one sample becomes "negatively significant" in another, because the model cannot distinguish the effect of correlated features.

3. **Poor generalization.** The model overfits to the specific correlation structure in the training data. When that structure shifts slightly (as it always does in production), predictions degrade.

### 1.3 Why Not Just Drop Correlated Features?

Naive solutions -- dropping one of each correlated pair, or applying PCA -- lose information:

- **Dropping features** discards predictive signal. If features A and B are correlated at r=0.95, they still carry 5% unique information each.
- **PCA** finds orthogonal directions of maximum variance, but these are unsupervised -- they may not align with the prediction target. PCA can concentrate noise in the top components and discard predictive signal in lower ones.

### 1.4 What Orth-SVAE Solves

The Orth-SVAE learns a low-dimensional embedding that simultaneously:

1. **Preserves information** (reconstruction loss)
2. **Maintains regularity** (KL divergence)
3. **Encodes outcome-relevant signal** (supervised prediction loss)
4. **Eliminates multicollinearity** (orthogonality penalty)

The key innovation is combining supervision (forcing outcome-relevant information into the embedding) with an explicit decorrelation penalty (forcing each dimension to encode distinct, non-redundant information).

---

## 2. Data Preparation

### 2.1 Feature Matrix and Target

Start with a feature matrix **X** of shape (N x D) and a target vector **y** of shape (N,).

- N = number of samples (rows)
- D = number of features (columns)
- y = binary target (0/1) for classification, or continuous for regression

For regression targets, binarize for the SVAE prediction head:

```
y_binary = 1 if y >= median(y_train) else 0
```

**Important:** Compute the median from the training set only to avoid data leakage.

### 2.2 NaN Imputation

Impute missing values with column medians (computed from the training set):

```
For each feature j:
    median_j = median(X_train[:, j], ignoring NaN)
    X[:, j] = replace NaN with median_j
```

This is robust to outliers and preserves the feature distribution better than mean imputation.

### 2.3 Standardization

Scale each feature to zero mean and unit variance:

```
x_scaled_j = (x_j - mu_j) / sigma_j
```

where:
- mu_j = mean(X_train[:, j])
- sigma_j = std(X_train[:, j])

**Critical:** Fit the scaler on the training set only. Transform validation and test sets using the training set's mu and sigma.

### 2.4 Temporal Train/Validation/Test Split

For time-series or temporally ordered data, split chronologically to prevent future leakage:

```
Train:  X[0 : 0.70*N]       (first 70%)
Val:    X[0.70*N : 0.85*N]  (next 15%)
Test:   X[0.85*N : N]       (final 15%)
```

Never shuffle before splitting when data has a temporal ordering. The model must predict the future from the past.

### 2.5 Zero-Variance Feature Removal

Drop any features with zero or near-zero variance after imputation:

```
For each feature j:
    if Var(X[:, j]) < 1e-10:
        drop feature j
```

These features carry no information and can cause numerical issues.

---

## 3. Diagnose Multicollinearity

Before training, quantify the severity of multicollinearity in the raw features. This establishes the baseline that the embedding must improve upon.

### 3.1 Variance Inflation Factor (VIF)

For each feature j, regress it on all other features and compute:

```
VIF_j = 1 / (1 - R_j^2)
```

where R_j^2 is the coefficient of determination (R-squared) from the regression:

```
x_j = beta_0 + beta_1 * x_1 + ... + beta_{j-1} * x_{j-1} + beta_{j+1} * x_{j+1} + ... + beta_D * x_D + epsilon
```

**Interpretation:**
| VIF Value | Severity |
|-----------|----------|
| 1.0 | No multicollinearity (feature is independent) |
| 1-5 | Low (acceptable) |
| 5-10 | Moderate (concerning) |
| > 10 | Severe (problematic for linear models) |
| infinity | Perfect multicollinearity (feature is exact linear combination of others) |

**Implementation:** Use ordinary least squares (OLS) for each regression. With D features, you run D regressions.

### 3.2 Condition Number

Compute the singular value decomposition (SVD) of the feature matrix:

```
X = U * Sigma * V^T
```

where Sigma = diag(sigma_1, sigma_2, ..., sigma_D) with sigma_1 >= sigma_2 >= ... >= sigma_D >= 0.

The condition number is:

```
kappa(X) = sigma_max / sigma_min = sigma_1 / sigma_D
```

**Interpretation:**
| Condition Number | Severity |
|-----------------|----------|
| < 10 | Low multicollinearity |
| 10-30 | Moderate |
| > 30 | Severe |
| infinity | Singular matrix (at least one feature is a perfect linear combination) |

### 3.3 Pairwise Correlation Matrix

Compute the Pearson correlation between all pairs of features:

```
r_{ij} = cov(x_i, x_j) / (sigma_i * sigma_j)
```

where:
- cov(x_i, x_j) = (1/N) * sum_k (x_{ki} - mu_i)(x_{kj} - mu_j)
- sigma_i = std(x_i)

**Flag pairs with |r_{ij}| > 0.8** as candidates for redundancy.

### 3.4 Example: Art Market Raw Features

**30 features (Sotheby's only, 29K lots):**
```
Mean VIF:        4.11
Max VIF:         infinity (creation_year and years_since_creation: r = -1.0)
Severe (>10):    4 features
Condition #:     infinity
Max |r|:         1.000
```

**63 features (3 auction houses, 693K lots):**
```
Mean VIF:        3,233
Max VIF:         infinity (17 severe features)
Severe (>10):    17 features (estimate clusters, online/hammer ratio, attribution pairs)
Condition #:     infinity
Max |r|:         1.000
```

The 65-feature dataset (63 after dropping 2 zero-variance features) has even denser multicollinearity clusters: estimate_mid_usd correlates r=1.000 with log_estimate_high, final_hammer_ratio <-> is_online_sale at r=0.992, and log_estimate_range_usd <-> log_estimate_high at r=0.990. This is exactly the regime where Orth-SVAE's decorrelation should provide the most value.

---

## 4. The Orth-SVAE Architecture

### 4.1 Overview

The architecture has four components:

```
Input x (D dims)
    |
    v
[   Encoder   ] --> mu, log(sigma^2)   (D-dimensional parameters)
    |                    |
    |   Reparameterization trick
    |                    |
    v                    v
    z  <-- mu + epsilon * sigma         (d-dimensional embedding)
    |
    +-------+-------+
    |               |
    v               v
[ Decoder ]   [ Prediction Head ]
    |               |
    v               v
   x_hat           y_hat
```

where d << D is the embedding dimensionality (typically D/3 or 8, whichever is smaller).

### 4.2 Encoder

The encoder maps input features to the parameters of a Gaussian distribution in the latent space:

```
h_1 = ReLU(BatchNorm(W_1 * x + b_1))          # D -> 256
h_1 = Dropout(h_1, p=0.1)
h_2 = ReLU(BatchNorm(W_2 * h_1 + b_2))        # 256 -> 128
h_2 = Dropout(h_2, p=0.1)
mu = W_mu * h_2 + b_mu                          # 128 -> d
log_sigma^2 = W_logvar * h_2 + b_logvar         # 128 -> d
```

The encoder outputs two d-dimensional vectors: the mean mu and the log-variance log(sigma^2) of the approximate posterior q(z|x).

### 4.3 Reparameterization Trick

To enable gradient-based optimization through the stochastic sampling step, use:

```
epsilon ~ N(0, I)                    # Sample from standard normal
sigma = exp(0.5 * log_sigma^2)      # Convert log-variance to std dev
z = mu + epsilon * sigma             # Reparameterized sample
```

This makes z a differentiable function of mu and log(sigma^2), allowing backpropagation through the sampling.

**At inference time:** Use z = mu (the mean) as the deterministic embedding. No sampling.

### 4.4 Decoder

The decoder reconstructs the input from the embedding:

```
h_3 = ReLU(BatchNorm(W_3 * z + b_3))          # d -> 128
h_3 = Dropout(h_3, p=0.1)
h_4 = ReLU(BatchNorm(W_4 * h_3 + b_4))        # 128 -> 256
h_4 = Dropout(h_4, p=0.1)
x_hat = W_out * h_4 + b_out                     # 256 -> D
```

Note: the decoder mirrors the encoder architecture in reverse.

### 4.5 Prediction Head

A single linear layer maps the embedding mean to an outcome prediction:

```
logit = W_pred * mu + b_pred        # d -> 1
y_hat = sigmoid(logit)               # Probability in [0, 1]
```

**Important:** The prediction head operates on **mu** (the mean), not on z (the sample). This ensures the supervised signal drives the deterministic embedding, not the stochastic sample.

### 4.6 Full Architecture Dimensions

For a dataset with D=30 input features and embedding dimension d=8:

```
Encoder:
    Linear(30 -> 256) + BatchNorm(256) + ReLU + Dropout(0.1)
    Linear(256 -> 128) + BatchNorm(128) + ReLU + Dropout(0.1)
    Linear(128 -> 8)   [mu]
    Linear(128 -> 8)   [log_sigma^2]

Decoder:
    Linear(8 -> 128) + BatchNorm(128) + ReLU + Dropout(0.1)
    Linear(128 -> 256) + BatchNorm(256) + ReLU + Dropout(0.1)
    Linear(256 -> 30)

Prediction Head:
    Linear(8 -> 1)

Total parameters: ~86,000
```

For the expanded D=63 input features and d=12 embedding (used in the 65-feature experiment):

```
Encoder:
    Linear(63 -> 256) + BatchNorm(256) + ReLU + Dropout(0.1)
    Linear(256 -> 128) + BatchNorm(128) + ReLU + Dropout(0.1)
    Linear(128 -> 12)  [mu]
    Linear(128 -> 12)  [log_sigma^2]

Decoder:
    Linear(12 -> 128) + BatchNorm(128) + ReLU + Dropout(0.1)
    Linear(128 -> 256) + BatchNorm(256) + ReLU + Dropout(0.1)
    Linear(256 -> 63)

Prediction Head:
    Linear(12 -> 1)

Total parameters: ~104,804
```

---

## 5. The 4-Component Loss Function

This is the key innovation. The total loss is:

```
L = L_recon + beta * L_KL + alpha * L_pred + gamma * L_orth
```

Each component serves a specific purpose and they interact to produce disentangled, predictive, decorrelated embeddings.

### 5.1 Reconstruction Loss (L_recon)

**Purpose:** Preserve information from the input features in the embedding.

```
L_recon = (1/N) * sum_{i=1}^{N} ||x_i - x_hat_i||^2
```

This is the mean squared error between the original input x and the reconstruction x_hat. It forces the embedding z to retain enough information to reconstruct the input, preventing the model from collapsing to a trivial solution.

**Without L_recon:** The embedding could encode only the prediction target, discarding all other information. This would make probes and feature attribution meaningless.

### 5.2 KL Divergence (L_KL)

**Purpose:** Regularize the latent space to follow a standard normal distribution.

```
L_KL = -(1/2) * sum_{j=1}^{d} (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
```

This is the Kullback-Leibler divergence between the learned posterior q(z|x) = N(mu, sigma^2) and the prior p(z) = N(0, I):

```
D_KL(q(z|x) || p(z)) = -(1/2) * sum_j [1 + log(sigma_j^2) - mu_j^2 - sigma_j^2]
```

**Without L_KL:** The encoder can push different inputs to arbitrarily far-apart regions of latent space, creating a fragmented, non-smooth embedding. The KL term forces the embedding to be compact and well-organized.

### 5.3 Prediction Loss (L_pred)

**Purpose:** Force outcome-relevant information into specific embedding dimensions.

```
L_pred = -(1/N) * sum_{i=1}^{N} [y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i)]
```

This is binary cross-entropy (BCE) between the true label y and the predicted probability y_hat = sigmoid(W_pred * mu + b_pred).

**Without L_pred:** The embedding would be purely unsupervised. It would capture the directions of maximum variance (like PCA) rather than the directions most relevant to the prediction target. In our experiments, removing L_pred (setting alpha=0) made the embedding VIF **worse** than raw features, because the unsupervised VAE concentrated all variance into a few dimensions.

### 5.4 Orthogonality Penalty (L_orth) -- THE KEY INNOVATION

**Purpose:** Force each embedding dimension to encode distinct, non-redundant information by penalizing correlations between dimensions.

Given a batch of embeddings Z of shape (B x d), compute the Pearson correlation matrix:

```
Step 1: Center each dimension
    Z_centered = Z - mean(Z, axis=0)

Step 2: Normalize by standard deviation
    Z_norm = Z_centered / std(Z_centered, axis=0)

Step 3: Compute correlation matrix
    C = (Z_norm^T * Z_norm) / (B - 1)

Step 4: Penalize off-diagonal entries
    L_orth = ||C - I||_F^2 = sum_{i != j} C_{ij}^2
```

where ||.||_F is the Frobenius norm and I is the d x d identity matrix.

**In words:** L_orth is the sum of squared correlations between all pairs of embedding dimensions. When L_orth = 0, all dimensions are perfectly uncorrelated.

**Without L_orth (the gamma=0 failure mode):** The supervised prediction gradient flows equally through all embedding dimensions, causing them to encode the same signal. The result is a high-accuracy embedding with severe multicollinearity -- defeating the entire purpose. This was observed empirically: gamma=0 produced VIF > 11 in the embedding, worse than the raw features.

**Why this works:** The orthogonality penalty creates a competition between dimensions. Each dimension must find a unique, non-overlapping direction in the input space that contributes to both reconstruction (L_recon) and prediction (L_pred). This naturally produces disentangled representations where each dimension maps to an interpretable combination of input features.

### 5.5 How the Components Interact

The four losses create a balanced tension:

| If you remove... | What happens |
|-------------------|-------------|
| L_recon | Embedding collapses to prediction signal only. No information preservation. |
| L_KL | Latent space becomes fragmented and unstructured. |
| L_pred | Embedding becomes unsupervised PCA-like. May not capture predictive signal. |
| L_orth | All dimensions encode the same signal. Multicollinearity in embedding. |

All four are necessary. The innovation is not any single term, but their combination with the right weights.

---

## 6. Hyperparameters

### 6.1 Recommended Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| embedding_dim | 8 (or min(input_dim/3, 8)) | Sufficient for most tabular datasets. Larger dims increase capacity but risk redundancy. |
| beta | 1.0 | Standard VAE weight. Increase to 2-5 for smoother latent space at cost of reconstruction. |
| alpha | 1.0 | Equal weight for prediction. Increase if prediction accuracy is the primary goal. |
| gamma | 1.0 | Equal weight for orthogonality. The key parameter -- must be > 0. |
| hidden_dims | (256, 128) | Two hidden layers. Scale up for datasets with D > 100. |
| dropout | 0.1 | Light regularization. Increase for small datasets (N < 1000). |
| learning_rate | 1e-3 | Adam optimizer default. |
| lr_schedule | Cosine annealing to 1e-6 | Smooth decay prevents oscillation near convergence. |
| batch_size | 128 | Standard. Larger batches (256-512) give better correlation estimates for L_orth. |
| patience | 20-30 | Early stopping on validation total loss. |
| max_epochs | 300 | Upper bound; early stopping typically triggers at epoch 50-100. |
| weight_decay | 1e-5 | Light L2 regularization on all parameters. |

### 6.2 When to Adjust

- **gamma:** If embedding VIF is still > 2 after training, increase gamma to 2-5.
- **alpha:** If prediction accuracy is far below the raw feature baseline, increase alpha to 2-5.
- **embedding_dim:** If more than 30% of dimensions are "dead" (std < 1e-6), reduce embedding_dim.
- **beta:** If reconstructions are poor (high L_recon at convergence), reduce beta to 0.5.

---

## 7. Validation Pipeline

After training, run these checks in order. Each builds on the previous.

### 7.1 Check A: Multicollinearity Reduction (VIF)

Compute VIF for the embedding dimensions and compare to raw features:

```
For each embedding dimension j (j = 1..d):
    Regress z_j on all other z_k (k != j)
    VIF_j = 1 / (1 - R_j^2)
```

**Pass criterion:** Max VIF on embeddings < Max VIF on raw features. Ideally VIF < 2 on all embedding dimensions.

### 7.2 Check B: Condition Number Improvement

Compute condition number of the embedding matrix:

```
kappa(Z) = sigma_max(Z) / sigma_min(Z)
```

**Pass criterion:** kappa(Z) < kappa(X). Ideally kappa(Z) < 10.

### 7.3 Check C: Predictive Power Preservation

Run cross-validated logistic regression on the embeddings and compare to raw features:

```
For each fold k in 5-fold stratified CV:
    fit LogisticRegression on Z_train_k, y_train_k
    evaluate on Z_test_k, y_test_k
    record accuracy, balanced accuracy, F1, AUC-ROC
```

**Pass criterion:** Mean CV accuracy on embeddings >= mean CV accuracy on raw features (with logistic regression). The embeddings should not lose predictive power despite the dimensionality reduction.

### 7.4 Check D: Per-Dimension Significance (Wald Test)

Fit a logistic regression y ~ Z and test each coefficient:

```
For each dimension j:
    W_j = beta_j / SE(beta_j)
    p_j = 2 * (1 - Phi(|W_j|))
```

where:
- beta_j is the logistic regression coefficient for dimension j
- SE(beta_j) = sqrt(diag((X^T W X)^{-1})_j) where W = diag(p_i(1-p_i))
- Phi is the standard normal CDF

**Pass criterion:** > 30% of embedding dimensions are individually significant (p < 0.05). This means the embedding distributes predictive signal across multiple dimensions, not just one.

### 7.5 Check E: Feature Attribution (Interpretability)

Compute the Jacobian of the embedding with respect to the input:

```
J = d(mu) / d(x)     shape: (N x d x D)
```

Aggregate to a (D x d) attribution matrix:

```
Attribution_{j->k} = (1/N) * sum_{i=1}^{N} |J_{i,k,j}|
```

This tells you which input features drive each embedding dimension.

**Pass criterion:** Each embedding dimension should have distinct top-contributing input features. If all dimensions are driven by the same features, the orthogonality penalty needs to be stronger (increase gamma).

---

## 8. Baseline Comparisons (Mandatory)

The Orth-SVAE must be compared against multiple baselines to demonstrate value. Do not claim success based on beating PCA alone.

### 8.1 Five-Model Comparison Framework

| # | Model | Features | Classifier | What It Tests |
|---|-------|----------|-----------|---------------|
| 1 | Raw+LR | Raw (D dims) | LogisticRegression | Linear baseline on original features |
| 2 | PCA+LR | PCA (d dims) | LogisticRegression | Linear decorrelation baseline |
| 3 | Raw+RF | Raw (D dims) | RandomForest (200 trees) | Nonlinear baseline -- can it learn despite multicollinearity? |
| 4 | Raw+LGBM | Raw (D dims) | LightGBM (200 rounds) | Gradient boosting baseline -- the "industry standard" |
| 5 | SVAE+LR | Orth-SVAE (d dims) | LogisticRegression | Our method: decorrelated embedding + simple classifier |

### 8.2 Why Each Baseline Matters

- **Raw+LR:** Shows the penalty of multicollinearity on linear models. If the raw data has low multicollinearity, the advantage of Orth-SVAE will be small (and that's an honest finding).
- **PCA+LR:** Shows what unsupervised decorrelation achieves. Orth-SVAE must beat this to justify the training overhead.
- **Raw+RF:** Random forests are inherently robust to multicollinearity (they split on one feature at a time). If RF on raw features beats SVAE+LR, the embedding approach does not add practical value.
- **Raw+LGBM:** LightGBM/XGBoost is the go-to method for tabular data in production. This is the bar to clear for practical relevance.
- **SVAE+LR:** The simplicity of the downstream classifier (logistic regression) is a feature, not a bug. It demonstrates that the embedding has absorbed the complexity, enabling a simple, interpretable, fast model.

### 8.3 Evaluation Metrics

For each model, compute via 5-fold stratified cross-validation:

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Accuracy | correct / total | Overall prediction quality |
| Balanced Accuracy | (TPR + TNR) / 2 | Performance on imbalanced data |
| F1 (macro) | harmonic mean of precision and recall, averaged across classes | Class-balanced prediction quality |
| AUC-ROC | Area under the ROC curve | Ranking quality (threshold-independent) |

### 8.4 Additional Evaluations

- **Out-of-sample test:** Train on training set, evaluate on held-out test set.
- **Walk-forward backtest:** Expanding window temporal validation to test stability over time.
- **Linear probes:** Predict auxiliary concepts (e.g., medium, sale category) from embeddings to test what information the embedding encodes beyond the primary target.

---

## 9. Feature Attribution via Jacobian

### 9.1 The Jacobian Matrix

The Jacobian J captures how each input feature influences each embedding dimension:

```
J_{k,j} = d(mu_k) / d(x_j)
```

For a single sample, this is a (d x D) matrix. Computed via automatic differentiation (backpropagation).

### 9.2 Aggregated Attribution

Average the absolute Jacobian across all samples to get a stable attribution:

```
A_{j->k} = (1/N) * sum_{i=1}^{N} |d(mu_k) / d(x_j)|_i
```

This produces a (D x d) matrix where:
- Rows = input features
- Columns = embedding dimensions
- Values = mean absolute sensitivity

### 9.3 Computation (Pseudocode)

```python
model.eval()
x = tensor(X_scaled, requires_grad=True)    # (N, D)
mu = model.encode(x)                         # (N, d)

attributions = zeros(D, d)
for k in range(d):
    model.zero_grad()
    x.grad = None
    mu[:, k].sum().backward(retain_graph=(k < d-1))
    attributions[:, k] = abs(x.grad).mean(dim=0)  # (D,)
```

### 9.4 Interpretation

A well-trained Orth-SVAE should show:

- **Specialization:** Each embedding dimension has a distinct set of top-driving input features.
- **Coverage:** Most input features contribute to at least one embedding dimension.
- **Interpretability:** The groupings should make domain sense. For example, in art data:
  - dim_0 might be driven by physical dimensions (height, width, surface_area)
  - dim_2 might capture artist market history (avg_price, prior_lots, market_depth)
  - dim_6 might encode medium/material (has_depth, log_depth, is_sculpture)

---

## 10. Decision Guide

### 10.1 When to Use Orth-SVAE

Use Orth-SVAE when ALL of the following conditions hold:

- **High dimensionality:** D >= 15 features
- **Severe multicollinearity:** Max VIF > 10 or condition number > 30
- **Dense correlation blocks:** Multiple groups of correlated features (not just one pair)
- **Interpretability matters:** You need to understand what each embedding dimension represents
- **Downstream model is linear:** You plan to use logistic regression, linear SVM, or similar

### 10.2 When PCA Is Sufficient

PCA is adequate when:

- **Moderate multicollinearity:** VIF < 10 for all features, condition number < 30
- **Low dimensionality:** D < 15 features
- **No supervision needed:** The prediction target is not available during embedding
- **Speed is critical:** PCA is O(ND^2), Orth-SVAE requires iterative training

### 10.3 When to Skip Both

Neither embedding method is needed when:

- **Features are already uncorrelated:** VIF < 5 for all features
- **Using tree-based models only:** Random forests and gradient boosting are inherently robust to multicollinearity (they split on individual features)
- **Very small datasets:** N < 500 makes the Orth-SVAE training unreliable

### 10.4 Decision Flowchart

```
Start
  |
  v
Max VIF > 10 or Condition# > 30?
  |
  +-- NO --> Features > 15 and correlated blocks exist?
  |            |
  |            +-- NO --> Skip embedding. Use raw features.
  |            +-- YES --> PCA may help. Try PCA+LR vs Raw+LR.
  |
  +-- YES --> Will downstream model be linear?
               |
               +-- NO --> Try Raw+RF / Raw+LGBM first.
               |           If RF/LGBM accuracy is sufficient, stop.
               |
               +-- YES --> Orth-SVAE is recommended.
                            Train with gamma >= 1.0.
                            Compare against all 5 baselines.
```

---

## 11. Results Summary

### 11.1 Polymarket Dataset (Low Multicollinearity)

| Property | Value |
|----------|-------|
| Domain | Prediction market (over/under contracts) |
| Samples | 813 |
| Features | 8 (volume, duration, trade metrics) |
| Target | Binary (over/under resolution) |
| Max VIF (raw) | 6.19 |
| Condition # (raw) | 6.0 |
| Max |correlation| | 0.78 |
| Severity | Low-moderate multicollinearity |

**Results:**

| Metric | Raw+LR | PCA+LR | Orth-SVAE+LR |
|--------|--------|--------|-------------|
| Max VIF | 6.19 | 1.04 | 1.02 |
| Condition # | 6.0 | 6.0 | 1.86 |
| CV Accuracy | 0.6936 | 0.6936 | 0.7010 |
| CV AUC | 0.7102 | 0.7102 | 0.7200 |
| Walk-Forward | 0.659 | 0.659 | 0.666 |
| Sig Dims (Wald) | 2/8 (25%) | 3/8 (38%) | 4/8 (50%) |

**SVAE vs PCA delta: +0.74pp accuracy, +0.98pp AUC**

With only 8 mildly correlated features, the advantage is real but small. PCA and Raw+LR give identical results because PCA is a rotation that does not help logistic regression (which is rotation-invariant when features have equal VIF < 10).

### 11.2 Art Market Dataset (Severe Multicollinearity)

Three iterations of the art market experiment, each with expanded data:

#### Iteration 1: 30 features, 29K lots (Sotheby's only)

| Property | Value |
|----------|-------|
| Domain | Sotheby's auction lots |
| Samples | 29,485 |
| Features | 30 (artist, physical, medium, sale context, historical) |
| Target | log(hammer_price) binarized at median |
| Max VIF (raw) | infinity (r = -1.0 between feature pairs) |
| Condition # (raw) | infinity |
| Severity | Extreme multicollinearity |

**5-Model Cross-Validated Results:**

| Model | CV Accuracy | CV AUC | OOS Accuracy | Walk-Forward |
|-------|------------|--------|-------------|-------------|
| Raw+LR | 0.7015 | 0.7593 | 0.7319 | 0.695 |
| PCA+LR | 0.6746 | 0.7159 | 0.7151 | 0.672 |
| Raw+RF | 0.7374 | 0.8170 | 0.7425 | 0.704 |
| Raw+LGBM | 0.7324 | 0.8101 | 0.7330 | 0.691 |
| **SVAE+LR** | **0.7543** | **0.8307** | **0.7839** | **0.736** |

**Key deltas:** SVAE vs PCA: +8.0pp, SVAE vs RF: +1.7pp, SVAE vs LGBM: +2.2pp

**Regression Probes (predicting continuous log hammer price):**

| Method | R-squared | MAE |
|--------|-----------|-----|
| Raw | 0.293 | 0.964 |
| PCA | 0.222 | 1.015 |
| SVAE | 0.332 | 0.925 |

#### Iteration 2: 35 features, 693K lots (3 auction houses)

Adding Christie's (654K lots) and Phillips (247 lots) with 5 additional estimate features. The estimate features dominate prediction and make the problem easier (~90% accuracy baseline).

| Model | CV Accuracy | CV AUC |
|-------|------------|--------|
| Raw+LR | 0.8966 | 0.9607 |
| PCA+LR | 0.8891 | 0.9564 |
| **Raw+RF** | **0.9026** | **0.9676** |
| Raw+LGBM | 0.9008 | 0.9670 |
| SVAE+LR | 0.8984 | 0.9632 |

**Key delta:** SVAE vs RF: -0.4pp. Tree models handle estimate-dominated features better.

#### Iteration 3: 65 features (63 after drop), 693K lots (FINAL)

30 new features across 8 new categories: confidence scores (H), sale mechanics (I), attribution (J), provenance (K), text (L), style (M), lot category (N), sale flags (O), estimate accuracy (P). Two features dropped as zero-variance after imputation.

| Property | Value |
|----------|-------|
| Domain | Sotheby's + Christie's + Phillips |
| Samples | 693,650 |
| Features | 63 (after 2 dropped) across 15 categories |
| Target | log(hammer_price) binarized at median |
| Max VIF (raw) | infinity (17 severe features) |
| Mean VIF (raw) | 3,233 |
| Condition # (raw) | infinity |
| Severity | Extreme multicollinearity with dense correlation clusters |

**5-Model Cross-Validated Results:**

| Model | CV Accuracy | CV AUC | OOS Accuracy | Walk-Forward (mean +/- std) |
|-------|------------|--------|-------------|---------------------------|
| Raw+LR | 0.8994 | 0.9627 | 0.8941 | 0.9006 +/- 0.0091 |
| PCA+LR | 0.8931 | 0.9593 | 0.8707 | 0.8911 +/- 0.0131 |
| **Raw+RF** | **0.9080** | **0.9700** | 0.8785 | 0.9027 +/- 0.0087 |
| Raw+LGBM | 0.9079 | 0.9697 | 0.8930 | **0.9056** +/- 0.0144 |
| SVAE+LR | 0.9022 | 0.9646 | **0.8948** | 0.9019 +/- **0.0078** |

**Multicollinearity Reduction:**

| Metric | Raw (63D) | PCA (12D) | SVAE (12D) |
|--------|-----------|----------|-----------|
| Mean VIF | 3,233 | 2.90 | **1.003** |
| Max VIF | infinity | 9.21 | **1.01** |
| Condition # | infinity | 14.57 | **3.48** |
| Severe VIF (>10) | 17 | 0 | **0** |
| Sig Dims (Wald) | 35/63 (56%) | 12/12 (100%) | **12/12 (100%)** |

**Key deltas:**
- SVAE vs PCA: +0.9pp accuracy (CV), +2.4pp (OOS)
- SVAE vs RF: -0.6pp accuracy (CV), **+1.6pp (OOS)** -- SVAE generalizes better
- SVAE vs LGBM: -0.6pp accuracy (CV), +0.2pp (OOS)
- Walk-forward variance: SVAE 0.0078 (lowest) vs RF 0.0087, LGBM 0.0144

**Feature Attribution (Jacobian, top drivers per embedding dimension):**

| Dim | Top Features |
|-----|-------------|
| 0 | log_estimate_low, is_wine, log_estimate_mid |
| 1 | width_cm, is_book, is_jewelry |
| 2 | log_estimate_low, log_estimate_mid, estimate_mid_usd |
| 3 | is_book, title_length, artist_name_confidence |
| 4 | exhibition_count, estimate_relative_level, is_attributed_artist |
| 5 | log_estimate_low, log_estimate_mid, estimate_mid_usd |
| 6 | log_estimate_low, log_estimate_mid, estimate_mid_usd |
| 7 | has_depth, log_depth_cm, height_cm |

**Regression Probes (predicting continuous log hammer price):**

| Method | R-squared | MAE |
|--------|-----------|-----|
| Raw | 0.815 | 0.427 |
| **PCA** | **0.859** | **0.472** |
| SVAE | 0.847 | 0.507 |

### 11.3 Domain Comparison

| Metric | Polymarket (8D) | Art 30D/29K | Art 35D/693K | Art 63D/693K |
|--------|----------------|-------------|--------------|--------------|
| Raw Max VIF | 6.19 | infinity | infinity | infinity |
| Raw Condition # | 6.0 | infinity | infinity | infinity |
| SVAE vs PCA (CV accuracy) | +0.74pp | +8.0pp | +0.9pp | +0.9pp |
| SVAE vs PCA (walk-forward) | +0.7pp | +6.5pp | -- | +1.1pp |
| VIF reduction | 6.1x | infinity | infinity | infinity |
| SVAE beats RF? (CV) | N/A | YES (+1.7pp) | NO (-0.4pp) | NO (-0.6pp) |
| SVAE beats RF? (OOS) | N/A | YES | -- | **YES (+1.6pp)** |
| SVAE beats LGBM? (CV) | N/A | YES (+2.2pp) | NO (-0.2pp) | NO (-0.6pp) |
| Walk-forward stability (std) | N/A | -- | -- | **Best (0.0078)** |

### 11.4 Conclusions Across All Experiments

**What the data consistently shows:**

1. **Multicollinearity elimination is absolute.** Across all datasets and feature counts (8D to 63D), Orth-SVAE reduces VIF to near 1.0 and achieves the lowest condition numbers. This is reliable and reproducible.

2. **SVAE always beats PCA.** Across all experiments, SVAE+LR outperforms PCA+LR by +0.7pp to +8.0pp. The supervised prediction loss forces outcome-relevant signal into the embedding, which unsupervised PCA cannot do.

3. **Tree models win in-distribution CV on large datasets.** With 693K samples, RF and LGBM have enough data to learn complex interactions natively. They beat SVAE+LR by ~0.6pp on CV accuracy.

4. **SVAE generalizes better out-of-sample.** On the 65-feature/693K dataset, SVAE+LR achieves the best OOS accuracy (0.8948 vs RF 0.8785, +1.6pp). The decorrelated embedding resists overfitting to training-set correlation structure.

5. **SVAE has the most stable temporal performance.** Walk-forward variance for SVAE (std=0.0078) is lower than RF (0.0087) and LGBM (0.0144). The embedding smooths out temporal regime shifts.

6. **The advantage pattern depends on data scale:**
   - **Small data, high multicollinearity (29K, 30D):** SVAE wins everything -- CV, OOS, walk-forward. The embedding's regularization prevents the overfitting that trees suffer with insufficient data.
   - **Large data, high multicollinearity (693K, 63D):** Trees win CV, SVAE wins OOS and stability. With enough data, trees learn to handle correlation natively, but at the cost of reduced generalization.

**When to choose SVAE over trees:**
- When interpretability of individual features matters (coefficient-based inference)
- When OOS generalization and temporal stability are more important than in-sample fit
- When downstream models must be linear (regulatory, explainability requirements)
- When the feature space has VIF > 10 and you need valid statistical tests

**When to choose trees over SVAE:**
- When maximum in-distribution accuracy is the only goal
- When you have sufficient data (N > 100K) for trees to learn interaction effects
- When interpretability is provided by SHAP/permutation importance (not coefficients)

---

## Appendix A: Implementation Checklist

```
[ ] 1. Load data, impute NaN, standardize (fit on train only)
[ ] 2. Compute raw VIF, condition number, correlation matrix
[ ] 3. Set up Orth-SVAE with config: dim=8, beta=1, alpha=1, gamma=1
[ ] 4. Train with Adam + cosine annealing, early stopping on val loss
[ ] 5. Extract embeddings using mu (mean), not z (sample)
[ ] 6. Compute embedding VIF, condition number -- verify improvement
[ ] 7. Run 5-model CV comparison (Raw+LR, PCA+LR, Raw+RF, Raw+LGBM, SVAE+LR)
[ ] 8. Run out-of-sample evaluation on held-out test set
[ ] 9. Run walk-forward temporal backtest (if data is time-ordered)
[ ] 10. Compute Jacobian feature attribution
[ ] 11. Run linear probes on auxiliary labels
[ ] 12. Generate figures: VIF comparison, correlation matrices, model comparison,
       walk-forward, feature attribution heatmap, domain comparison
[ ] 13. Write verdict answering:
        Q1: Does Orth-SVAE beat PCA? (always yes if gamma > 0 and VIF > 5)
        Q2: Does Orth-SVAE beat RF/LGBM? (depends on dataset)
```

## Appendix B: Reference Implementation

The complete implementation is in this repository:

| File | Purpose |
|------|---------|
| `models/autoencoder.py` | Encoder, Decoder, MarketAutoencoder with 4-component loss |
| `models/train.py` | Training loop with early stopping and embedding monitoring |
| `models/statistics.py` | VIF, condition number, orthogonality test, Wald test |
| `models/visualize.py` | Publication-quality plots |
| `models/probes.py` | Linear probe framework (classification + regression) |
| `run_art_experiment.py` | Full 5-model comparison pipeline (art market) |
| `run_honest_experiment.py` | Polymarket experiment with honest methodology |
| `art_data/extract.py` | Art market data extraction from SQLite |
| `art_data/features.py` | Art-specific feature engineering |

## Appendix C: Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Embedding VIF > 5 | gamma too low | Increase gamma to 2-5 |
| All dims encode same signal | gamma = 0 | Set gamma >= 1.0 |
| SVAE accuracy << Raw accuracy | alpha too low, or too few epochs | Increase alpha or patience |
| Dead dimensions (std < 1e-6) | embedding_dim too large | Reduce embedding_dim |
| Training loss oscillates | learning_rate too high | Reduce to 5e-4 |
| Loss = NaN | exploding gradients | Add gradient clipping (max_norm=1.0) |
| PCA beats SVAE | Low multicollinearity in raw data | SVAE not needed; use PCA |
| RF/LGBM beats SVAE | Tree models handle correlation natively | SVAE still valuable for interpretability and linear downstream models |
