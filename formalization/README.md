# Causal-Embedding Framework for Prediction Markets

A Lean 4 formalization of the causal-embedding framework for prediction market outcome spaces. This document walks through the mathematical construction step by step, mirroring the whiteboard derivation.

## Overview

The framework answers: **given a prediction market with open, won, and sold positions, how do we embed outcomes into a vector space where causal dynamics can be learned and evaluated?**

The construction proceeds in five layers:

1. **Abstract outcome space** — classify market positions
2. **Partition** — separate resolved from unresolved
3. **Structural bijection** — map open positions to their resolved counterparts
4. **Embedding** — inject the resolved space into ℝⁿ
5. **Causal map** — learn dynamics in the embedding space with a commutative guarantee

---

## 1. Sample Space and Outcome Classification

Let **Ω** be a sample space of market outcomes (positions, contracts, or bets).

Define the **outcome classifier** ω : Ω → {Win, Open, Sold}:

| Value | Symbol | Meaning |
|-------|--------|---------|
| 1     | Win    | Position resolved in profit |
| 0     | Open   | Position still unresolved |
| -1    | Sold   | Position exited / sold / short |

In the formalization, this is an inductive type `Outcome` and a structure `MarketSpace` bundling Ω with ω.

## 2. Partition into W, O, S

The classifier induces a three-way partition of Ω:

```
W ≜ { x ∈ Ω | ω(x) = Win  }    — winning outcomes
O ≜ { x ∈ Ω | ω(x) = Open }    — open/unresolved outcomes
S ≜ { x ∈ Ω | ω(x) = Sold }    — sold/short outcomes
```

**Proven properties:**
- **Exhaustive**: W ∪ O ∪ S = Ω
- **Pairwise disjoint**: W ∩ O = ∅, W ∩ S = ∅, O ∩ S = ∅

These are the theorems `partition_exhaustive`, `W_O_disjoint`, `W_S_disjoint`, `O_S_disjoint`.

## 3. Resolved Space Π

Define the **resolved space** as the complement of open outcomes:

```
Π ≜ Ω \ O = W ∪ S
```

Π contains every outcome whose fate is known — it either won or was exited. The theorem `Pi_eq_compl_O` proves Π = Oᶜ.

## 4. Structural Bijection St

Not all open positions are structurally distinct from resolved ones. Let **Ωc ⊆ O** be the "closable" subset — open outcomes that have a resolved counterpart.

Define **St : Ωc → Π**, a bijection mapping each closable open outcome to the resolved outcome it would become. This is the key bridge between the unresolved present and the resolved past: it lets us treat open positions as if they were already resolved, for the purpose of embedding and prediction.

In the formalization, `StructuralBijection` bundles:
- `Ωc` with proof that Ωc ⊆ O
- `St` as a `≃` (Lean equiv / bijection)
- `target` with proof that target ⊆ Π

## 5. Embedding ψ : Π → ℝⁿ

Embed the resolved space into n-dimensional real space:

```
ψ : Π → ℝⁿ    (injective)
```

This gives each resolved outcome a numeric representation. The **embedded resolved set** is:

```
X ≜ ψ(Π) = { x ∈ ℝⁿ | ψ⁻¹(x) ∈ Π }
```

In the formalization, `CausalEmbedding` bundles ψ with a proof of injectivity (`ψ_inj`).

## 6. Causal Map f : ℝⁿ → ℝⁿ

The **causal map** operates in the embedding space:

```
f : ℝⁿ → ℝⁿ
Y ≜ f(X)    — predicted outcomes
```

Given an embedded state x ∈ X, f(x) is the model's prediction of the next state (or final outcome).

## 7. Squared-Error Loss

The loss function evaluates prediction quality:

```
ℓ(y, ŷ) ≜ ‖y - ŷ‖² = Σᵢ (yᵢ - ŷᵢ)²
```

**Proven properties:**
- **Non-negative**: ℓ(y, ŷ) ≥ 0 (`sqLoss_nonneg`)
- **Faithful**: ℓ(y, ŷ) = 0 ⟺ y = ŷ (`sqLoss_eq_zero_iff`)

## 8. Commutative Diagram

The central requirement: **the embedding must commute with the causal dynamics**.

```
    Π ———g———→ Π
    |            |
  ψ |            | ψ
    ↓            ↓
   ℝⁿ ———f———→ ℝⁿ
```

For some abstract dynamics g : Π → Π, we require:

```
∀ x ∈ Π,  f(ψ(x)) = ψ(g(x))
```

This says: it doesn't matter whether you first embed then apply f, or first apply g then embed — you get the same result. This is the `commutes` field of `CommutativeCausalEmbedding`.

**Why this matters:** If commutativity holds, the learned dynamics f in ℝⁿ faithfully represent the true dynamics g on the abstract outcome space. Predictions made in the embedding space correspond to real outcome transitions.

## 9. Homomorphism Property

For independent markets, the embedding should preserve product structure:

```
φ(x ⊗ y) = φ(x) · φ(y)
```

This ensures that the joint embedding of two independent market outcomes equals the product of their individual embeddings. In the formalization, `MarketHomomorphism` captures this as a standard multiplicative homomorphism.

## 10. Convergence and Continuity

The bottom of the board records the standard analytic foundations:

**Sequence convergence:**
```
∀ε > 0, ∃ N ∈ ℕ, ∀n ≥ N, |aₙ - L| < ε
```

**ε-δ continuity:**
```
∀ε > 0, ∃ δ > 0, |x - x'| < δ → |f(x) - f(x')| < ε
```

These apply to the embedding ψ and causal map f — both should be continuous for the framework to be well-behaved. The formalization includes a proof that convergent sequences are bounded (`SeqConvergesTo.bounded`).

## 11. Gradient Decomposition

The core calculus result. Given the composite prediction pipeline:

```
x ∈ ℝᵐ  ──ψ──▸  z ∈ ℝⁿ  ──f──▸  ŷ ∈ ℝⁿ
```

The loss is L(x) = ‖f(ψ(x)) - y‖². By the chain rule, the gradient decomposes into **three factors**:

```
∇ₓL  =  2  ·  Jψ(x)ᵀ  ·  Jf(z)ᵀ  ·  r
         ↑        ↑           ↑        ↑
       scalar  embedding    causal   residual
               Jacobian    Jacobian
```

where z = ψ(x) and r = f(ψ(x)) - y.

### Three terms and their interpretations

| Term | Formula | Interpretation |
|------|---------|----------------|
| **Residual** r | f(ψ(x)) - y | "How wrong is the prediction?" — a vector in ℝⁿ pointing from the target to the prediction |
| **Causal backprop** Jf(z)ᵀ · r | ∂f/∂z transposed times residual | "How does the error propagate through the causal map?" — transforms the error signal from output space back through f's local linearization |
| **Embedding backprop** Jψ(x)ᵀ · (Jf(z)ᵀ · r) | ∂ψ/∂x transposed times causal backprop | "How does the error reach the abstract input space?" — maps the causal error signal back through the embedding |

### Expanded index form

Component j of the gradient expands to a double sum over the internal dimensions:

```
∂L/∂xⱼ = 2 · Σₖ Jψ(x)ₖⱼ · Σᵢ Jf(z)ᵢₖ · rᵢ
```

This is `grad_decomposition` in the formalization.

### Per-component attribution

We can further decompose by asking: "how much does prediction error in output dimension i contribute to the gradient at input dimension j?"

```
Aᵢⱼ = 2 · rᵢ · Σₖ Jf(z)ᵢₖ · Jψ(x)ₖⱼ
```

The gradient is the sum of these attributions: ∂L/∂xⱼ = Σᵢ Aᵢⱼ

This is proven as `grad_eq_sum_attributions`.

### Composite Jacobian form

Equivalently, using the chain-rule Jacobian J(f∘ψ) = Jf · Jψ:

```
∇ₓL = 2 · J(f∘ψ)(x)ᵀ · r
```

This collapses the two-Jacobian product into a single composite Jacobian, proven as `grad_via_composite_jacobian`.

### Key proven properties

| Theorem | Statement |
|---------|-----------|
| `grad_zero_of_perfect_prediction` | ∇ₓL = 0 when f(ψ(x)) = y |
| `residual_norm_eq_loss` | ‖r‖² = L(x) |
| `attribution_zero_of_component_perfect` | Aᵢⱼ = 0 when prediction dimension i is exact |
| `grad_eq_sum_attributions` | ∇ₓL(j) = Σᵢ Aᵢⱼ |
| `grad_via_composite_jacobian` | Equivalence of factored and composite forms |

### Practical meaning for prediction markets

Each factor tells you something different about why the model is updating:

- **Large ‖r‖**: the model is badly wrong about some market — gradient is large regardless of Jacobian structure
- **Large ‖Jfᵀ · r‖ but small ‖r‖**: the causal map amplifies small errors — sensitive dynamics, potential instability
- **Large ‖Jψᵀ · (Jfᵀ · r)‖ but small ‖Jfᵀ · r‖**: the embedding compresses causal error into specific input directions — reveals which abstract features (price, volume, OBI, etc.) drive the update

---

## Connection to Polymarket

This framework maps directly onto prediction market infrastructure:

| Math | Polymarket |
|------|------------|
| Ω | All positions across all markets |
| ω(x) = Win | Resolved in profit (correct prediction) |
| ω(x) = Open | Active, unresolved position |
| ω(x) = Sold | Exited position (sold shares) |
| ψ | Market embedding (autoencoder, transformer, or GNN encoder) |
| f | Predictive model (GNN-TCN, signal compositor) |
| ℓ | Training loss for the predictive model |
| Commutativity | Model predictions correspond to real outcome transitions |
| Homomorphism | Independent markets decompose cleanly |

## File Structure

```
formalization/
├── MarketOutcomeSpace.lean   — Full Lean 4 formalization (Mathlib dependency)
└── README.md                 — This document
```

## Building

Requires Lean 4 with Mathlib. Add to `lakefile.lean`:

```lean
require mathlib from git
  "https://github.com/leanprover-community/mathlib4"
```

Then:

```bash
lake build
```
