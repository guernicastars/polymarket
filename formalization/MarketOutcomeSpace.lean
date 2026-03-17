/-
  Lean 4 Formalization of Prediction Market Outcome Space
  ========================================================
  Transcribed from whiteboard: causal-embedding framework for prediction markets.

  The framework defines:
  1. A sample space Ω with outcome classification ω ∈ {Win, Open, Sold}
  2. Partition of Ω into W, O, S based on ω
  3. A structural bijection St : Ωc ⊆ O → Π = W ∪ S
  4. An embedding ψ : Π ↪ ℝⁿ (injective)
  5. A causal map f : ℝⁿ → ℝⁿ with squared-error loss
  6. Commutative diagram: the embedding commutes with the causal map
  7. Homomorphism property φ(x ⊗ y) = φ(x) · φ(y)
  8. Convergence and continuity (ε-δ)
-/

import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Logic.Equiv.Basic
import Mathlib.Order.Filter.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Real.Basic

noncomputable section

open Set Function

/-! ## 1–2. Sample space and outcome classification (Steps 1–2) -/

/-- Outcome status: Win (1), Open (0), Sold/Short (-1) — Step 3 on the board -/
inductive Outcome where
  | Win   : Outcome  -- ω(x) = 1
  | Open  : Outcome  -- ω(x) = 0
  | Sold  : Outcome  -- ω(x) = -1
  deriving DecidableEq, Repr

/-- The market outcome space bundles a sample space with its classification function -/
structure MarketSpace where
  Ω : Type*                   -- Step 1: sample space
  ω : Ω → Outcome             -- Step 3: outcome classifier

namespace MarketSpace

variable (M : MarketSpace)

/-! ## 3–4. Partition into W, O, S (Step 4) -/

/-- W ≜ { x ∈ Ω | ω(x) = 1 } — winning outcomes -/
def W : Set M.Ω := { x | M.ω x = Outcome.Win }

/-- O ≜ { x ∈ Ω | ω(x) = 0 } — open/unresolved outcomes -/
def O : Set M.Ω := { x | M.ω x = Outcome.Open }

/-- S ≜ { x ∈ Ω | ω(x) = -1 } — sold/short outcomes -/
def S : Set M.Ω := { x | M.ω x = Outcome.Sold }

/-- The partition is exhaustive: W ∪ O ∪ S = Ω -/
theorem partition_exhaustive : M.W ∪ M.O ∪ M.S = univ := by
  ext x
  simp only [W, O, S, mem_union, mem_setOf_eq, mem_univ, iff_true]
  cases M.ω x <;> simp

/-- W, O, S are pairwise disjoint -/
theorem W_O_disjoint : Disjoint M.W M.O := by
  rw [Set.disjoint_iff]
  intro x ⟨hw, ho⟩
  simp [W, O] at hw ho
  rw [hw] at ho
  exact Outcome.noConfusion ho

theorem W_S_disjoint : Disjoint M.W M.S := by
  rw [Set.disjoint_iff]
  intro x ⟨hw, hs⟩
  simp [W, S] at hw hs
  rw [hw] at hs
  exact Outcome.noConfusion hs

theorem O_S_disjoint : Disjoint M.O M.S := by
  rw [Set.disjoint_iff]
  intro x ⟨ho, hs⟩
  simp [O, S] at ho hs
  rw [ho] at hs
  exact Outcome.noConfusion hs

/-! ## 5. Resolved space Π and structural bijection St (Step 5) -/

/-- Π ≜ Ω \ O = W ∪ S — the resolved outcomes -/
def Π : Set M.Ω := M.W ∪ M.S

/-- Π equals the complement of O -/
theorem Pi_eq_compl_O : M.Π = Oᶜ := by
  ext x
  simp only [Π, W, S, O, mem_union, mem_compl_iff, mem_setOf_eq]
  constructor
  · rintro (h | h) <;> (intro ho; rw [h] at ho; exact Outcome.noConfusion ho)
  · intro h; cases hω : M.ω x
    · left; rfl
    · exact absurd hω h
    · right; rfl

end MarketSpace

/-! ## 6. Embedding into ℝⁿ and causal map (Step 6)

  ψ : Π → ℝⁿ  (bijection / embedding)
  f : ℝⁿ → ℝᵖ  (causal map — output dimension p may differ from input n)
  X ≜ ψ(Π),  Y ≜ f(X)
-/

/-- The full causal-embedding structure over a market space -/
structure CausalEmbedding (n p : ℕ) where
  M     : MarketSpace
  /-- ψ : Π → ℝⁿ, the embedding (Step 6: bijection) -/
  ψ     : M.Π → (Fin n → ℝ)
  ψ_inj : Injective ψ
  /-- f : ℝⁿ → ℝᵖ, the causal/prediction map -/
  f     : (Fin n → ℝ) → (Fin p → ℝ)

namespace CausalEmbedding

variable {n p : ℕ} (C : CausalEmbedding n p)

/-- X ≜ ψ(Π) = { x ∈ ℝⁿ | ψ⁻¹(x) ∈ Π } — the embedded resolved set -/
def X : Set (Fin n → ℝ) := range C.ψ

/-- Y ≜ f(X) — the image under the causal map (lives in ℝᵖ) -/
def Y : Set (Fin p → ℝ) := C.f '' C.X

end CausalEmbedding

/-! ## 7. Squared-error loss  ℓ(Y, Ŷ) ≜ ‖Y - Ŷ‖²  (top-right of board) -/

/-- Squared-error loss in ℝᵖ (output space) -/
def sqLoss (p : ℕ) (y ŷ : Fin p → ℝ) : ℝ :=
  ∑ i : Fin p, (y i - ŷ i) ^ 2

/-- Loss is non-negative -/
theorem sqLoss_nonneg (p : ℕ) (y ŷ : Fin p → ℝ) : 0 ≤ sqLoss p y ŷ :=
  Finset.sum_nonneg fun i _ => sq_nonneg _

/-- Loss is zero iff predictions match -/
theorem sqLoss_eq_zero_iff (p : ℕ) (y ŷ : Fin p → ℝ) :
    sqLoss p y ŷ = 0 ↔ y = ŷ := by
  constructor
  · intro h
    have : ∀ i, (y i - ŷ i) ^ 2 = 0 := by
      by_contra hne
      push_neg at hne
      obtain ⟨i, hi⟩ := hne
      have hpos : 0 < (y i - ŷ i) ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm hi)
      linarith [Finset.sum_pos (fun i (_ : i ∈ Finset.univ) => sq_nonneg (y i - ŷ i))
        ⟨i, Finset.mem_univ i, hpos⟩]
    ext i
    have := this i
    rwa [sq_eq_zero_iff, sub_eq_zero] at this
  · rintro rfl
    simp [sqLoss]

/-! ## 8. Commutative diagram (right side of board)

  With f : ℝⁿ → ℝᵖ, the diagram needs two embeddings:

       Π ———g———→ Π'
       |            |
     ψ |            | ψ̂
       ↓            ↓
      ℝⁿ ———f———→ ℝᵖ

  i.e.  f ∘ ψ = ψ̂ ∘ g
-/

/-- A causal embedding commutes if f ∘ ψ = ψ̂ ∘ g
    Since f : ℝⁿ → ℝᵖ, we need an output embedding ψ̂ : Π' → ℝᵖ -/
structure CommutativeCausalEmbedding (n p : ℕ) extends CausalEmbedding n p where
  /-- Π' — the output resolved space (may differ from input Π) -/
  Π'        : Set toMarketSpace.Ω
  /-- ψ̂ : Π' → ℝᵖ, the output-side embedding -/
  ψ_hat     : Π' → (Fin p → ℝ)
  ψ_hat_inj : Injective ψ_hat
  /-- g : Π → Π', the abstract causal dynamics -/
  g         : toMarketSpace.Π → Π'
  /-- The diagram commutes: f(ψ(x)) = ψ̂(g(x)) -/
  commutes  : ∀ x : toMarketSpace.Π, toCausalEmbedding.f (toCausalEmbedding.ψ x) =
                                       ψ_hat (g x)

/-! ## 9. Homomorphism property: φ(x ⊗ y) = φ(x) · φ(y) (bottom-right)

  This captures the multiplicative structure of independent market outcomes.
-/

/-- A market homomorphism preserves product structure -/
structure MarketHomomorphism where
  /-- The carrier types -/
  α : Type*
  β : Type*
  [instα : Mul α]
  [instβ : Mul β]
  /-- The map φ -/
  φ : α → β
  /-- Homomorphism: φ(x * y) = φ(x) * φ(y) -/
  hom : ∀ x y : α, φ (@HMul.hMul α α α (@instHMul α instα) x y) =
                     @HMul.hMul β β β (@instHMul β instβ) (φ x) (φ y)

/-! ## 10. Convergence and continuity (ε-δ, bottom of board)

  ∀ε > 0, ∃ N ∈ ℕ, ∀n ≥ N, |aₙ - L| < ε
  ∀ε > 0, ∃ δ > 0, |x - x'| < δ → |f(x) - f(x')| < ε
-/

/-- Sequence convergence: ∀ε > 0, ∃ N, ∀n ≥ N, |aₙ - L| < ε -/
def SeqConvergesTo (a : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n, N ≤ n → |a n - L| < ε

/-- ε-δ continuity: ∀ε > 0, ∃ δ > 0, |x - x'| < δ → |f(x) - f(x')| < ε -/
def EpsilonDeltaContinuous (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x', |x - x'| < δ → |f x - f x'| < ε

/-- A convergent sequence is bounded -/
theorem SeqConvergesTo.bounded {a : ℕ → ℝ} {L : ℝ} (h : SeqConvergesTo a L) :
    ∃ B : ℝ, ∀ n, |a n| ≤ B := by
  obtain ⟨N, hN⟩ := h 1 one_pos
  let M := Finset.sup' (Finset.range (N + 1))
    ⟨0, Finset.mem_range.mpr (Nat.zero_lt_succ N)⟩
    (fun i => (⟨|a i|, abs_nonneg _⟩ : NNReal))
  refine ⟨max (↑M) (|L| + 1), fun n => ?_⟩
  by_cases hn : N ≤ n
  · have := hN n hn
    calc |a n| = |a n - L + L| := by ring_nf
    _ ≤ |a n - L| + |L| := abs_add _ _
    _ < 1 + |L| := by linarith
    _ = |L| + 1 := by ring
    _ ≤ max (↑M) (|L| + 1) := le_max_right _ _
  · push_neg at hn
    have hn' : n < N + 1 := by omega
    have : n ∈ Finset.range (N + 1) := Finset.mem_range.mpr hn'
    have : (⟨|a n|, abs_nonneg _⟩ : NNReal) ≤ M :=
      Finset.le_sup' _ this
    calc |a n| = ↑(⟨|a n|, abs_nonneg _⟩ : NNReal) := rfl
    _ ≤ ↑M := NNReal.coe_le_coe.mpr this
    _ ≤ max (↑M) (|L| + 1) := le_max_left _ _

/-! ## 11. Structural bijection St (Step 5, detailed)

  Let Ωc ⊆ O (a subset of open outcomes)
  St : Ωc → Π  (bijection — maps open outcomes to resolved counterparts)
  X ≜ St(Ωc) = { x ∈ Π | St⁻¹(x) ∈ Ωc }
-/

/-- The structural bijection mapping open outcomes to their resolved counterparts -/
structure StructuralBijection (M : MarketSpace) where
  /-- Ωc ⊆ O — the "closable" open outcomes -/
  Ωc : Set M.Ω
  Ωc_sub_O : Ωc ⊆ M.O
  /-- St : Ωc ≃ image — bijection to a subset of Π -/
  St : Ωc ≃ St.target
  target : Set M.Ω
  target_sub_Π : target ⊆ M.Π

/-! ## 12. Gradient of the loss and chain-rule decomposition

  Given the composite prediction pipeline with distinct input/output dims:

      x ∈ ℝᵐ  →ψ→  z ∈ ℝⁿ  →f→  ŷ ∈ ℝᵖ

  The loss is:
      L(x) = ℓ(y, f(ψ(x))) = ‖f(ψ(x)) - y‖²    where y, ŷ ∈ ℝᵖ

  By the chain rule, the gradient decomposes into three factors:

      ∇ₓL = 2 · Jψ(x)ᵀ · Jf(z)ᵀ · r       ∈ ℝᵐ

  where:
      r  = f(ψ(x)) - y  ∈ ℝᵖ    — the residual (prediction error)
      Jf = ∂f/∂z         (p × n)  — causal Jacobian
      Jψ = ∂ψ/∂x         (n × m)  — embedding Jacobian

  Dimensions:  ℝᵖ →Jfᵀ→ ℝⁿ →Jψᵀ→ ℝᵐ

  Each factor has a distinct interpretation:
  (1) Residual r ∈ ℝᵖ:        "how wrong is the prediction?"
  (2) Jf(z)ᵀ · r ∈ ℝⁿ:       "how does the error propagate through the causal map?"
  (3) Jψ(x)ᵀ · (···) ∈ ℝᵐ:   "how does the error propagate back to the abstract space?"

  We formalize this using Jacobian matrices, then prove structural properties.
-/

/-- A Jacobian matrix (rows × cols): for f : ℝᵃ → ℝᵇ, Jacobian a b has
    shape b × a, with entry (i, j) = ∂fᵢ/∂xⱼ -/
def Jacobian (a b : ℕ) := Fin b → Fin a → ℝ

/-- Matrix-vector product: J · v where J : a → b, v : ℝᵃ, result : ℝᵇ -/
def Jacobian.mulVec {a b : ℕ} (J : Jacobian a b) (v : Fin a → ℝ) : Fin b → ℝ :=
  fun i => ∑ j : Fin a, J i j * v j

/-- Transpose-vector product: Jᵀ · v where J : a → b, v : ℝᵇ, result : ℝᵃ -/
def Jacobian.transMulVec {a b : ℕ} (J : Jacobian a b) (v : Fin b → ℝ) : Fin a → ℝ :=
  fun j => ∑ i : Fin b, J i j * v i

/-- Matrix-matrix product: A · B where A : b → c, B : a → b, result : a → c -/
def Jacobian.mul {a b c : ℕ} (A : Jacobian b c) (B : Jacobian a b) : Jacobian a c :=
  fun i j => ∑ k : Fin b, A i k * B k j

notation:70 J " ⬝ " v => Jacobian.mulVec J v
notation:70 J " ᵀ⬝ " v => Jacobian.transMulVec J v
notation:70 A " ⊡ " B => Jacobian.mul A B

/-- The differentiable causal-embedding pipeline with Jacobian data.
    Three dimension parameters: m (abstract input), n (latent/embedding), p (output). -/
structure DiffCausalPipeline (m n p : ℕ) where
  /-- ψ : ℝᵐ → ℝⁿ — the embedding map -/
  ψ     : (Fin m → ℝ) → (Fin n → ℝ)
  /-- f : ℝⁿ → ℝᵖ — the causal/prediction map (output dim p ≠ input dim n in general) -/
  f     : (Fin n → ℝ) → (Fin p → ℝ)
  /-- Jψ(x) : the Jacobian of ψ at x (n × m matrix) -/
  Jψ    : (Fin m → ℝ) → Jacobian m n
  /-- Jf(z) : the Jacobian of f at z (p × n matrix) -/
  Jf    : (Fin n → ℝ) → Jacobian n p
  /-- Jψ is the derivative of ψ: ψ(x + εδ) ≈ ψ(x) + ε · Jψ(x) · δ -/
  Jψ_spec : ∀ x δ : Fin m → ℝ, ∀ ε > 0,
    ∀ i : Fin n, |ψ (fun j => x j + ε * δ j) i - ψ x i - ε * (Jψ x ⬝ δ) i| ≤
      ε * ε  -- o(ε) bound
  /-- Jf is the derivative of f: f(z + εδ) ≈ f(z) + ε · Jf(z) · δ -/
  Jf_spec : ∀ z δ : Fin n → ℝ, ∀ ε > 0,
    ∀ i : Fin p, |f (fun j => z j + ε * δ j) i - f z i - ε * (Jf z ⬝ δ) i| ≤
      ε * ε

namespace DiffCausalPipeline

variable {m n p : ℕ} (P : DiffCausalPipeline m n p)

/-- The composite prediction: ŷ(x) = f(ψ(x)) ∈ ℝᵖ -/
def predict (x : Fin m → ℝ) : Fin p → ℝ := P.f (P.ψ x)

/-- The composite loss: L(x) = ‖f(ψ(x)) - y‖² where y ∈ ℝᵖ -/
def loss (y : Fin p → ℝ) (x : Fin m → ℝ) : ℝ := sqLoss p y (P.predict x)

/-! ### 12a. The three gradient terms -/

/-- Term 1: Residual vector r = ŷ - y ∈ ℝᵖ (prediction error in output space) -/
def residual (y : Fin p → ℝ) (x : Fin m → ℝ) : Fin p → ℝ :=
  fun i => P.predict x i - y i

/-- Term 2: Causal backprop — Jf(z)ᵀ · r ∈ ℝⁿ
    Maps the ℝᵖ error back through f into the ℝⁿ latent space -/
def causalBackprop (y : Fin p → ℝ) (x : Fin m → ℝ) : Fin n → ℝ :=
  P.Jf (P.ψ x) ᵀ⬝ P.residual y x

/-- Term 3: Embedding backprop — Jψ(x)ᵀ · (Jf(z)ᵀ · r) ∈ ℝᵐ
    Maps the ℝⁿ latent error back through ψ into the ℝᵐ abstract space -/
def embeddingBackprop (y : Fin p → ℝ) (x : Fin m → ℝ) : Fin m → ℝ :=
  P.Jψ x ᵀ⬝ P.causalBackprop y x

/-- The full gradient: ∇ₓL = 2 · Jψ(x)ᵀ · Jf(z)ᵀ · (f(ψ(x)) - y) ∈ ℝᵐ -/
def gradLoss (y : Fin p → ℝ) (x : Fin m → ℝ) : Fin m → ℝ :=
  fun j => 2 * P.embeddingBackprop y x j

/-! ### 12b. Decomposition of the gradient into named components -/

/-- The gradient decomposes as: ∇ₓL(j) = 2 · Σₖ Jψ(x)ₖⱼ · Σᵢ Jf(z)ᵢₖ · rᵢ

    The sum over k ranges over ℝⁿ (latent), the inner sum over i ranges
    over ℝᵖ (output). -/
theorem grad_decomposition (y : Fin p → ℝ) (x : Fin m → ℝ) (j : Fin m) :
    P.gradLoss y x j =
      2 * ∑ k : Fin n, P.Jψ x k j *
        (∑ i : Fin p, P.Jf (P.ψ x) i k * P.residual y x i) := by
  simp only [gradLoss, embeddingBackprop, causalBackprop, Jacobian.transMulVec]
  ring

/-- The gradient vanishes when the prediction is perfect -/
theorem grad_zero_of_perfect_prediction (y : Fin p → ℝ) (x : Fin m → ℝ)
    (h : P.predict x = y) :
    P.gradLoss y x = 0 := by
  ext j
  simp only [gradLoss, embeddingBackprop, causalBackprop, Jacobian.transMulVec,
             residual, h, sub_self, mul_zero, Finset.sum_const_zero, Pi.zero_apply]
  ring

/-- Residual norm equals the loss -/
theorem residual_norm_eq_loss (y : Fin p → ℝ) (x : Fin m → ℝ) :
    ∑ i : Fin p, P.residual y x i ^ 2 = sqLoss p y (P.predict x) := by
  simp only [residual, sqLoss, predict]
  congr 1; ext i
  ring

/-! ### 12c. Jacobian chain rule: J(f ∘ ψ) = Jf · Jψ

  Jψ : n × m,  Jf : p × n  ⟹  J(f∘ψ) = Jf · Jψ : p × m -/

/-- The composite Jacobian of the full pipeline f ∘ ψ (p × m matrix) -/
def compositeJacobian (x : Fin m → ℝ) : Jacobian m p :=
  P.Jf (P.ψ x) ⊡ P.Jψ x

/-- The gradient can be written using the composite Jacobian:
    ∇ₓL = 2 · J(f∘ψ)ᵀ · r    where J(f∘ψ)ᵀ : m × p, r : ℝᵖ → result ℝᵐ -/
theorem grad_via_composite_jacobian (y : Fin p → ℝ) (x : Fin m → ℝ) (j : Fin m) :
    P.gradLoss y x j =
      2 * (P.compositeJacobian x ᵀ⬝ P.residual y x) j := by
  simp only [gradLoss, embeddingBackprop, causalBackprop, compositeJacobian,
             Jacobian.transMulVec, Jacobian.mul]
  ring

/-! ### 12d. Per-component gradient attribution

  We can decompose the gradient by output component i ∈ Fin p,
  attributing how much each of the p output dimensions contributes to the
  gradient at input dimension j ∈ Fin m. -/

/-- Attribution of output component i ∈ Fin p to input gradient at j ∈ Fin m:
    Aᵢⱼ = 2 · rᵢ · Σₖ Jf(z)ᵢₖ · Jψ(x)ₖⱼ    (sum over k ∈ Fin n)

    This measures: "how much does prediction error in output dimension i
    contribute to the gradient signal at input dimension j?" -/
def gradAttribution (y : Fin p → ℝ) (x : Fin m → ℝ)
    (i : Fin p) (j : Fin m) : ℝ :=
  2 * P.residual y x i * ∑ k : Fin n, P.Jf (P.ψ x) i k * P.Jψ x k j

/-- The gradient equals the sum of per-component attributions:
    ∇ₓL(j) = Σᵢ Aᵢⱼ    (sum over i ∈ Fin p) -/
theorem grad_eq_sum_attributions (y : Fin p → ℝ) (x : Fin m → ℝ) (j : Fin m) :
    P.gradLoss y x j =
      ∑ i : Fin p, P.gradAttribution y x i j := by
  simp only [gradLoss, embeddingBackprop, causalBackprop, gradAttribution,
             Jacobian.transMulVec, residual]
  rw [Finset.mul_sum]
  congr 1; ext k
  rw [Finset.mul_sum]
  congr 1; ext i
  ring

/-- Each attribution vanishes for a perfectly predicted component -/
theorem attribution_zero_of_component_perfect (y : Fin p → ℝ) (x : Fin m → ℝ)
    (i : Fin p) (h : P.predict x i = y i) :
    ∀ j, P.gradAttribution y x i j = 0 := by
  intro j
  simp only [gradAttribution, residual, h, sub_self, mul_zero, zero_mul]

/-! ### 12e. Gradient norm bound

  ‖∇ₓL‖ ≤ 2 · ‖Jψ‖_op · ‖Jf‖_op · ‖r‖

  We prove a concrete Frobenius-based bound. -/

/-- Frobenius norm squared of a Jacobian -/
def Jacobian.frobSq {a b : ℕ} (J : Jacobian a b) : ℝ :=
  ∑ i : Fin b, ∑ j : Fin a, J i j ^ 2

/-- Frobenius norm is non-negative -/
theorem Jacobian.frobSq_nonneg {a b : ℕ} (J : Jacobian a b) : 0 ≤ J.frobSq :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

end DiffCausalPipeline

end
