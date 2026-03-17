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
  f : ℝⁿ → ℝⁿ  (causal map)
  X ≜ ψ(Π),  Y ≜ f(X)
-/

/-- The full causal-embedding structure over a market space -/
structure CausalEmbedding (n : ℕ) where
  M     : MarketSpace
  /-- ψ : Π → ℝⁿ, the embedding (Step 6: bijection) -/
  ψ     : M.Π → (Fin n → ℝ)
  ψ_inj : Injective ψ
  /-- f : ℝⁿ → ℝⁿ, the causal/prediction map -/
  f     : (Fin n → ℝ) → (Fin n → ℝ)

namespace CausalEmbedding

variable {n : ℕ} (C : CausalEmbedding n)

/-- X ≜ ψ(Π) = { x ∈ ℝⁿ | ψ⁻¹(x) ∈ Π } — the embedded resolved set -/
def X : Set (Fin n → ℝ) := range C.ψ

/-- Y ≜ f(X) — the image under the causal map -/
def Y : Set (Fin n → ℝ) := C.f '' C.X

end CausalEmbedding

/-! ## 7. Squared-error loss  ℓ(Y, Ŷ) ≜ ‖Y - Ŷ‖²  (top-right of board) -/

/-- Squared-error loss in ℝⁿ -/
def sqLoss (n : ℕ) (y ŷ : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, (y i - ŷ i) ^ 2

/-- Loss is non-negative -/
theorem sqLoss_nonneg (n : ℕ) (y ŷ : Fin n → ℝ) : 0 ≤ sqLoss n y ŷ :=
  Finset.sum_nonneg fun i _ => sq_nonneg _

/-- Loss is zero iff predictions match -/
theorem sqLoss_eq_zero_iff (n : ℕ) (y ŷ : Fin n → ℝ) :
    sqLoss n y ŷ = 0 ↔ y = ŷ := by
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

  The diagram asserts that embedding commutes with the causal map:

       Π ---St--→ Π
       |           |
     ψ |           | ψ
       ↓           ↓
      ℝⁿ ---f--→ ℝⁿ

  i.e.  f ∘ ψ = ψ ∘ (causal action on Π)
-/

/-- A causal embedding commutes if f ∘ ψ = ψ ∘ g for some g : Π → Π -/
structure CommutativeCausalEmbedding (n : ℕ) extends CausalEmbedding n where
  /-- g : the causal dynamics on the abstract resolved space -/
  g         : toMarketSpace.Π → toMarketSpace.Π
  /-- The diagram commutes: f(ψ(x)) = ψ(g(x)) -/
  commutes  : ∀ x : toMarketSpace.Π, toCausalEmbedding.f (toCausalEmbedding.ψ x) =
                                       toCausalEmbedding.ψ (g x)

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

end
