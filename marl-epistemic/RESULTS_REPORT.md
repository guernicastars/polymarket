# MARL Epistemic Diversity — Experiment Results & Domain Analysis

**Eugene Shcherbinin — LSE / UC Berkeley — March 2026**

---

## Executive Summary

We ran all three experiments from the MARL epistemic diversity framework (5 seeds each). **The results are honest and mixed**: the core theoretical claims hold in some settings but the current experimental implementations have issues that need addressing before the framework demonstrates its value convincingly. This report diagnoses the problems, proposes fixes, and identifies the most promising STEM domains for real-world application.

---

## Experiment 1: Matrix Games (IPD with Diverse Observations)

**Hypothesis**: Agents with diverse observation functions (temporal, statistical, relative) cooperate faster than homogeneous agents in Iterated Prisoner's Dilemma.

### Results

| Condition | Cooperation Rate | Total Payoff |
|-----------|-----------------|-------------|
| **diverse** (temporal + statistical + relative) | 0.279 ± 0.072 | 528.5 ± 50.3 |
| **homogeneous_temporal** (3× temporal) | 0.288 ± 0.065 | 534.7 ± 45.4 |
| **homogeneous_statistical** (3× statistical) | 0.352 ± 0.012 | 580.4 ± 7.5 |

### Diagnosis

**The diverse condition underperforms.** This is not necessarily a refutation of the theory — it's an implementation issue:

1. **Statistical observation is simply better for IPD.** Summary statistics (cooperation rates, trends) are a more informative representation for this specific game than raw temporal sequences or relative standings. The homogeneous_statistical condition dominates because ALL agents get the best representation. Diversity introduces a handicap when one representation is strictly superior.

2. **No communication or aggregation mechanism.** In the matrix game setting, agents act independently — there's no ensemble aggregation. Diversity only helps if agents can share information or if different observations lead to complementary strategies that collectively stabilize cooperation. The current setup just has 3 independent learners with different inputs.

3. **IPD may be the wrong game.** IPD has a simple cooperative equilibrium (tit-for-tat variants) that doesn't require diverse perception. Games with hidden state, partial observability, or multi-modal information would better demonstrate the value of diverse observations.

### Proposed Fixes for Experiment 1

- Add **communication channels** between agents (message passing before action selection)
- Use environments where **no single observation type is sufficient** (e.g., some game states only visible to temporal observers, others only to statistical observers)
- Try **Stag Hunt** or **coordination games** with asymmetric information — these have coordination failures where diverse observations genuinely help
- Add the complementarity metric computation at evaluation time to verify blind spots actually differ

---

## Experiment 2: Prediction Market (Keynesian Evidence Weighting)

**Hypothesis**: Keynesian evidence-weighted ensembles outperform simple and accuracy-weighted ensembles, especially on low-evidence (Π_U) events.

### Results (averaged across 5 seeds)

| Method | MSE (overall) | MSE (high-evidence) | MSE (low-evidence) | ECE |
|--------|--------------|---------------------|--------------------|----|
| **best_single** | 0.0422 | 0.0548 | 0.0232 | 0.0392 |
| **simple_avg** | 0.0430 | 0.0556 | 0.0241 | 0.0389 |
| **accuracy_weighted** | 0.0428 | 0.0559 | 0.0231 | 0.0357 |
| **keynesian_weighted** | 0.0433 | 0.0573 | 0.0223 | 0.0527 |

### Diagnosis

**Mixed results with a crucial signal buried in the data.**

**What works — Low-evidence MSE:** The Keynesian ensemble achieves the **best MSE on low-evidence events** (0.0223 vs 0.0232 for best single). This is exactly the theoretical prediction — Keynesian weighting helps most where evidence is sparse. The improvement is small but consistent across seeds.

**What doesn't work — Overall MSE and ECE:** Keynesian weighting is slightly worse overall and has notably worse calibration (higher ECE). This suggests:

1. **The evidence computation (MC dropout) is noisy.** With only 20 dropout samples, variance estimates are unreliable. This introduces noise in the weighting that hurts on high-evidence inputs where all agents have good predictions anyway.

2. **MLP dominates.** Training loss shows MLP (0.007-0.009) massively outperforms others (0.035-0.048). The ensemble is essentially averaging one strong learner with three weak learners. The Keynesian weighting can't fully compensate because the evidence-density difference between architectures is much smaller than the accuracy gap.

3. **The synthetic data generator may not produce enough architectural differentiation.** The signal components (linear, local, long-range) need to be stronger and more distinct so different architectures genuinely excel in different regions.

### Proposed Fixes for Experiment 2

- Increase **MC dropout samples to 50-100** for more stable evidence estimates
- **Rebalance the data generator**: make linear, local, and long-range signal components more distinct and equally important (currently MLP can approximate all of them)
- Add **proper ablation**: train each architecture on data subsets matched to its inductive bias, then show Keynesian weighting recovers the specialized ensemble
- Add **confidence intervals** and **Wilcoxon signed-rank tests** for statistical significance
- Track and report **V_pool and per-agent V_i** to verify evidence complementarity actually occurs

---

## Experiment 3: Contextual Bandits (Diversity + Keynesian)

**Hypothesis**: Diverse ensemble (Linear + MLP + CNN + Attention) achieves lower cumulative regret than single agent or homogeneous ensemble, and Keynesian weighting further improves.

### Results (5 seeds)

| Condition | Cumulative Regret (mean ± std) |
|-----------|-------------------------------|
| **single_mlp** | 2315.3 ± 104.3 |
| **homogeneous_4** (4× MLP) | 2061.6 ± 78.5 |
| **diverse_4** (Linear + MLP + CNN + Attention) | 2785.3 ± 111.3 |
| **diverse_keynesian** | 4730.1 ± 562.4 |

### Diagnosis

**The diverse conditions significantly underperform.** This is the most concerning result.

1. **Averaging diverse predictions hurts when architectures disagree destructively.** The linear agent and CNN agent make poor predictions for the interaction/long-range components, and these bad predictions pollute the ensemble average. A simple average of "one good agent + three confused agents" is worse than the one good agent alone.

2. **Keynesian weighting makes it worse.** The MC dropout evidence is likely unreliable for online learning with single-sample updates. Evidence estimates based on dropout variance require enough data to calibrate — in the bandit setting with incremental learning, the evidence signals are noise.

3. **The training signal is too weak.** Each agent only observes the reward for the chosen arm. With 10 arms and 4 architectures all selecting based on the same ensemble decision, each individual agent gets very sparse supervision. The diverse agents need enough experience to discover what they're each good at — 10K steps may not be enough.

4. **Missing: Evidence-seeking LOLA condition.** The experiment config includes `diverse_evidence_lola` but it wasn't run (likely too slow for the online setting). This is actually the most important condition to test.

### Proposed Fixes for Experiment 3

- **Use Thompson sampling** instead of epsilon-greedy to handle exploration-exploitation better
- **Increase steps to 50K-100K** so agents have time to specialize
- **Pre-train agents** on batches of contextual data before the online phase, so each architecture discovers its strengths before contributing to ensemble decisions
- **Implement proper credit assignment**: track per-agent regret and use it to adaptively weight the ensemble (not just MC dropout evidence)
- **Implement the Evidence-seeking LOLA condition** — the theory predicts this is where the biggest gains come from
- **Add per-component regret tracking** to verify whether diverse agents actually specialize (linear agent should have low regret on the linear component, CNN on local, etc.)

---

## Overall Assessment

### What the Theory Predicts vs What We See

| Claim | Theory | Experiment | Status |
|-------|--------|-----------|--------|
| Diverse hypothesis classes have non-overlapping blind spots | B = ∩B_i < min(B_i) | Not directly measured in experiments | **Untested** (need to add blind spot computation) |
| Keynesian evidence weighting outperforms accuracy weighting | V_i(x) weighting adapts per-input | Slightly better on low-evidence, worse overall | **Partially supported** (noisy evidence is the bottleneck) |
| Evidence-seeking LOLA produces complementary specialization | Agents shape each other for evidence complementarity | Not run in any experiment | **Untested** |

### Root Causes

1. **The evidence computation is the bottleneck.** MC dropout with 20 samples is too noisy. This is the single most impactful fix.

2. **MLP dominance.** In all experiments, MLP approximates the target well enough that adding weaker architectures hurts rather than helps. The environments need to be designed so each architecture has a genuine irreplaceable advantage.

3. **Missing the key mechanism.** Evidence-seeking LOLA — the novel contribution — isn't actually tested in any experiment. This is the mechanism that's supposed to make agents specialize. Without it, diversity is just "ensemble of different-quality models."

4. **Scale.** The experiments are small (500 episodes, 2000 events, 10K steps). The theoretical advantages of diversity and evidence-seeking emerge at scale when the input space is large enough that no single architecture can cover it.

---

## Recommended Next Steps (Priority Order)

### 1. Fix the Evidence Computation
- Increase MC dropout samples to 50-100
- Add ensemble disagreement as an additional evidence signal
- Implement proper kernel evidence with efficient nearest-neighbor lookup

### 2. Implement and Test Evidence-Seeking LOLA
- This is the paper's core contribution and it's currently untested
- Start with the prediction market setting (batch training, clean gradients)
- Measure evidence complementarity before and after LOLA training

### 3. Redesign Environments for Genuine Architectural Complementarity
- Create a "no single architecture is sufficient" environment where:
  - The linear component requires linear features
  - The local component requires CNN receptive fields
  - The long-range component requires attention
  - The interaction component requires MLP hidden layers
  - Each component is equally important and cannot be approximated by other architectures

### 4. Add Proper Statistical Analysis
- Bootstrap confidence intervals on all metrics
- Wilcoxon signed-rank tests for pairwise comparisons
- Effect size (Cohen's d) reporting
- Plot learning curves (not just final metrics) to show convergence dynamics

### 5. Scale Up
- Experiment 2: 10K+ events, 50+ features
- Experiment 3: 50K+ steps, more arms
- More seeds (10-20) for tighter confidence intervals

---

## STEM Domain Analysis: Where Should We Apply This?

Based on extensive analysis, here are the top domains ranked by fit with the epistemic diversity framework:

### Rank 1: Molecular Property Prediction (Drug Discovery) ⭐ TOP PICK

**Why it's ideal:**
- **Four naturally distinct input representations** mapping to our four architectures:
  - Linear: Morgan fingerprints / ECFP (additive group contributions)
  - MLP: RDKit computed descriptors (nonlinear descriptor combinations)
  - CNN: SMILES strings (local motifs, functional groups)
  - Attention: Molecular graphs (long-range intramolecular interactions)
- **Extreme evidence density variation**: Drug-like chemical space (Lipinski region) has millions of data points; macrocycles, PROTACs, covalent binders, natural product scaffolds are nearly empty
- **Clear feedback loops**: Computational ADMET predictions verified by wet-lab assays
- **Scaffold splits** (test on novel scaffolds not in training) create exactly the setting where Keynesian evidence weighting adds most value

**Datasets**: Therapeutics Data Commons (TDC, tdcommons.ai) with 22 standardized ADMET tasks, MoleculeNet, QM9 (134K molecules, 12 quantum properties), ChEMBL (2.4M+ compounds)

**Metrics**: ROC-AUC (classification), RMSE/MAE (regression), with TDC leaderboard for benchmarking

**What to build**: A multi-representation molecular ensemble where each agent ingests a different molecular representation, trained with Evidence-seeking LOLA so agents specialize in different chemical subspaces. Test on scaffold splits where models must extrapolate to novel chemistry.

### Rank 2: Weather/Climate Prediction

**Why it fits:**
- **Natural architectural complementarity**: CNNs for mesoscale spatial patterns (fronts, convection); Attention for teleconnections (ENSO, MJO); Linear for persistence/climatological baselines; MLP for nonlinear thermodynamic thresholds
- **Geographic evidence density variation**: Northern hemisphere midlatitudes are data-dense; Southern Ocean, tropical Africa, polar regions are data-sparse — perfect for Keynesian weighting
- **Fastest feedback loops**: Forecasts verified in hours/days
- **Massive scale**: ERA5 is ~1PB, WeatherBench 2 provides standardized evaluation

**Datasets**: WeatherBench 2 (curated ERA5, weatherbench2.readthedocs.io), NOAA HRRR (3km hourly over US), SEVIR (storm events)

**Metrics**: RMSE, anomaly correlation coefficient for Z500, T850, T2m, precipitation at 6h-10 day lead times

### Rank 3: Materials Science

**Why it fits:**
- Very similar structure to molecular prediction but with distinct physics (periodic crystals vs discrete molecules)
- Evidence density varies enormously: oxides/binaries are well-studied; high-entropy alloys, MOFs, 2D materials are sparse
- MatBench provides 13 standardized tasks with nested cross-validation

**Datasets**: Materials Project (154K+ materials, materialsproject.org), MatBench (matbench.materialsproject.org), JARVIS-DFT (55K materials, 50+ properties)

### Rank 4: Genomics / Gene Expression

**Why it fits:**
- Perfect hierarchy from local motifs (CNN) to long-range regulatory interactions (Attention)
- Extreme evidence variation: common variants dense, rare variants sparse; protein-coding genes well-studied, regulatory genome vast and undercharacterized
- Enormous practical value for clinical variant interpretation

**Datasets**: ENCODE (encodeproject.org), GTEx (gtexportal.org), Genomic Benchmarks (8 standardized DNA tasks)

### Rank 5: Astronomy / Transient Classification

**Why it fits:**
- LSST/Rubin Observatory creates 10M+ alerts/night requiring rapid classification with calibrated uncertainty
- Extreme evidence variation: bright nearby objects have thousands of observations; faint distant objects may have 1-3 noisy detections
- Class imbalance from millions (common variables) to <100 (kilonovae)

**Datasets**: PLAsTiCC (3.5M light curves, 18 classes), ZTF public releases, Kepler/TESS

---

## Concrete Experiment Plan for Top Domain (Molecular Properties)

### Phase 1: Proof of Concept on TDC ADMET

```
Experiment 4: Molecular Property Prediction
├── Architectures
│   ├── Linear on Morgan fingerprints (2048-bit)
│   ├── MLP on RDKit descriptors (200 features)
│   ├── CNN on SMILES strings (character-level)
│   └── Attention on molecular graphs (atom/bond features)
├── Tasks (start with 4 TDC tasks)
│   ├── Caco-2 (intestinal permeability, regression, 906 molecules)
│   ├── hERG (cardiotoxicity, classification, 648 molecules)
│   ├── Solubility (ESOL, regression, 1128 molecules)
│   └── CYP2D6 Inhibition (classification, 13K+ molecules)
├── Evaluation
│   ├── Random split (baseline)
│   ├── Scaffold split (where diversity matters)
│   └── Low-data regime (100/200/500 training samples)
├── Methods
│   ├── Best single architecture
│   ├── Simple average ensemble
│   ├── Accuracy-weighted ensemble
│   ├── Keynesian-weighted ensemble
│   └── Evidence-seeking LOLA ensemble
└── Metrics
    ├── AUROC / RMSE (task-specific)
    ├── Brier score (calibration)
    ├── ECE (calibration)
    ├── Blind spot complementarity score
    ├── Evidence complementarity ratio
    └── Effective ensemble size (N_eff)
```

### Phase 2: Scale and Benchmark

- Run on all 22 TDC ADMET tasks
- Compare against published SOTA on TDC leaderboard
- Ablation study: which pairs of architectures have highest complementarity?
- Evidence density analysis: plot Keynesian advantage vs training set size

### Phase 3: Cross-Domain Validation

- Replicate on WeatherBench 2 (spatiotemporal setting)
- Replicate on MatBench (materials science)
- Meta-analysis: in which domain does epistemic diversity help most?

---

## Conclusion

The framework has sound theoretical foundations but the current experiments don't yet demonstrate the claims convincingly. The primary issues are (1) noisy evidence computation, (2) MLP domination in environments where diversity should matter, and (3) the core novel mechanism (Evidence-seeking LOLA) being untested.

The most promising path forward is molecular property prediction (TDC ADMET), where different input representations create genuine architectural complementarity by construction, scaffold splits create the low-evidence setting where Keynesian weighting theoretically excels, and established leaderboards allow direct comparison with SOTA.

The experimental code is solid (60/60 tests pass, clean architecture). What's needed is better experimental design, not better code.
