# MARL Epistemic Diversity

**Epistemic Diversity and Meta-Learning in Multi-Agent Systems**

Eugene Shcherbinin — LSE / UC Berkeley — 2026

## Core Claims

1. **Blind Spot Complementarity**: Agents with diverse hypothesis classes collectively learn strictly more than any single agent
2. **Keynesian Evidence Pooling**: Weighting by weight of evidence V_i outperforms accuracy-weighted ensembles
3. **Evidence-Seeking LOLA**: LOLA + Keynesian evidence-seeking produces complementary evidence specialization

## Setup

```bash
pip install -e ".[dev]"
```

## Experiments

```bash
# Experiment 1: Matrix Games
python scripts/run_experiment1.py --seeds 5

# Experiment 2: Prediction Market
python scripts/run_experiment2.py --seeds 5

# Experiment 3: Contextual Bandits
python scripts/run_experiment3.py --seeds 5 --steps 10000

# Generate figures
python scripts/generate_figures.py --output-dir figures
```

## Tests

```bash
pytest tests/ -v
```

## Architecture

- `src/agents/` — Four hypothesis classes (Linear, MLP, CNN, Attention) + ensembles
- `src/learning/` — REINFORCE, LOLA, Evidence-Seeking LOLA, Keynesian loss
- `src/environments/` — Matrix games, prediction market, contextual bandits
- `src/metrics/` — Blind spot, evidence, calibration, diversity measures
- `scripts/` — Experiment runners and figure generation
