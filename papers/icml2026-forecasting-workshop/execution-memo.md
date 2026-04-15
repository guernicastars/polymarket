# Execution Memo

## Purpose

This memo translates the workshop proposal document into a repo-grounded action
plan. It answers one question:

Which paper can we defend honestly from the current `polymarket` repo, and what
must be built or run next?

## Bottom Line

- Proposal A is the strongest primary submission.
- Proposal B is still worth pursuing as a companion paper.
- Proposal A should be narrowed to metadata plus trade-derived calibration
  drivers unless orderbook history is backfilled before submission.
- Proposal B should be framed as a benchmark-and-systems paper with a minimum
  viable baseline set rather than a fully mature public platform.

## Why Proposal A Is Feasible

The repo already contains:

- resolved-market extraction logic
- calibration tracking code
- microstructure computation
- causal-analysis modules
- enough trade-derived data to support a real main analysis

The drafts can therefore claim:

- there is a live data and analysis stack
- calibration can be computed from observed market states
- causal structure learning is a concrete next step, not a speculative idea

The drafts should not yet claim:

- full retrospective orderbook coverage for resolved markets
- complete PC + FCI + GES execution unless that is actually run
- dense wallet coverage across the whole resolved universe

## Why Proposal B Is Feasible

The data substrate for a benchmark exists:

- market metadata
- trades
- orderbooks
- calibration and backtest infrastructure
- signal layers and market-state features

What does not yet exist as a neat paper artifact:

- benchmark episode specification
- stable replay service
- agent registration / logging interface
- dedicated benchmark baselines

That means the benchmark paper should focus on:

- benchmark framing
- scoring protocol
- live-plus-replay architecture
- baseline-agent evaluation on a limited but clean slice

## Highest-Risk Repo Facts

These are the facts most likely to distort the paper if ignored.

1. `experiments/RESEARCH.md` reports that the `market_prices` table was broken
   in the audited snapshot because `condition_id` was empty.
2. The same audit reports only 107 resolved markets with orderbook overlap.
3. Wallet-derived features are sparse and unevenly distributed.
4. Sports and esports dominate the resolved market universe, so pooled results
   can hide domain effects.

## Recommended Claim Discipline

### Proposal A

Safe central claim:

- calibration varies systematically with observable market structure and trading
  behavior, and a causal-analysis pipeline can separate candidate drivers from
  descriptive correlations

Avoid unless fully supported:

- "we recover the full causal DAG of prediction-market calibration"
- "orderbook microstructure is the primary causal driver" across the full
  universe

### Proposal B

Safe central claim:

- prediction markets enable a new benchmark regime in which AI forecasters are
  evaluated against a continuously updating crowd baseline

Avoid unless fully supported:

- "PolyBench is already a mature public benchmark with live agent support"
- "LLM agents reliably beat the market" before those runs exist

## Immediate Work Queue

1. Build the resolved-market calibration dataset for Proposal A.
2. Compute Brier and ECE slices by category and horizon.
3. Add a tail-aware layer using threshold ladders rather than only single
   binary contracts.
4. Extract the primary trade-derived feature table for all eligible resolved
   markets.
5. Run a first structure-learning pass on the main feature families.
6. Decide whether orderbook features are strong enough for main text or should
   move to an appendix.
7. Implement Market Echo and Momentum baselines for Proposal B.
8. Decide whether Proposal B includes a replay benchmark only, or a live
   benchmark plus replay appendix.
9. Port the final drafts into the ICML 2026 template once the first results are
   frozen.

## Tail-Aware Extension

The strongest conceptual refinement so far is:

- contract-level calibration is necessary;
- ladder-level exceedance-curve quality is closer to the real object for
  tail-sensitive decisions.
- truncated first- and second-moment proxies can be recovered from threshold
  ladders, giving us a practical variance proxy without pretending a single
  binary contract is the whole forecast.
- for trading systems, a useful additional label is deployability: whether the
  ladder is coherent enough to size risk, not merely accurate enough to quote.

We have now started that extension empirically through
`experiments/run_threshold_ladder_study.py`, which reconstructs multiple O/U
thresholds on the same event and evaluates the implied survival curve rather
than just one binary contract. The corrected pass now groups ladders by
normalized question template inside each event slug so it does not mix, e.g.,
match totals with set totals. On that cleaner definition we get 1,144 ladders.

The most useful new empirical bridge to profitability is cross-strike
incoherence:

- 16.1% of ladders show some adjacent dominance violation;
- 5.2% show a material (>2c) violation;
- ATM Brier has near-zero correlation with adjacent arbitrage gross edge.

That means the next paper table should probably not target ``profitability'' in
the abstract, but a more precise deployability label:

- does this ladder support Kelly-style sizing without a coherence haircut?
- should this market be gated or downweighted by the portfolio risk manager?

## Trader Feedback Integration

We also now have a concrete market-practitioner hypothesis from a Polymarket
trader:

- longshot bias is a central distortion, likely amplified by 100% margin
  requirements on the expensive side of the book;
- the effect is stronger when one leg is in the deep tail;
- the effect may be shrinking over time as more sophisticated capital enters;
- concentration and label quality matter, but are harder to operationalize.

That feedback has been converted into a dedicated experiment,
`experiments/run_longshot_bias_study.py`. The first pass supports the core
claim:

- final-horizon 1--2% longshots trade at roughly 3.5x realized frequency;
- 24h-horizon 0--2% longshots trade at roughly 4.9x realized frequency;
- 24h-horizon 0--10% longshot bias appears lower in March/April 2026 than in
  February 2026, though the trend is not yet clean enough to headline without
  category controls.

## Suggested Result Ownership

- Proposal A main text:
  calibration slicing, feature extraction, causal graph, effect table
- Proposal A appendix:
  subset orderbook analysis, additional category stratification, data quality
  notes
- Proposal B main text:
  benchmark framing, scoring, architecture, two baseline agents
- Proposal B appendix:
  replay assumptions, logging schema, additional benchmark slices

## Workshop Dates

Verified from the workshop website:

- Abstract registration deadline: May 15, 2026
- Submission deadline: May 20, 2026
- Paper length: up to 4 pages excluding references and appendix

Source:

- <https://forecasting-workshop.github.io/>

## Sources Consulted For This Draft Package

- Workshop proposal memo:
  `/Users/meuge/Downloads/ICML2026_Forecasting_Workshop_Proposals (1).docx`
- Repo:
  `/Users/meuge/coding/maynard/polymarket`
- Prediction-market calibration background:
  Page and Clemen (2013), Le (2026)
- Benchmark background:
  ForecastBench (2024), PredictionMarketBench (2026)
