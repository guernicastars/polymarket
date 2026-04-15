# ICML 2026 Forecasting Workshop Drafts

This folder contains the first paper-draft pass for the ICML 2026 workshop
"Forecasting as a New Frontier of Intelligence."

The drafting strategy follows the workshop memo prepared on April 14, 2026:

- Proposal A is the primary paper.
- Proposal B is the companion systems / benchmark paper.
- Proposal C is not being drafted as a submission target in this pass, but some
  of its "thick-to-thin" transfer framing can later be moved into discussion or
  future-work sections.

These drafts are grounded in two sources of truth:

- the workshop proposal memo in
  `/Users/meuge/Downloads/ICML2026_Forecasting_Workshop_Proposals (1).docx`
- the local repo at `/Users/meuge/coding/maynard/polymarket`

They are intentionally honest about current evidence. Where the repo already
contains implemented infrastructure, the drafts say so. Where major results
still need to be run, the drafts mark result slots explicitly instead of
pretending the experiments already exist.

## Workshop Constraints

Verified from the workshop website on April 14, 2026:

- Abstract registration deadline: May 15, 2026
- Full submission deadline: May 20, 2026
- Format: up to 4 pages excluding references and appendix
- Call for Papers: <https://forecasting-workshop.github.io/>

## Files

- `proposal-a-causal-calibration-draft.md`
  Primary paper draft on causal drivers of prediction market calibration.
- `proposal-b-polybench-draft.md`
  Companion benchmark / systems paper draft.
- `execution-memo.md`
  Repo-grounded assessment of what is implemented, what is missing, and what to
  run next.

## Empirical Scripts

- `experiments/run_calibration_study.py`
  First descriptive calibration pass on resolved binary markets.
- `experiments/run_threshold_ladder_study.py`
  Tail-aware extension that reconstructs threshold ladders from multiple
  Over/Under strikes on the same event, scores the implied exceedance curve,
  and recovers truncated moment proxies plus deployability diagnostics between a
  central threshold contract and the full ladder, including cross-strike
  incoherence labels relevant for trading/risk management.
- `experiments/run_longshot_bias_study.py`
  Trader-feedback extension that tests longshot bias, extreme-tail
  overpricing, and simple time-trend/category heterogeneity using resolved
  market calibration outputs.

## Repo Grounding

The current draft package assumes the following repo facts:

- Continuous ingestion infrastructure exists for market metadata, trades,
  orderbooks, wallet activity, and derived analytics.
- The repo already contains:
  `pipeline/bayesian/calibration.py`,
  `pipeline/jobs/news/microstructure.py`,
  `pipeline/causal/*.py`,
  `pipeline/jobs/signal_compositor.py`,
  `network/backtest.py`,
  and a resolved-market embedding pipeline.
- The checked-in audit at `experiments/RESEARCH.md` reports a February 20, 2026
  data snapshot with 11,368 resolved markets, 5,290 resolved markets with trade
  data, and only 107 resolved markets with orderbook overlap.

That last point matters: Proposal A is feasible now, but the cleanest first
submission should primarily rely on metadata plus trade-derived features, with
orderbook-heavy analyses treated as a subset or appendix unless a backfill is
completed before submission.
