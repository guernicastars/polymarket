# Proposal B Draft

## PolyBench: A Real-Time Prediction Market Benchmark for AI Forecasting Agents

Draft status: first prose draft grounded in current repo capabilities. This
draft is intentionally split between what is already implemented and what still
needs to be built before a submission-ready benchmark paper exists.

## Abstract

Current AI forecasting benchmarks mostly evaluate forecasters on static question
sets with delayed feedback and no continuously updating market baseline.
Prediction markets offer a different regime: probabilities update in real time,
settlement is objective, and the benchmark includes a strong crowd-implied
reference forecast at every evaluation point. We propose PolyBench, a benchmark
for forecasting agents built on a live Polymarket data stack. The underlying
repository already ingests market metadata, trades, orderbooks, wallet activity,
and derived analytics into ClickHouse; it also contains calibration tracking,
backtesting, and market-signal infrastructure that can be repurposed for agent
evaluation. PolyBench evaluates an agent not only by standard proper scoring
rules such as Brier score and log score, but also by *informational alpha*: the
marginal value of an agent relative to the market-implied baseline at the moment
the forecast is made. The benchmark is designed to support both live and replay
evaluation, to stratify performance by liquidity and domain, and to make a
clean distinction between beating the event outcome and beating the market. This
draft defines the benchmark task, data model, scoring protocol, baseline agent
set, and minimum viable implementation path for a workshop submission.

## 1. Motivation

Forecasting benchmarks have improved quickly, but they still leave an important
gap between forecasting as a question-answering task and forecasting as a live
decision problem.

ForecastBench is valuable because it is dynamic and leakage-resistant, but it
does not provide a market-implied baseline that updates continuously between
question creation and resolution. PredictionMarketBench moves closer to market
realism by replaying exchange data for trading agents, but it focuses on
execution-realistic backtesting rather than a general benchmark for probability
forecasting agents operating against a live crowd baseline.

PolyBench targets the missing middle:

- a real-time benchmark
- objective resolution
- continuous crowd baseline
- domain coverage across many event types
- evaluation that rewards useful disagreement with the market rather than
  disagreement for its own sake

The main question is:

When an AI forecaster diverges from the market, is it adding information or just
adding noise?

## 2. Benchmark Task

At time `t`, an agent observes a benchmark market and emits a probability
forecast `p_agent(t)` for the event's positive outcome.

For each forecast event, PolyBench records:

- market identifier
- timestamp
- market-implied probability at that timestamp
- spread
- depth or other liquidity-quality features when available
- category and horizon metadata
- agent forecast
- eventual resolved outcome

The benchmark should support two modes:

- live mode, where agents forecast active markets
- replay mode, where historical market states are served deterministically

The workshop paper can introduce both modes conceptually, while reporting only
the mode that is actually implemented by submission time.

## 3. Why Prediction Markets Change Benchmark Design

Prediction markets provide three benchmark properties that static forecasting
sets do not.

### 3.1 Continuous Feedback Before Resolution

Even before the outcome is known, the benchmark can evaluate whether an agent is
consistently moving toward or away from the crowd in informative ways. This
does not replace proper scoring at settlement, but it creates richer diagnostic
signals during a market's life.

### 3.2 Strong Baseline At Every Timestamp

The benchmark is never scoring an agent in a vacuum. It always compares the
agent against the market price available at the time the forecast was made.

### 3.3 Liquidity-Aware Difficulty

A 5-point disagreement with the market in a deep, liquid contract means
something different from a 5-point disagreement in a thin, low-depth market.
PolyBench should therefore report performance stratified by liquidity regime,
spread regime, and horizon.

## 4. Scoring Protocol

The benchmark should use standard proper scoring rules plus a differential
metric that explicitly measures value added over the market.

### 4.1 Core Metrics

- Brier score
- log score
- Expected Calibration Error
- coverage by domain and horizon

### 4.2 Informational Alpha

For a single forecast instance `i`, define:

`alpha_i = (p_market_i - y_i)^2 - (p_agent_i - y_i)^2`

where `y_i` is the realized binary outcome.

Interpretation:

- `alpha_i > 0`: the agent beat the market on that forecast
- `alpha_i = 0`: the agent matched the market
- `alpha_i < 0`: the agent was worse than the market

The benchmark should report:

- mean informational alpha
- disagreement-conditioned alpha, computed only when
  `|p_agent - p_market| >= tau`
- liquidity-stratified alpha
- category-stratified alpha

This is the conceptual heart of Proposal B. The benchmark is not asking whether
an agent is merely calibrated in isolation; it is asking whether the agent adds
marginal information over an already strong crowd baseline.

## 5. Repo Grounding

PolyBench is plausible because the repo already contains most of the data layer.

Implemented today:

- continuous ingestion of market metadata, trades, orderbooks, and WebSocket
  events
- derived tables for OHLCV, latest prices, wallet activity, arbitrage,
  composite signals, and market similarity
- calibration tracking code in `pipeline/bayesian/calibration.py`
- backtesting and calibration analysis in `network/backtest.py`
- signal and microstructure layers that can provide benchmark covariates

Not yet packaged as a benchmark:

- a stable benchmark episode specification
- a public agent interface
- baseline-agent runners dedicated to the benchmark paper
- a canonical replay layer that serves historical observations to agents

This means the paper should be framed as a systems / benchmark paper with a
minimum viable implementation rather than as a fully mature platform paper.

## 6. Baseline Agents

The benchmark paper needs at least two or three baselines that are simple,
transparent, and fast to run.

### Baseline 1: Market Echo

Always predict the current market-implied probability.

Purpose:

- lower bound for informational alpha
- reference point for Brier and calibration

### Baseline 2: Momentum

Use recent market movement and possibly volume acceleration to extrapolate the
next forecast.

Minimal implementation path:

- derive signal from `one_day_price_change`, recent returns, and trade imbalance
- clip into `[0, 1]`
- compare against Market Echo on the same forecast timestamps

### Baseline 3: Retrieval-Augmented LLM Forecaster

This is the most interesting but least necessary baseline for the first pass.
If time is short, the paper can define this baseline and leave it for future
work, or include only a small pilot on one category.

The benchmark paper does not need a large model story to be valuable. It needs
clean benchmark design, reproducible scoring, and at least one or two working
baseline agents.

## 7. Benchmark Slices

To avoid a single pooled score that hides difficulty differences, the paper
should report results by slice:

- politics
- sports / esports
- crypto
- science / weather
- short-horizon vs long-horizon markets
- high-liquidity vs low-liquidity markets

If baseline results are sparse by submission time, it is better to report fewer
well-powered slices than many thin ones.

## 8. Proposed Evaluation Protocol

### Data Inclusion

Use only binary or otherwise cleanly binarized markets for the first paper.

Recommended initial filters:

- resolved markets only
- known winner
- minimum trade threshold
- minimum market lifetime
- market snapshots with sufficient observability

### Temporal Splitting

Train or tune on earlier resolutions, evaluate on later resolutions.

This matters even for simple baselines because it avoids using future market
behavior to pick thresholds.

### Two Benchmark Modes

If implementation bandwidth permits:

- replay benchmark for reproducible paper results
- live benchmark for public leaderboard follow-on

If not, the workshop paper can focus on replay and describe live evaluation as
the next release.

## 9. Honest Limitations

This draft should not hide the current implementation gaps.

1. The repo already has the data substrate, but not yet a polished benchmark
   API.
2. Historical orderbook and price coverage for old resolved markets may be too
   uneven for a fully execution-realistic benchmark on the first pass.
3. The first paper may need to benchmark probability forecasters more than full
   tool-using trading agents.
4. A live leaderboard is attractive, but it is not necessary for a credible
   workshop submission.

## 10. Minimum Viable Submission Path

The strongest realistic version of Proposal B by the workshop deadline is:

1. Introduce PolyBench and its scoring protocol.
2. Describe the live data stack and replay design.
3. Report Market Echo and Momentum baselines.
4. Show informational alpha stratified by at least domain and liquidity.
5. Position the benchmark as infrastructure for future agent competitions and
   forecasting-agent evaluation.

That is enough to make the benchmark paper complementary to Proposal A rather
than redundant with it.

## 11. Result Slots To Fill

### Main Text

- Figure 1: benchmark architecture and event flow
- Table 1: market universe and slice definitions
- Table 2: baseline-agent results on Brier, log score, ECE, and alpha
- Figure 2: alpha by liquidity regime
- Figure 3: alpha by category

### Appendix

- episode specification
- forecast record schema
- replay assumptions
- fee and slippage assumptions if trading-style replay is included

## References To Carry Into The Final Version

- Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary Jacobs, Danny Halawi,
  Fred Zhang, and Philip E. Tetlock. "ForecastBench: A Dynamic Benchmark of AI
  Forecasting Capabilities." arXiv:2409.19839, 2024.
  <https://arxiv.org/abs/2409.19839>
- Avi Arora and Ritesh Malpani. "PredictionMarketBench: A SWE-bench-Style
  Framework for Backtesting Trading Agents on Prediction Markets."
  arXiv:2602.00133, 2026.
  <https://arxiv.org/abs/2602.00133>
- Glenn W. Brier. "Verification of Forecasts Expressed in Terms of
  Probability." *Monthly Weather Review*, 1950.
- Nam Anh Le. "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics
  in Prediction Markets." arXiv:2602.19520, 2026.
  <https://arxiv.org/abs/2602.19520>
