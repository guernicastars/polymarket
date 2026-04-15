# Threshold Ladder Study

- Generated at: `2026-04-14T17:37:29.297853+00:00`
- O/U markets analyzed: `13166`
- Ladder events with >=3 thresholds: `1144`
- Median thresholds per ladder: `4.0`

## Ladder-Level Averages

- Mean final integrated Brier: `0.1843`
- Mean final ATM Brier: `0.2004`
- Mean final monotonicity violation rate: `0.0162`
- Mean abs. truncated-mean error: `0.9981`
- Share with adjacent arbitrage: `0.1608`
- Share with material (>2c) adjacent arbitrage: `0.0524`
- Mean adjacent arbitrage gross edge: `0.0113`
- Mean isotonic L1 repair gap: `0.0030`
- Tail-fit winner counts: `{'insufficient': 1074, 'tie': 54, 'power': 8, 'exp': 8}`

## Decoupling Diagnostics

- Corr(ATM Brier, curve Brier): `0.6788`
- Corr(ATM Brier, abs. mean-proxy error): `0.4969`
- Corr(ATM Brier, adjacent arbitrage gross edge): `0.0834`
- Share of low-ATM-Brier ladders with any monotonicity violation: `0.0662`
- Share of low-ATM-Brier ladders with high abs. mean error: `0.0397`
- Share of low-ATM-Brier ladders with any adjacent arbitrage: `0.3146`
- Share of low-ATM-Brier ladders with material (>2c) adjacent arbitrage: `0.0662`

## Interpretation

These ladders are a first step toward distributional forecasting in
prediction markets. A single binary contract only gives one tail
probability, but a ladder of thresholds approximates an exceedance
curve, which is much closer to the object needed for expected-value
and tail-risk analysis.
