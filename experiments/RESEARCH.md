# Polymarket Embedding Experiment: Data Coverage Research

## Executive Summary

We have **11,368 resolved markets** in ClickHouse with binary ground truth (winning outcome known for 11,054 of them). Trade data covers 5,290 of these markets (46.6%), with a median of 28 trades per market and up to 9,891. The pipeline has been collecting data since **Feb 14, 2026** (~6 days), so time-series depth is limited but market metadata (from Gamma API) spans the full history of all resolved markets.

**Recommended dataset for the experiment:** ~5,290 resolved markets with trade data, filtered to the ~4,092 markets with 10+ trades for minimum signal quality. For the richest feature set, the ~1,541 markets with 100+ trades provide the best balance of sample size and feature quality.

---

## 1. Market Universe

| Metric | Value |
|--------|-------|
| Total markets in DB | 69,218 |
| Open (active) | 57,850 |
| Closed | 11,368 |
| Closed + Resolved | 11,368 |
| With known winner | 11,054 (97.2%) |
| No winner set | 314 (2.8%) |
| Unique events | 2,009 |
| Avg markets per event | 5.66 |
| Max markets per event | 126 |
| Unique categories | 163 |
| Pipeline start date | 2026-02-14 |
| Latest market update | 2026-02-20 |

### Outcome Types
| Outcome type | Winner | Count |
|--------------|--------|-------|
| Yes/No | Yes | 981 |
| Yes/No | No | 3,474 (most binary markets resolve No) |
| Over/Under | Over | 1,998 |
| Over/Under | Under | 2,114 |
| Named outcomes (sports/esports teams, etc.) | Various | 2,480 |
| No winner recorded | - | 314 |

**Binary markets (Yes/No + Over/Under):** 8,756 total -- these are the cleanest for a binary classification experiment since outcome encoding is standardized.

### Category Breakdown (Top 20, Resolved)
| Category | Count | Avg Volume ($) | With Winner |
|----------|-------|----------------|-------------|
| Esports | 3,037 | 22,719 | 2,857 |
| Sports | 2,456 | 67,130 | 2,449 |
| Tennis | 1,987 | 10,037 | 1,869 |
| Weather | 328 | 20,527 | 328 |
| Culture | 312 | 226,622 | 312 |
| Winter Olympics | 238 | 2,914 | 238 |
| Olympics | 184 | 3,291 | 184 |
| Games | 178 | 47,167 | 178 |
| League of Legends | 176 | 17,496 | 176 |
| Daily Temperature | 152 | 14,913 | 152 |
| (empty) | 138 | 37,372 | 132 |
| Politics | 136 | 191,356 | 136 |
| Movies | 113 | 25,018 | 113 |
| Basketball | 109 | 21,305 | 109 |
| Mentions | 97 | 15,404 | 97 |
| Crypto | 88 | 224,633 | 88 |
| Awards | 75 | 102,595 | 75 |
| Ethereum | 71 | 119,049 | 71 |
| Geopolitics | 64 | 4,276,178 | 64 |
| Trump | 62 | 88,357 | 62 |

---

## 2. Data Coverage per Table

### 2.1 Market Metadata (`markets` table)
- **All 11,368** resolved markets have full metadata from Gamma API
- Fields universally available: condition_id, question, description, event_id, category, tags, outcomes, outcome_prices, token_ids, volume_total, volume_24h, volume_1wk, volume_1mo, one_day_price_change, one_week_price_change, neg_risk, start_date, end_date
- `competitive_score` is always 0 (not populated by Gamma API for closed markets)
- `neg_risk` flag set for 2,776 markets (24.4%)

### 2.2 Volume Distribution (Resolved Markets)
| Threshold | Count | % of 11,368 |
|-----------|-------|-------------|
| Any volume > 0 | 10,483 | 92.2% |
| Volume > $100 | 9,545 | 84.0% |
| Volume > $1,000 | 6,692 | 58.9% |
| Volume > $10,000 | 3,065 | 27.0% |
| Volume > $100,000 | 1,015 | 8.9% |
| Volume > $1,000,000 | 178 | 1.6% |

### 2.3 Trade Data (`market_trades` table)
- **Total rows:** 14.8M across 27,397 unique markets
- **Resolved markets with trades:** 5,290 (46.6% of resolved)
- **Time range:** 2026-02-14 to 2026-02-20 (~6 days)
- **Aggregate for resolved:** 894,269 trades, $217.9M total volume

| Trade count threshold | Markets | % of 5,290 |
|----------------------|---------|------------|
| >= 10 trades | 4,092 | 77.4% |
| >= 50 trades | 2,121 | 40.1% |
| >= 100 trades | 1,541 | 29.1% |
| >= 500 trades | 388 | 7.3% |
| >= 1,000 trades | 156 | 2.9% |

- **Median trades per market:** 28
- **Average trades per market:** 169
- **Max trades per market:** 9,891
- **Buy/sell ratio:** 79.5% buys / 20.5% sells

### 2.4 Price Data (`market_prices` table)
- **22.8M rows** but **condition_id is empty for all rows** -- this table is unusable in its current state
- Only 1 unique "market" (the empty string)
- The price poller appears to write rows without the condition_id field
- **Impact:** We cannot use tick-level price data. Must derive price features from `market_trades` or `ohlcv_1h` instead.

### 2.5 OHLCV Data (`ohlcv_1h` table)
- **Total rows:** 223,598 across 27,390 unique markets
- **Resolved markets with OHLCV:** 5,290 (same as trades -- materialized view)
- **Median bars per resolved market:** 3
- **Average bars per resolved market:** 7.97
- **Max bars per resolved market:** 214
- **Time range:** 2026-02-14 to 2026-02-20

Note: Low bar count per market because the pipeline has only been running ~6 days. Markets that resolved quickly after pipeline start have fewer bars.

### 2.6 Orderbook Snapshots (`orderbook_snapshots` table)
- **Total rows:** 983,976 across 431 unique markets
- **Resolved markets with orderbook data:** 107 markets (0.9%)
- **7-day TTL** means older data is automatically deleted
- Very limited overlap -- only useful as supplementary feature for the ~107 markets

### 2.7 Wallet/User Data
| Table | Total Rows | Unique Wallets | Unique Markets | Coverage for Resolved |
|-------|-----------|----------------|----------------|----------------------|
| trader_rankings | 64,216 | 24,195 | - | Leaderboard rankings |
| trader_profiles | 14,067 | ~14,067 | - | Profile data |
| wallet_positions | 458,280 | 4,716 | 121,699 | Positions |
| wallet_activity | 115,974 | 143 | 2,334 | 13,791 rows / 52 wallets for resolved |
| market_holders | 0 | 0 | 0 | Not yet populated |

### 2.8 Analytics Tables
| Table | Count | Notes |
|-------|-------|-------|
| composite_signals | 1,530 total; 426 for resolved markets | Multi-factor scores |
| insider_scores | 14,144 | 50 relevant to resolved market wallets |
| wallet_clusters | 45 | Small; clustering capped at 200 wallets |
| arbitrage_opportunities | 10,967,648 | Large table; mostly historical |

---

## 3. Ground Truth: Binary Label

For the embedding experiment, the **target variable** is the market outcome:

**For binary markets (Yes/No, Over/Under):**
- Label = 1 if first outcome wins (Yes, Over)
- Label = 0 if second outcome wins (No, Under)
- 8,756 binary markets available

**For named-outcome markets (sports matchups, etc.):**
- Can encode as "did the favored outcome win?" based on pre-resolution price
- Or exclude from initial experiment (focus on binary markets only)

**Class balance (binary markets):**
- Yes/No: 981 Yes (22%) vs 3,474 No (78%) -- **imbalanced** (No-heavy)
- Over/Under: 1,998 Over (48.6%) vs 2,114 Under (51.4%) -- **balanced**

---

## 4. Proposed Feature Set (27 features, 5 categories)

### Category A: Market Structure (5 features) -- Available for ALL 11,368 resolved markets
1. **volume_total** -- Total lifetime USDC volume (log-scaled)
2. **liquidity** -- Liquidity at close (currently 0 for most closed markets; may need snapshot)
3. **neg_risk** -- Binary: is this a negative-risk market? (0/1)
4. **market_duration_days** -- `end_date - start_date` in days (market lifespan)
5. **num_outcomes** -- Number of possible outcomes (2 for binary, N for multi)

### Category B: Price Signal Features (6 features) -- Available for ~5,290 markets with trade data
6. **last_price** -- Final price before resolution (from last trade or `outcome_prices[0]`)
7. **one_day_price_change** -- 24h price change (from Gamma API, stored on market)
8. **one_week_price_change** -- 7d price change (from Gamma API)
9. **price_range** -- max(price) - min(price) from trades (volatility proxy)
10. **price_at_75pct_life** -- Price at 75% of market lifetime (how early was the outcome priced in?)
11. **final_price_velocity** -- Rate of price change in the last 10% of observed trades

### Category C: Trade Microstructure (8 features) -- Available for ~5,290 markets
12. **trade_count** -- Total number of trades (log-scaled)
13. **avg_trade_size** -- Average USDC per trade
14. **max_trade_size** -- Largest single trade
15. **buy_sell_ratio** -- Fraction of trades that are buys
16. **buy_volume_ratio** -- Fraction of volume from buys (USDC-weighted)
17. **trade_size_gini** -- Gini coefficient of trade sizes (concentration of large traders)
18. **trades_per_day** -- Average daily trade frequency
19. **late_volume_ratio** -- Fraction of volume in the last 25% of market lifetime

### Category D: Volume & Activity Dynamics (4 features) -- Available for ~5,290 markets
20. **volume_24h** -- Rolling 24h volume at close (from Gamma API)
21. **volume_1wk** -- Rolling 1-week volume (Gamma API)
22. **volume_acceleration** -- volume_24h / (volume_1wk / 7) -- spike ratio
23. **volume_total_vs_category_median** -- Market's volume relative to its category median

### Category E: Wallet/Smart Money Features (4 features) -- Available for subset (~2,334 markets with wallet_activity)
24. **unique_wallet_count** -- Number of distinct tracked wallets trading this market
25. **whale_buy_ratio** -- Fraction of tracked-wallet volume on the buy side
26. **top_wallet_concentration** -- Fraction of volume from the single largest tracked wallet
27. **avg_insider_score** -- Average insider risk score of wallets active in this market

### Potential Bonus Features (from composite_signals, 426 resolved markets)
- **obi_score** -- Order book imbalance
- **momentum_score** -- Price momentum
- **smart_money_score** -- Whale direction signal
- **composite_score** -- Overall composite signal

These are only available for 426 resolved markets, so they would be used as a secondary analysis or for a smaller, richer subset experiment.

---

## 5. Dataset Tiers for the Experiment

| Tier | Filter | Markets | Features Available |
|------|--------|---------|-------------------|
| **Tier 1 (Full)** | Binary + resolved + winner known | ~8,442 | A (5 features) |
| **Tier 2 (With Trades)** | Tier 1 + has trade data (>=3 trades) | ~4,500 | A + B + C + D (23 features) |
| **Tier 3 (Rich)** | Tier 1 + >=100 trades | ~1,500 | A + B + C + D (23 features, higher quality) |
| **Tier 4 (Full Signal)** | Tier 3 + wallet + signals data | ~400 | All 27 features + bonus |

**Recommendation:** Use **Tier 2** (~4,500 markets) as the primary dataset with 23 features. This gives enough samples for meaningful training/test splits while having trade-based features. Use Tier 1 as a validation set for metadata-only features. Tier 4 is too small for robust neural network training.

---

## 6. Key Observations for Experiment Design

1. **Class imbalance in Yes/No markets.** Yes wins only 22% of the time. Over/Under is balanced (~49/51). Consider: (a) stratified sampling, (b) separate models per outcome type, or (c) combine and use outcome-type as a feature.

2. **market_prices table is broken.** The condition_id column is empty for all 22.8M rows. All price-based features must come from `market_trades` or `ohlcv_1h`.

3. **Short pipeline history.** Only ~6 days of collection means time-series features (OHLCV bars, orderbook depth over time) are shallow. Gamma API metadata fields (volume_total, price changes) capture historical aggregates and are more complete.

4. **Feature correlation risk (the multicollinearity problem).** Several features are expected to be highly correlated:
   - volume_total, trade_count, volume_24h, volume_1wk
   - buy_sell_ratio, buy_volume_ratio
   - one_day_price_change, one_week_price_change
   - This is exactly what the embedding experiment aims to solve: the autoencoder should learn a decorrelated latent space.

5. **Category as a confound.** Esports (3,037) and Sports (2,456) dominate the resolved market set. Category strongly determines base rates (e.g., weather markets may have different dynamics than politics). Include category as a feature or stratify by it.

6. **Negative risk markets** (24.4%) are structurally different (correlated multi-outcome events). May warrant separate handling or a dedicated feature.

---

## 7. Data Quality Notes

- **Missing data pattern:** Features in Category E (wallet/smart money) are missing for ~60% of markets. Need imputation strategy (e.g., fill with category-level medians, or use separate feature matrices).
- **Zero liquidity:** Most closed markets report `liquidity = 0` since liquidity drains at resolution. This field is not useful as-is; would need historical snapshots.
- **Stale competitive_score:** Always 0. Drop from feature set.
- **Winner encoding:** 314 resolved markets (2.8%) have no `winning_outcome` set. Exclude these from the labeled dataset.

---

## 8. Recommended Next Steps

1. **Extract dataset** from ClickHouse: Join `markets FINAL` with `market_trades` aggregations. Output as Parquet.
2. **Feature engineering** in Python: Compute the 23-27 features above from raw data.
3. **Exploratory data analysis**: Correlation matrix, distribution plots, missingness map.
4. **Autoencoder training**: Encode the 23 features into a latent space (dim 8-12).
5. **Linear probe comparison**: Train linear classifiers on (a) raw features, (b) PCA-reduced features, (c) autoencoder embeddings. Compare AUROC.
6. **Permutation importance**: Compare feature importance rankings between raw vs. embedded features.
