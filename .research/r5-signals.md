# Research: Polymarket User Signals & Analytics Opportunities

## Executive Summary

The Polymarket analytics ecosystem has exploded with 170+ tools, and ICE (Intercontinental Exchange) just launched its institutional "Polymarket Signals and Sentiment" product on Feb 11, 2026. The market is $44B+ in annual volume with $358M open interest. The most valuable signals for a Polymarket Signals product center on whale monitoring, smart money tracking, order flow analysis, and cross-market arbitrage detection. Our existing pipeline infrastructure (ClickHouse with market_prices, market_trades, orderbook_snapshots) positions us well to build many of these features without additional data sources.

---

## 1. Whale Monitoring & Large Order Detection

### What It Is
Real-time detection and alerting when large trades ($10K+) occur, tracking whale wallet positions, and monitoring for unusual trading patterns that precede market moves.

### Why It's Valuable
- Whale trades are the strongest directional signal — when multiple whales converge on a position, markets move
- Large order detection can signal insider knowledge (positions placed before news breaks)
- Whale consensus (2-3 top traders entering same direction) reduces false signal rate significantly

### Existing Competitors
| Tool | Features | Pricing |
|------|----------|---------|
| **Polywhaler** (polywhaler.com) | #1 whale tracker, $10K+ trades, AI predictions, insider detection | Freemium |
| **PolyTrack** (polytrackhq.app) | Alerts <30s, P&L history, cluster detection, curated whale lists | Subscription |
| **Polyburg** (polyburg.com) | Hundreds of profitable wallets tracked, smart money flows | Free |
| **LayerHub** (layerhub.xyz) | Whale + smart money tracking, large positions | Free |
| **WhaleWatch Poly** (mobyscreener.com) | Large trade monitoring | Free |

### Data Sources Required
- **Already have**: `market_trades` table (individual trades with side/size), `market_prices` (tick-level snapshots)
- **Need to add**: Polymarket Data API `GET /activity?user=<address>` for per-wallet trade history
- **Need to add**: Polymarket Data API `GET /positions?user=<address>` for current holdings
- **Need to add**: Polymarket Data API `GET /holders?market=<conditionId>` for top holders per market
- **Need to add**: On-chain monitoring via Polygon subgraph for real-time wallet-level trade detection

### Implementation Approach
1. **Trade size threshold detection**: Flag trades above configurable thresholds ($10K, $50K, $100K) from `market_trades`
2. **Wallet tracking table**: New `whale_wallets` table in ClickHouse tracking known profitable wallets
3. **Position change detection**: Poll Data API positions endpoint every 4-10 seconds for tracked wallets
4. **Alert scoring**: Score alerts by trade size, wallet profitability, order book imbalance, position type

---

## 2. Smart Money Tracking & Copy Trading Signals

### What It Is
Identifying consistently profitable wallets, tracking their positions in real-time, and generating copy-trading signals when smart money enters markets.

### Why It's Valuable
- Polymarket leaderboard API exposes top traders by PnL across multiple time windows
- Smart money consensus is the highest-confidence signal available
- Copy trading is the most popular consumer feature in the ecosystem
- PolyTrack's copy trading feature is their primary selling point

### Existing Competitors
| Tool | Features |
|------|----------|
| **Stand.trade** | Free terminal, one-click copy trading, real-time feeds |
| **polyHFT** (polyhft.com) | Copy specific high-profit traders |
| **PolyTrack** | Copy trading with P&L tracking, win rate data |
| **Polytrackerbot** | Twitter bot surfacing whale activity |

### Data Sources Required
- **Leaderboard API**: `GET https://data-api.polymarket.com/v1/leaderboard` — top traders by PnL/volume, filterable by category (POLITICS, SPORTS, CRYPTO, etc.) and time period (DAY, WEEK, MONTH, ALL)
  - Returns: rank, proxyWallet, userName, vol, pnl, profileImage, xUsername, verifiedBadge
  - Limit 1-50 per request, offset 0-1000
- **User Positions**: `GET https://data-api.polymarket.com/positions?user=<address>` — current holdings
- **User Activity**: `GET https://data-api.polymarket.com/activity?user=<address>` — trade history with type (TRADE, SPLIT, MERGE, REDEEM), timestamps, sizes, prices
- **User Profiles**: `GET https://data-api.polymarket.com/profiles?address=<address>` — pseudonym, bio, profile image

### Implementation Approach
1. **Leaderboard scraping job**: Poll leaderboard every hour across all categories/time periods, store in `trader_rankings` table
2. **Smart wallet watchlist**: Auto-populate from top 100 PnL leaders + manually curated
3. **Position polling**: Track position changes for watchlisted wallets every 10-30 seconds
4. **Signal generation**: Emit signal when smart money enters a market, weighted by trader rank and conviction size
5. **Performance tracking**: Track signal accuracy over time (hit rate, avg return)

---

## 3. Wallet Clustering & Insider Detection

### What It Is
Identifying groups of wallets controlled by the same entity using on-chain analysis, detecting potential insider trading through abnormal pre-announcement activity.

### Why It's Valuable
- Sophisticated actors use multiple wallets to disguise large positions
- Cluster detection reveals true position sizes and conviction levels
- Insider detection is high-value for market integrity and alpha generation
- PolyTrack is currently the ONLY tool with cluster detection — significant moat opportunity

### Detection Techniques
1. **Transaction timing analysis**: Wallets trading within seconds/minutes of each other
2. **Common counterparties**: Wallets frequently interacting with same addresses
3. **Gas fee patterns**: Same gas price settings linking wallets to same controller
4. **Funding source analysis**: Wallets funded with round numbers from same source
5. **DBSCAN clustering**: Unsupervised clustering on behavioral features (as used by `polymarket-insider-tracker`)
6. **Synchronized trading**: Correlated entry/exit timing across wallets

### Insider Trading Indicators
- Abnormal volume 3-5x baseline before announcements
- Fresh wallets placing large single-market bets
- Unusual win rates in niche/low-liquidity markets
- Position concentration in single event outcomes

### Data Sources Required
- **On-chain**: Polygon subgraph (`OrderFilled` events from CTF Exchange at `0x4bfb41d5...bd8b8982e`)
- **The Graph**: `https://gateway.thegraph.com/api/{key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`
- **PolygonScan**: Transaction history, internal transactions, token transfers
- **Data API**: Activity endpoint with wallet filtering
- **Dune Analytics**: Pre-built dashboards for cross-referencing (dune.com/filarm/polymarket-activity, dune.com/rchen8/polymarket)

### Implementation Approach
1. **Wallet graph database**: Store wallet-to-wallet interactions (could use ClickHouse graph queries or dedicated Neo4j)
2. **Clustering pipeline**: Batch job computing wallet similarity scores using timing, funding, and behavior features
3. **Anomaly scoring**: Real-time scoring of trades against baseline volume/win-rate distributions
4. **Severity system**: Tiered alerts (low/medium/high/critical) based on composite anomaly score

---

## 4. Order Flow & Orderbook Analysis

### What It Is
Analyzing orderbook depth, imbalance, buy/sell walls, and cumulative volume delta to predict short-term price movements.

### Why It's Valuable
- Orderbook imbalance (OBI) is a strong short-term directional signal
- Buy/sell wall detection reveals market maker intentions and support/resistance levels
- Liquidity depth analysis helps identify thin markets vulnerable to large moves
- Our existing `orderbook_snapshots` table (top 100 markets, 60s intervals) is a unique dataset

### Key Metrics
1. **Order Book Imbalance (OBI)**: Bid volume vs ask volume ratio (e.g., 54% bid / 46% ask = moderate bid pressure)
2. **Buy/Sell Walls**: Large resting orders at specific price levels
3. **Liquidity Depth**: Total liquidity within 0.1%, 0.5%, 1.0% of mid-price
4. **Cumulative Volume Delta (CVD)**: Running total of buy vs sell market orders
5. **Volume Profile**: Distribution of volume by price level
6. **Spread Analysis**: Bid-ask spread as liquidity/efficiency indicator

### Data Sources Required
- **Already have**: `orderbook_snapshots` table (top 100 markets, L2 depth, 60s intervals, 7-day TTL)
- **Already have**: `market_trades` table (with buy/sell side)
- **Already have**: `market_prices` table (tick-level)
- **Enhancement**: Increase orderbook snapshot frequency for high-activity markets
- **Enhancement**: Compute OBI, CVD as materialized views

### Implementation Approach
1. **OBI materialized view**: Compute from orderbook_snapshots as `sum(bid_size) / (sum(bid_size) + sum(ask_size))`
2. **CVD tracking**: Cumulative sum of signed trade volume from market_trades
3. **Wall detection**: Identify price levels where resting order size > 3x average
4. **Liquidity heatmap**: Aggregate depth across time for visualization
5. **Signal generation**: OBI > 0.65 or < 0.35 = directional signal

---

## 5. Cross-Market Arbitrage & Mispricing Detection

### What It Is
Identifying logical inconsistencies between related markets (e.g., "Trump Wins" vs "Republican Wins") and price mismatches between YES/NO outcomes.

### Why It's Valuable
- Study of 86M trades found 7,000+ mispriced markets between Apr 2024 - Apr 2025
- Top 3 arbitrage wallets earned $4.2M profit from 10,200+ bets
- Total arbitrage profits ~$40M in that period
- Two types: single-market (YES+NO != $1) and combinatorial (cross-market logical inconsistencies)

### Arbitrage Types
1. **Single-Market Rebalancing**: YES price + NO price should = $1.00; deviations > fees = arb
2. **Combinatorial Arbitrage**: Logically linked events with inconsistent pricing
3. **Cross-Platform**: Price differences between Polymarket and Kalshi/other platforms
4. **Temporal Arbitrage**: Slow price adjustment after news events

### Data Sources Required
- **Already have**: `markets` table with prices, `market_prices` for tick data
- **Need**: Market relationship mapping (which markets are logically related)
- **Need**: External platform APIs (Kalshi, etc.) for cross-platform arb
- **Enhancement**: Event/category grouping from Gamma API tags for related market discovery

### Implementation Approach
1. **Sum-to-one checker**: Continuous validation that outcome prices sum to ~$1.00
2. **Relationship graph**: Map logically related markets using event grouping and NLP on market questions
3. **Spread monitoring**: Track deviations with configurable alerting thresholds
4. **Historical arb tracker**: Record all detected opportunities and whether they closed (calibration data)

---

## 6. Sentiment & News Correlation

### What It Is
Correlating prediction market price movements with news events, social media sentiment, and external data sources to generate alpha signals.

### Why It's Valuable
- ICE just launched institutional "Polymarket Signals and Sentiment" product (Feb 11, 2026) — validates the market
- Prediction markets reflect collective expectations on market-moving events in near real-time
- Polymarket achieved 95% accuracy on 2024 elections, outperforming traditional polling
- News-to-price latency creates tradeable windows

### ICE Polymarket Signals Product (Launched Feb 11, 2026)
- Exclusive institutional distribution via ICE Consolidated Feed (real-time) and ICE Consolidated History (backtesting)
- Maps Polymarket signals to specific securities via ICE entity identification databases
- Targets professional/institutional traders consuming crowd-sourced probability as signals
- Validates that this data has institutional-grade value

### Existing Competitors
| Tool | Features |
|------|----------|
| **Polysights** (polysights.xyz) | AI-powered, 30+ metrics, news insights, momentum/trend detection |
| **Mention Metrix** (mentionmetrix.com) | X/Twitter and news mention tracking for markets |
| **AI PolyMarket** (ai-polymarket.com) | "Bloomberg Terminal of Prediction Markets," 87% accuracy claim |
| **Tremor.live** | Unusual volatility/odds movement detection, momentum anomaly scoring |
| **YN Signals** (Telegram) | 24/7 prediction market alpha signals, odds anomalies |
| **PolySpyBot** (Telegram) | Early-stage market discovery, noise reduction |

### Data Sources Required
- **Already have**: `markets` table with price changes, volume metrics
- **Need**: News API integration (NewsAPI, GDELT, or similar)
- **Need**: Twitter/X API for social sentiment around market topics
- **Need**: RSS feeds for category-specific news sources
- **Enhancement**: Price-news correlation engine comparing timestamps

### Implementation Approach
1. **News ingestion pipeline**: Aggregate headlines from multiple sources, tag with market categories
2. **Sentiment scoring**: NLP-based sentiment classification of news/social content
3. **Correlation engine**: Match news events to market price movements within time windows
4. **Leading indicator detection**: Identify when news leads vs lags market prices
5. **Signal dashboard**: Show markets where news sentiment diverges from current price

---

## 7. Position Concentration & Risk Analytics

### What It Is
Analyzing how concentrated positions are across wallets and markets, identifying risk hotspots and potential market manipulation.

### Why It's Valuable
- High concentration = fragile market (one whale exit can crash price)
- Position concentration signals conviction level and potential manipulation
- Risk metrics help traders size positions appropriately

### Key Metrics
1. **Herfindahl-Hirschman Index (HHI)**: Concentration of holdings across wallets per market
2. **Top-N holder share**: What % of open interest is held by top 5/10/20 wallets
3. **Single-wallet exposure**: Maximum position as % of total open interest
4. **Whale-to-retail ratio**: Large positions vs small positions distribution
5. **Market depth ratio**: Open interest vs orderbook depth (how easily could positions be unwound)

### Data Sources Required
- **Need**: `GET /holders?market=<conditionId>` — top 20 holders per market
- **Need**: Open interest endpoint: market-level open interest data
- **Already have**: `orderbook_snapshots` for depth analysis
- **Already have**: `markets` table with volume metrics

### Implementation Approach
1. **Holder tracking job**: Poll top holders for active markets, store in `market_holders` table
2. **Concentration metrics**: Compute HHI, top-N share as materialized views
3. **Risk scoring**: Composite risk score per market combining concentration, liquidity, volatility
4. **Fragility alerts**: Flag markets where single wallet controls >20% of open interest

---

## 8. Technical / Quantitative Signals

### What It Is
Traditional technical analysis and quantitative metrics adapted for prediction markets.

### Why It's Valuable
- Prediction markets exhibit momentum, mean-reversion, and volatility patterns similar to financial markets
- Quantitative signals can be backtested against historical data
- Combined with fundamental signals (whale activity, news), creates multi-factor model

### Key Metrics (from existing data)
1. **RSI (Relative Strength Index)**: Overbought/oversold from OHLCV candles
2. **Bollinger Bands**: Volatility bands from price history
3. **VWAP**: Volume-weighted average price from trade data
4. **Momentum**: Rate of price change over configurable windows
5. **Volatility regime detection**: High-vol vs low-vol periods
6. **Volume anomaly detection**: Volume spikes relative to historical baseline

### Data Sources Required
- **Already have**: `ohlcv_1m`, `ohlcv_1h` materialized views (candle data)
- **Already have**: `volume_daily` (daily volume rollups with VWAP)
- **Already have**: `market_prices` (tick-level price history)
- **Already have**: `market_trades` (individual trades)

### Implementation Approach
1. **Technical indicator views**: Compute RSI, Bollinger, momentum as ClickHouse queries on OHLCV data
2. **Signal scoring**: Combine multiple indicators into composite signal strength
3. **Backtest framework**: Use historical data to validate signal accuracy
4. **Dashboard integration**: Add technical chart overlays to existing price charts

---

## Priority Ranking for Polymarket Signals Product

### Tier 1 — High Value, Buildable Now (use existing pipeline data)
| Signal | Data Ready? | Effort | Differentiation |
|--------|-------------|--------|-----------------|
| **Orderbook imbalance / OBI** | Yes (orderbook_snapshots) | Low | Medium — few tools do this |
| **Volume anomaly detection** | Yes (market_trades, OHLCV) | Low | Low — common feature |
| **Technical signals (RSI, VWAP)** | Yes (OHLCV views) | Low | Low — but foundational |
| **Large trade detection** | Yes (market_trades) | Low | Medium — threshold alerting |

### Tier 2 — High Value, Needs New Data Sources
| Signal | Data Needed | Effort | Differentiation |
|--------|-------------|--------|-----------------|
| **Whale wallet tracking** | Data API positions/activity | Medium | Medium — crowded space |
| **Smart money tracking** | Leaderboard API + position polling | Medium | Medium |
| **Position concentration** | Holders API + OI | Medium | High — few tools do this well |
| **Cross-market arbitrage** | Market relationship mapping | Medium | High — few automated tools |

### Tier 3 — Highest Value, Highest Effort
| Signal | Data Needed | Effort | Differentiation |
|--------|-------------|--------|-----------------|
| **Wallet clustering** | On-chain subgraph + graph analysis | High | Very High — only PolyTrack does this |
| **Insider detection** | Clustering + anomaly scoring | High | Very High |
| **News/sentiment correlation** | External news/social APIs | High | Medium — ICE doing this institutionally |
| **Copy trading signals** | Full wallet tracking infra | High | Low — very crowded |

---

## Recommended Roadmap

### Phase 1: Leverage Existing Data (1-2 weeks)
- Orderbook imbalance signals from `orderbook_snapshots`
- Volume anomaly detection from `market_trades` and OHLCV views
- Large trade alerts (configurable thresholds) from `market_trades`
- Basic technical indicators (RSI, momentum) from OHLCV views
- Dashboard: new "Signals" page with ranked signal cards

### Phase 2: Add User/Wallet Data (2-4 weeks)
- New pipeline job: Leaderboard scraper (hourly, all categories)
- New pipeline job: Top holder tracker (per active market)
- New tables: `trader_rankings`, `market_holders`, `wallet_positions`
- Smart money signal: alert when top-ranked traders enter markets
- Position concentration metrics per market
- Dashboard: "Whales" page with wallet profiles and position tracking

### Phase 3: Advanced Analytics (4-8 weeks)
- Cross-market arbitrage detection engine
- Wallet clustering via on-chain analysis (Polygon subgraph)
- Insider trading anomaly scoring
- News/sentiment correlation (if external API budget available)
- Multi-factor composite signal combining all sources

---

## Key API Endpoints Reference

### Data API (https://data-api.polymarket.com)
| Endpoint | Purpose | Auth? |
|----------|---------|-------|
| `GET /v1/leaderboard` | Trader rankings by PnL/volume | No |
| `GET /positions?user=<addr>` | Current positions for wallet | No |
| `GET /activity?user=<addr>` | Trade history for wallet | No |
| `GET /holders?market=<condId>` | Top 20 holders per market | No |
| `GET /trades?user=<addr>` | Trades by user or market | No |
| `GET /profiles?address=<addr>` | User profile info | No |

### On-Chain (Polygon)
| Source | Purpose |
|--------|---------|
| CTF Exchange (`0x4bfb41d5...`) | OrderFilled events for all trades |
| NegRisk CTF Exchange | Matched orders for negative-risk markets |
| Polymarket Subgraph (The Graph) | Indexed on-chain trade/volume/user data |
| PolygonScan | Transaction history, internal txns |

### External Analytics
| Source | Purpose |
|--------|---------|
| Dune Analytics | SQL-queryable on-chain data, pre-built dashboards |
| Bitquery | GraphQL API for CTF Exchange events |
| Goldsky | Polymarket Analytics data infrastructure |

---

## Competitive Landscape Summary

The analytics ecosystem is crowded (170+ tools) but mostly fragmented — most tools do one thing (whale alerts, copy trading, dashboards). The opportunities for differentiation are:

1. **Integrated multi-signal product**: No single tool combines orderbook analysis + whale tracking + technical signals + arbitrage detection
2. **Wallet clustering / insider detection**: Only PolyTrack offers cluster detection; high moat
3. **Quantitative backtesting**: ICE is the only player offering historical time-series for backtesting; an accessible version for retail would be valuable
4. **Real-time composite scoring**: Combining multiple signal types into a single actionable score per market

The biggest validation is ICE's institutional product launch — it proves prediction market signals have value for professional traders. Our pipeline already captures the foundational data (prices, trades, orderbooks, volumes). The main gap is wallet-level data, which requires adding Data API polling jobs to the pipeline.
