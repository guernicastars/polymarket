# Research: Existing Polymarket User Tracking Tools & Projects

## Executive Summary

The Polymarket ecosystem has 170+ third-party tools across 19 categories. User/whale tracking is one of the most active segments, with dozens of commercial products, open-source repos, Telegram bots, and Twitter/X accounts dedicated to monitoring wallets, detecting insider activity, and enabling copy trading. Most tools rely on the same public Polymarket APIs (Gamma, CLOB, Data) and Polygon blockchain data.

---

## 1. Open-Source GitHub Projects

### polymarket-insider-tracker (pselamy/polymarket-insider-tracker)
- **URL**: https://github.com/pselamy/polymarket-insider-tracker
- **What it does**: Detects potential insider trading on Polymarket by tracking suspicious wallet behavior patterns -- fresh wallets, unusual position sizing, niche market entries, and funding chain analysis.
- **Tech stack**: Python 3.11+, PostgreSQL, Redis, Docker, SQLAlchemy, Alembic
- **Data sources**: Polymarket CLOB API, Polygon RPC endpoints (Web3)
- **Detection signals**: Fresh wallets (<5 lifetime txns making large trades), unusual sizing (>2% of order book), niche market activity, funding chain tracing (wallet linkage via DBSCAN clustering)
- **Alert channels**: Discord, Telegram, email
- **Relevance**: Most sophisticated open-source insider detection. Good reference for anomaly scoring and wallet profiling.

### polymaster (neur0map/polymaster)
- **URL**: https://github.com/neur0map/polymaster
- **What it does**: Monitors large transactions on Polymarket and Kalshi with anomaly detection. CLI-based whale alerting.
- **Tech stack**: Rust (Tokio async runtime, reqwest, tokio-tungstenite, rusqlite, Clap CLI)
- **Data sources**: Polymarket Data API (trades, positions, leaderboard), Gamma API (market context), CLOB API (order book depth). All public, no API keys needed.
- **Key features**:
  - Configurable threshold (default $25K)
  - Portfolio tracking (total value, profit metrics)
  - Leaderboard integration (checks if trader is top 500)
  - 12-hour wallet memory (detects returning whales)
  - Win rate and trading history analysis
  - Order book depth visualization
  - Top 5 holders per market with concentration data
  - Webhook support (Discord, Slack, n8n)
  - Audio alerts with elevated alerts for repeat actors
- **Relevance**: Very complete whale monitoring tool. Good reference for what data is available from public APIs.

### polymarket-trade-tracker (leolopez007/polymarket-trade-tracker)
- **URL**: https://github.com/leolopez007/polymarket-trade-tracker
- **What it does**: Free web app for analyzing any wallet's Polymarket trading activity. Input wallet address + market URL, get detailed PnL, maker/taker analysis, and charts.
- **Tech stack**: Flask (Python), HTML/Tailwind/JS, Matplotlib
- **Data sources**: Gamma API, CLOB API, Data API, Polygon RPC
- **Key features**:
  - Realized & unrealized PnL calculation
  - Maker/taker role identification (via on-chain tx receipts)
  - Splits, merges, and redemption detection
  - Price movement and cumulative PnL charts
  - JSON export
  - Single-market and batch multi-market analysis
- **Limitation**: Uses free public RPC endpoints; slow for wallets with 500+ trades (5-10 min).
- **Relevance**: Best open-source reference for per-wallet trade analysis and PnL computation.

### polymarket--whale-trading-bot (iengineer/polymarket--whale-trading-bot)
- **URL**: https://github.com/iengineer/polymarket--whale-trading-bot
- **What it does**: Automated whale tracking + proportional copy trading. Monitors successful traders and copies their trades in real-time.
- **Features**: Proportional position sizing, auto take-profit, trailing stop-loss

### polymarket-copy-trading-bot-version-3 (Trust412)
- **URL**: https://github.com/Trust412/polymarket-copy-trading-bot-version-3
- **What it does**: Mirrors positions of a target wallet address on Polymarket. Continuous monitoring with customizable risk management.

### polymarket-arbitrage-bot (runesatsdev)
- **URL**: https://github.com/runesatsdev/polymarket-arbitrage-bot
- **What it does**: Detects arbitrage opportunities including single-condition, NegRisk rebalancing, and whale tracking modes.

### polyterm (NYTEMODEONLY/polyterm)
- **URL**: https://github.com/NYTEMODEONLY/polyterm
- **What it does**: Polymarket in the terminal. Tracks market shifts, whale activity, insider patterns, and arbitrage opportunities.

### Awesome-Prediction-Market-Tools (aarora4)
- **URL**: https://github.com/aarora4/Awesome-Prediction-Market-Tools
- **What it does**: Curated directory of 170+ prediction market tools across AI agents, analytics, APIs, dashboards, copy trading, alerting, and tracking. Best meta-resource for the ecosystem.

---

## 2. Commercial Whale Tracking Platforms

### Polywhaler (polywhaler.com)
- Self-described "#1 Polymarket whale tracker"
- Monitors $10K+ trades in real-time
- Insider activity detection with AI-powered predictions
- Market sentiment analysis
- Premium product (pricing not public)

### PolyWatch (polywatch.tech)
- Free Polymarket whale tracker
- Instant Telegram alerts when whales trade
- No subscription/fees
- Good entry-level free alternative

### PolyInsider (polyinsider.io)
- Real-time dashboard for whale trades and insider activity
- Fresh wallet monitoring ($5K+ first-time bets)
- Customizable interface themes
- Comprehensive market analytics

### Whale Tracker Livid (whale-tracker-livid.vercel.app)
- Monitors traders with $50K+ portfolios
- Tiered alert system: free (1-hour delay) vs Pro ($29/month, real-time)
- Leaderboard rankings

### PolyTrack (polytrack.cash)
- Real-time insider trading detection
- Severity scoring system
- Paid: $9.99/week

### Polymarket Bros (brosonpm.trade)
- Community-focused whale tracking + copy trading
- Monitors trades over $4,000 in real-time
- One-click trade replication
- Verified whale badges, win rate display
- Free forever

### Stand (stand.trade)
- Professional prediction market trading terminal
- Whale alerts, copy trading, trader comparison
- Multi-market view (8 markets on 1 screen)
- Aggregates across prediction market platforms

### FirePolymarket (firepolymarket.com)
- Hot market movements
- Smart/Whale trader categorization

---

## 3. Telegram Bots

### PolyCopy (@PolyCopy_bot)
- Real-time monitoring of any trader's activity
- Instant notifications with trade details
- Complete portfolio visibility (open positions, closed trades, realized PnL, portfolio value)

### PolyTracker (@polytracker0_bot)
- Monitor specified wallet activities
- Real-time notifications on new transactions
- Market details and direct links

### Polylerts (@Polylerts_bot)
- Free: track up to 15 wallets
- Real-time trade alerts
- No subscription

### PolyIntel (@PolyIntel_bot)
- Detects whale movements and insider trading
- Automated alerts every 10 minutes
- Scoring systems, portfolio tracking, probability analysis

### Wincy Polymarket Bot (@wincy_polymarket_bot)
- Top 5 largest position holders per market
- Trader direction and whale movement analysis

### Polycool (polycool.live)
- Tracks smart traders from top 0.5% of wallets
- 24/7 Telegram notifications

### Predictify Bot (@PredictifyBot)
- Multi-market alert and AI assistant
- Copy trading support

### PolyData Trade Monitor Bot
- Free Telegram-based trade tracking
- Real-time notifications per wallet
- On-demand PnL reports and portfolio snapshots

---

## 4. Portfolio Analytics & Dashboards

### Polymarket Analytics (polymarketanalytics.com)
- Global analytics platform
- Custom portfolio creation (multi-wallet consolidation)
- PnL, positions, performance metrics
- Trader rankings and market analytics

### PredictFolio (predictfolio.com)
- Free analytics
- Track own portfolio + analyze other traders
- PnL, volume, win rate benchmarking vs top wallets

### Polysights (app.polysights.xyz)
- AI-powered analytics with 30+ custom metrics
- User portfolio tracking (PnL, trade history, performance)
- News insights

### PolymarketDash (polymarketdash.com)
- Professional trader analytics
- Real-time monitoring

### Parsec (parsec.fi/polymarket)
- Real-time flow, live trades, customizable dashboards
- Professional-grade analytics

### HashDive (hashdive.com)
- Proprietary "Smart Score" rating traders from -100 to 100
- Unique scoring methodology

### MobyScreener (mobyscreener.com)
- Live feed tracking top traders' buys and sells in real time

### PolyWallet (polywallet.info)
- Track up to 20 wallets with Telegram alerts
- Leaderboard integration

### Predicting Top (predicting.top)
- "Kolscan of Polymarket"
- Top traders' daily performance display

### Predicts.guru (predicts.guru)
- Advanced wallet statistics
- Visualization and strategy analysis

### PredScan (predscan.io)
- Daily PnL monitoring
- ROI tracking

---

## 5. Twitter/X Tracking Accounts

| Account | Handle | Focus |
|---------|--------|-------|
| PolyAlertHub | @PolyAlertHub | Comprehensive alerts + whale detection |
| WhaleWatch Poly | @whalewatchpoly | Large trade tracking |
| Polytrackerbot | @polytrackerbot | High-conviction whale activities (excludes sports) |
| PolyWhaleWatch | @polywhalewatch | Whale trade monitoring |
| PolyTale AI | @polytaleai | AI-powered research agent for prediction markets |

---

## 6. Blockchain Analytics Platforms (Multi-chain)

### Arkham Intelligence (intel.arkm.com)
- Entity de-anonymization (connects addresses to real entities)
- Supports Polygon (Polymarket's chain)
- Fund flow analysis and wallet linkage
- Tracks institutions, funds, and individuals
- Entity pages: https://intel.arkm.com/explorer/entity/polymarket

### Dune Analytics
- Community-built dashboards for Polymarket on-chain data
- Key dashboards:
  - `dune.com/rchen8/polymarket` -- General Polymarket overview
  - `dune.com/filarm/polymarket-activity` -- Activity and volume
  - `dune.com/petertherock/polymarket-on-polygon` -- On-chain Polygon data
  - `dune.com/andrew_wang/polymarket-user-analysis` -- User analysis
  - `dune.com/seoul/poly` -- Address tracker / airdrop eligibility
- SQL-based, can query any Polygon contract event

### Nansen (nansen.ai)
- Smart money tracking
- On-chain analytics with entity labeling

---

## 7. Key Polymarket APIs Used by These Tools

All the tools above rely on 3-4 public Polymarket APIs plus optional Polygon RPC:

### Gamma API (gamma-api.polymarket.com)
- `/events` -- Market metadata, volumes, tags, categories
- `/public-profile?address=0x...` -- User profile (username, bio, X handle, verified badge, proxy wallet)
- No auth required

### Data API (data-api.polymarket.com)
- `/positions?user=0x...` -- User's current positions (filterable by market/event)
- `/activity?user=0x...` -- User trade activity history
- `/trades?user=0x...` -- Trade history
- `/value?user=0x...` -- Total portfolio value
- `/v1/leaderboard` -- Trader rankings (by PNL or volume, filterable by category/time period)
  - Categories: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE
  - Time periods: DAY, WEEK, MONTH, ALL
  - Returns: rank, proxyWallet, userName, vol, pnl, profileImage, xUsername, verifiedBadge
- No auth for read-only access

### CLOB API (clob.polymarket.com)
- `/prices` -- Current prices for tokens
- `/books` -- Order book depth
- Auth required for trading; read-only endpoints are public

### Polygon RPC
- On-chain transaction history, wallet age, token transfers
- Used for maker/taker classification and funding trail analysis
- Free public RPCs available (slow) or paid (Alchemy, Infura)

---

## 8. Competitive Landscape Analysis

### What the market has plenty of:
- **Whale alert bots** -- Dozens of Telegram bots, all doing similar $10K+ trade monitoring
- **Copy trading bots** -- Multiple GitHub repos and commercial tools
- **Basic portfolio analytics** -- PnL, win rate, volume tracking per wallet
- **Leaderboard views** -- Several tools wrapping the public leaderboard API

### What's relatively underserved:
- **Cross-market user behavior analysis** -- Most tools track single-market or single-wallet. Few analyze user behavior patterns across all their positions.
- **Predictive signals from user data** -- HashDive's Smart Score is novel but most tools are reactive (alert after trade) not predictive.
- **Historical user analytics** -- Most tools are real-time only. Few build longitudinal profiles of trader behavior over weeks/months.
- **Cohort analysis** -- No tools segment users into meaningful groups (e.g., by strategy type, market category preference, trade frequency).
- **Integrated pipeline + dashboard** -- Most tools are standalone. A pipeline that continuously ingests and warehouses user data for analytics queries is differentiated.
- **Open-source full-stack solution** -- Individual components exist (tracker repos, bots) but no integrated open-source pipeline + analytics dashboard for user tracking.

### Differentiation opportunity for Polymarket Signals:
Our existing pipeline already ingests market data, prices, trades, and orderbooks into ClickHouse. Adding user-level tracking (positions, activity, leaderboard snapshots, profile data) would create a unique longitudinal dataset that most tools don't have. The dashboard could offer:
1. User profile pages with historical position/PnL evolution
2. Whale activity feed (configurable thresholds)
3. Smart money signals (what top-ranked traders are buying/selling)
4. Category-specific leaderboards with trend analysis
5. User cohort analytics (strategy classification)

This would be differentiated because it combines continuous data warehousing (not just real-time alerts) with analytical dashboards (not just Telegram notifications).

---

## 9. Ecosystem Directory

For the most comprehensive and up-to-date listing of all tools:
- **polymark.et** -- Curated Polymarket tools directory (web)
- **Awesome-Prediction-Market-Tools** (GitHub) -- Community-curated list
- **DeFiPrime guide** -- 170+ tools across 19 categories
