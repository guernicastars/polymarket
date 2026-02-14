# R3: Polymarket User Volumes and Trading Metrics

## 1. Official Polymarket User Data APIs

### Data API (`https://data-api.polymarket.com`)

The Data API is the primary source for per-user portfolio and trading metrics. All endpoints accept a user's wallet address (0x-prefixed, 40 hex chars).

#### GET /positions — User Portfolio Positions
- **Required**: `user` (wallet address)
- **Optional filters**: `market` (condition IDs, CSV), `eventId`, `title`, `sizeThreshold` (min size, default 1), `redeemable`, `mergeable`
- **Pagination**: `limit` (max 500), `offset` (max 10000)
- **Sort options**: CURRENT, INITIAL, TOKENS, CASHPNL, PERCENTPNL, TITLE, RESOLVING, PRICE, AVGPRICE
- **Response fields per position**:
  - `size` — number of shares held
  - `avgPrice` — average entry price
  - `initialValue` — cost basis
  - `currentValue` — mark-to-market value
  - `cashPnl` — realized P&L in USD
  - `percentPnl` — percent return
  - `realizedPnl` — realized profit/loss
  - `totalBought` — total shares purchased
  - `curPrice` — current market price
  - Market metadata: `conditionId`, `asset`, `title`, `slug`, `outcome`, `endDate`, `eventSlug`, `negativeRisk`

#### GET /trades — User Trade History
- **Optional**: `user`, `market` (condition ID, CSV), `side` (BUY/SELL), `takerOnly` (default true), `filterType` (CASH/TOKENS), `filterAmount`
- **Pagination**: `limit` (max 500), `offset`
- **Response**: Array of trades with side, asset, size, price, timestamp, transaction hash

#### GET /activity — Full On-Chain Activity
- **Required**: `user`
- **Types**: TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION (CSV-separated)
- **Filters**: `market`, `start`/`end` (Unix timestamps), `side` (BUY/SELL for trades only)
- **Sort**: TIMESTAMP, TOKENS, CASH (ASC/DESC)
- **Response**: Activity events with type, size, USD value, market metadata

#### GET /value — Total Portfolio Value
- **Required**: `user`
- **Optional**: `market` (CSV condition IDs)
- **Response**: Aggregated USD value of user's positions

#### GET /holders — Top Market Holders
- **Required**: `market` (condition ID)
- **Response**: Token holders with wallet, pseudonym, holdings amount, outcome index

### Leaderboard API (`https://data-api.polymarket.com/v1/leaderboard`)

#### GET /v1/leaderboard — Trader Rankings
- **Parameters**:
  - `category`: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, MENTIONS, WEATHER, ECONOMICS, TECH, FINANCE
  - `timePeriod`: DAY, WEEK, MONTH, ALL
  - `orderBy`: PNL or VOL
  - `limit`: 1-50 (default 25)
  - `offset`: 0-1000
  - `user`: specific wallet address (optional)
  - `userName`: search by display name (optional)
- **Response fields**:
  - `rank` — leaderboard position
  - `proxyWallet` — wallet address
  - `userName` — display name
  - `vol` — trading volume
  - `pnl` — profit/loss amount
  - `profileImage`, `xUsername`, `verifiedBadge`

### Undocumented / Semi-Documented Endpoints
The `polymarket-apis` Python library (v0.4.3) references additional Data API methods:
- `get_pnl_timeseries()` — PnL history by period/frequency (1d, 1w, 1m, all)
- `get_pnl()` — aggregated PnL/volume for time windows
- `get_leaderboard_rank()` — user's rank on profit/volume leaderboards
- `get_markets_traded_count()` — total markets traded by a user
- `get_closed_positions()` — historical closed positions
- `get_live_volume()` — event volume by event_id
- `get_open_interest()` — OI for condition IDs

These likely hit undocumented Data API endpoints (the official docs only formally document /positions, /trades, /activity, /value, /holders, /v1/leaderboard).

---

## 2. On-Chain Data via Subgraphs (Goldsky/The Graph)

Polymarket maintains 5 subgraphs on Goldsky, providing GraphQL access to on-chain data:

### Subgraph Endpoints
1. **Positions Subgraph**: `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn`
2. **PnL Subgraph**: `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn`
3. **Activity Subgraph**: (activity tracking)
4. **Orders Subgraph**: (order data)
5. **Open Interest Subgraph**: (OI metrics)

### Position Entity Schema (GraphQL)
```graphql
type Position {
  id: ID!
  condition: String!       # Market condition ID
  outcomeIndex: Int!       # 0 = YES, 1 = NO
  user: User!              # Wallet address
  balance: BigInt!          # Token balance
  averagePrice: BigDecimal! # Average entry price
  realizedPnl: BigDecimal!  # Realized profit/loss
}
```

### Key Characteristics
- Real-time indexing as blocks are mined on Polygon
- Reorg-aware (handles blockchain reorganizations)
- 99.9% uptime via Goldsky infrastructure
- Can be queried freely via GraphQL (no auth for public subgraphs)

### Example: Query User Positions
```graphql
{
  positions(where: { user: "0xABC..." }, first: 100) {
    condition
    outcomeIndex
    balance
    averagePrice
    realizedPnl
  }
}
```

---

## 3. Polymarket Website User-Facing Metrics

### Profile Page (`polymarket.com/profile/{username}`)
- Total PnL (profit/loss across all trades)
- Portfolio value (market value of all positions + cash)
- Open positions (active markets with money at stake)
- Trade history

### Portfolio Page (`polymarket.com/portfolio`)
- Portfolio Value: total market value of positions + cash
- Open Positions: market value of active positions
- Position details: entry price, current price, unrealized PnL

### Leaderboard (`polymarket.com/leaderboard`)
- Rankings by: Profit (PnL) or Volume
- Time periods: Day, Week, Month, All-time
- Categories: Overall, Politics, Sports, Crypto, Culture, etc.
- Shows: rank, username, PnL/volume figure, verified badge

---

## 4. Per-User Metrics Available (Summary)

### Directly Available via API
| Metric | Source | Endpoint |
|--------|--------|----------|
| Current positions (size, value) | Data API | GET /positions |
| Average entry price | Data API | GET /positions (avgPrice) |
| Realized PnL per position | Data API | GET /positions (cashPnl, realizedPnl) |
| Percent return per position | Data API | GET /positions (percentPnl) |
| Total portfolio value (USD) | Data API | GET /value |
| Trade history (with prices) | Data API | GET /trades |
| Full activity log | Data API | GET /activity |
| Leaderboard rank (PnL or Volume) | Data API | GET /v1/leaderboard |
| Trading volume (by time period) | Data API | GET /v1/leaderboard (vol field) |
| PnL timeseries | Data API | get_pnl_timeseries() (semi-documented) |
| Markets traded count | Data API | get_markets_traded_count() (semi-documented) |
| Closed/historical positions | Data API | get_closed_positions() (semi-documented) |

### Computable from API Data
| Metric | How to Compute |
|--------|----------------|
| Win rate | Count positions with cashPnl > 0 / total resolved positions |
| Average position size | Sum of position sizes / count |
| Portfolio concentration | Largest position value / total portfolio value |
| Trade frequency | Count trades per time period via /activity |
| Buy/sell ratio | Filter /trades by side, count each |
| Average holding period | Difference between entry and exit timestamps |
| Sector allocation | Group positions by category/tags |
| Risk exposure | Sum of position values in unresolved markets |

### Available via Third-Party Tools Only
| Metric | Source |
|--------|--------|
| Smart Score (-100 to 100) | Hashdive (proprietary algorithm) |
| Insider risk score (0-100) | Polywhaler (proprietary) |
| Wallet clustering (multi-wallet users) | Polywhaler, Dune dashboards |
| Portfolio scenario forecasting | PredictFolio |
| Benchmarking vs top wallets | PredictFolio |

---

## 5. Third-Party Analytics Platforms (User Volume Focus)

### Polymarket Analytics (polymarketanalytics.com)
- **Trader Leaderboard**: 1M+ traders tracked, updates every 5 minutes
- **Per-trader metrics**: Overall PnL, win rate, current holdings value, active positions, total wins/losses, total positions
- **Portfolio tracker**: Track any wallet's positions, PnL, and performance
- **Free tier** with most features

### PredictFolio (predictfolio.com)
- Real-time PnL tracking across positions
- Risk exposure analysis by category
- Position health metrics (entry prices, breakeven thresholds)
- Historical performance dashboard
- Portfolio scenario forecasting
- Benchmarking PnL/volume/win rate against top wallets

### Hashdive (hashdive.com)
- **Smart Score** (-100 to 100) per trader based on performance, open bets, consistency
- Market screener (filter by liquidity, volume, whale activity, momentum)
- Wallet lookup: positions, PnL history, unusual trades
- Covers both Polymarket and Kalshi

### Polywhaler (polywhaler.com)
- Real-time whale trade monitoring ($10K+ trades)
- Impact scoring (how trades affect markets)
- Insider risk detection (0-100 score)
- Market sentiment analysis
- **Pro ($10/mo)**: API access, historical trade analysis, portfolio tracking, wallet monitoring

### Predicting.top
- Live leaderboard tracking Polymarket + Kalshi traders
- Real-time PnL rankings across daily/weekly/monthly periods
- Wallet address visibility, profile views, join dates
- "Kolscan of prediction markets"

### Dune Analytics Dashboards
- **Polymarket Leaderboard** (dune.com/genejp999/polymarket-leaderboard): Custom SQL queries on on-chain data
- **Polymarket Activity & Volume** (dune.com/filarm/polymarket-activity): Volume, transactions, unique users
- **Polymarket CLOB Stats** (dune.com/lifewillbeokay/polymarket-clob-stats): Order book analytics
- **Main Polymarket Dashboard** (dune.com/rchen8/polymarket): Comprehensive metrics
- All Dune dashboards query Polygon blockchain data directly

### Unusual Whales (unusualwhales.com/predictions)
- Unusual trade detection on Polymarket
- Smart money movement tracking
- Cross-platform (also covers options, equities)

---

## 6. Python Libraries for User Data

### polymarket-apis (PyPI, v0.4.3)
```bash
pip install polymarket-apis  # Requires Python >=3.12
```
- **PolymarketDataClient** — unified interface for all user data:
  - `get_positions()`, `get_trades()`, `get_activity()`
  - `get_positions_value()`, `get_closed_positions()`
  - `get_pnl_timeseries()`, `get_pnl()`
  - `get_leaderboard_rank()`, `get_leaderboard()`
  - `get_markets_traded_count()`, `get_open_interest()`, `get_live_volume()`
- Pydantic data validation on all responses
- Also includes ClobClient, GammaClient, Web3Client, WebsocketsClient, GraphQLClient

### py-clob-client (Official Polymarket)
```bash
pip install py-clob-client
```
- Official CLOB client (trading-focused)
- Can retrieve user orders and trades
- Requires API key for authenticated endpoints

### Apify Scrapers
- **Polymarket Leaderboard Scraper** (apify.com/saswave/polymarket-leaderboard-scraper): Scrapes leaderboard data, filter by Profit/Volume, time ranges
- **Polymarket Markets Scraper** (apify.com/louisdeconinck/polymarket-events-scraper): Market/event data extraction

---

## 7. Key Observations for Building User Analytics

1. **Data API is the richest source**: The `/positions` endpoint alone gives per-position PnL, average price, current/initial values. Combined with `/trades` and `/activity`, you can reconstruct a user's complete trading history.

2. **Leaderboard API is limited**: Max 50 results per page, 1000 offset. Cannot enumerate all users — only top performers are visible. To track arbitrary users, you need their wallet address.

3. **Win rate is not a first-class field**: Must be computed from positions data (count positions with positive cashPnl vs total resolved positions).

4. **Volume is per-leaderboard-entry**: The leaderboard gives aggregate volume per time period, but per-market volume must be computed from trade history.

5. **On-chain subgraphs provide ground truth**: For accurate PnL and position data that doesn't depend on Polymarket's API availability, the Goldsky subgraphs offer direct blockchain data. The PnL subgraph specifically tracks averagePrice and realizedPnl per position.

6. **Rate limits are undocumented**: The Data API doesn't publicly document rate limits, but the polymarket-apis library handles pagination with sensible defaults (100-500 per page).

7. **Wallet address is the universal key**: All user-level queries require a Polygon wallet address (0x-prefixed). Display names are optional and only available via the leaderboard. There is no username-to-address lookup API — you must discover addresses through the leaderboard, on-chain activity, or third-party tools.

8. **Multi-wallet users exist**: The famous "French Whale" used multiple wallet addresses. Polywhaler and some Dune dashboards attempt wallet clustering, but this is non-trivial and imperfect.
