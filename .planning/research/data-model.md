# Polymarket Data Model & Available Data

## 1. Data Hierarchy: Events -> Markets -> Outcomes -> Tokens

Polymarket organizes data in a clear hierarchy:

```
Event (e.g., "2026 US Midterms")
  |-- slug: "2026-us-midterms"
  |-- id: integer (e.g., 16085)
  |
  +-- Market (e.g., "Will Democrats win the Senate?")
  |     |-- conditionId: 0x8e9b... (hex hash, CTF condition)
  |     |-- slug: "will-democrats-win-senate"
  |     |-- questionID: bytes32 (hash of UMA ancillary data)
  |     |
  |     +-- Outcome "Yes" --> Token (clobTokenIds[0])
  |     |     |-- ERC1155 position ID on Polygon
  |     |     |-- Price = implied probability (e.g., 0.65)
  |     |
  |     +-- Outcome "No"  --> Token (clobTokenIds[1])
  |           |-- ERC1155 position ID on Polygon
  |           |-- Price = 1 - Yes price (e.g., 0.35)
  |
  +-- Market (e.g., "Will Republicans win the Senate?")
        |-- (same structure)
```

### Event Types
- **Single Market Event (SMP)**: 1 event = 1 market (simple binary question)
- **Group Market Event (GMP)**: 1 event = multiple markets (e.g., "How many rate cuts?" with separate markets for 0, 1, 2, 3+)

### Key Identifiers and Their Relationships

| Identifier | Format | Where Used | Purpose |
|---|---|---|---|
| Event ID | Integer | Gamma API | Organizational grouping |
| Event slug | String | Gamma API, URLs | Human-readable event identifier |
| Condition ID | Hex hash (0x...) | Gamma API, CLOB API, Data API, on-chain | Unique market identifier; also the CTF condition ID |
| Market slug | String | Gamma API, URLs | Human-readable market identifier |
| Token ID (clobTokenId) | Numeric string | CLOB API | Identifies a specific outcome token for trading |
| Question ID | bytes32 | On-chain (CTF) | Hash of UMA ancillary data |
| Position ID | bytes32 | On-chain (ERC1155) | Derived from collateral + collection ID |

### How IDs Connect

```
conditionId = getConditionId(oracle, questionId, outcomeSlotCount=2)
collectionId = getCollectionId(parentCollectionId=0, conditionId, indexSet)
  - indexSet=1 (0b01) -> Outcome "Yes"
  - indexSet=2 (0b10) -> Outcome "No"
positionId = getPositionId(USDC_address, collectionId)
  -> This becomes the ERC1155 token ID
```

The `clobTokenIds` array on the market object maps directly to these ERC1155 position IDs. Index 0 = first outcome, Index 1 = second outcome. The `outcomes` and `outcomePrices` arrays use the same indexing.

---

## 2. API Architecture

Polymarket exposes three REST APIs and two WebSocket services:

### REST APIs

| Service | Base URL | Auth Required | Purpose |
|---|---|---|---|
| Gamma API | `https://gamma-api.polymarket.com` | No | Market discovery, metadata, events, volume/liquidity |
| CLOB API | `https://clob.polymarket.com` | Read: No, Write: Yes | Orderbook, pricing, trading |
| Data API | `https://data-api.polymarket.com` | No (wallet address as param) | User positions, trades, activity |

### WebSocket Services

| Service | URL | Auth | Purpose |
|---|---|---|---|
| CLOB WS | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | No (market), Yes (user) | Real-time orderbook, price changes, trades (~100ms latency) |
| RTDS | `wss://ws-live-data.polymarket.com` | Optional (gamma_auth) | Comments, crypto prices |

### Rate Limits
Not explicitly documented. No API key required for read operations. Fees are currently 0% for both makers and takers.

---

## 3. Pricing Data

### Current Price
- **Endpoint**: `GET https://clob.polymarket.com/price?token_id={id}&side={BUY|SELL}`
- **Returns**: `{"price": "0.65"}` -- the best available price for the given side
- **Note**: More reliable than /book for real-time data (known staleness issue with /book)

### Midpoint
- **Endpoint**: `GET https://clob.polymarket.com/midpoint?token_id={id}`
- **Returns**: `{"mid": "0.64"}` -- average of best bid and best ask
- **Use**: Reference "market price" when spread is tight

### Spread
- **Endpoint**: `GET https://clob.polymarket.com/spread?token_id={id}`
- **Returns**: Bid-ask spread object

### Last Trade Price
- **Endpoint**: `GET https://clob.polymarket.com/last_trade_price?token_id={id}`
- **Returns**: Price, side, and size of most recent trade

### Outcome Prices (from Gamma API)
- Market objects include `outcomePrices` array (e.g., `["0.52", "0.48"]`)
- Maps 1:1 with `outcomes` array (e.g., `["Yes", "No"]`)
- Prices represent implied probabilities and sum to ~1.0

### Gamma Market Price Fields
Each market object from the Gamma API includes:
- `lastTradePrice` -- most recent trade price
- `bestBid` -- current best bid
- `bestAsk` -- current best ask
- `spread` -- current bid-ask spread
- `outcomePrices` -- array of current prices per outcome
- `oneDayPriceChange`, `oneHourPriceChange`, `oneWeekPriceChange`, `oneMonthPriceChange`, `oneYearPriceChange` -- pre-computed price deltas

---

## 4. Historical Price Data (Candles/OHLCV)

### Price History
- **Endpoint**: `GET https://clob.polymarket.com/prices-history?market={token_id}`
- **Returns**: Array of `{t: timestamp, p: price}` pairs
- **NOT** OHLCV candles -- just timestamp/price series

### Parameters
| Parameter | Type | Description |
|---|---|---|
| `market` | string (required) | CLOB token ID |
| `startTs` | unix timestamp | Custom start (UTC) |
| `endTs` | unix timestamp | Custom end (UTC) |
| `interval` | enum | `1m`, `1w`, `1d`, `6h`, `1h`, `max` (mutually exclusive with startTs/endTs) |
| `fidelity` | integer | Data resolution in minutes |

### Limitations
- Only price data (no volume, no OHLCV)
- No documented maximum lookback period
- No documented rate limits
- Must construct OHLCV candles yourself from trade data if needed

### Signal Opportunity
Since native OHLCV isn't provided, building candles from trade events or price snapshots creates derived data that most participants don't have readily available.

---

## 5. Volume Data

### From Gamma API (Market Object)
| Field | Description |
|---|---|
| `volume` | Total all-time volume |
| `volumeNum` | Numeric version of volume |
| `volume24hr` | Last 24 hours |
| `volume1wk` | Last 7 days |
| `volume1mo` | Last 30 days |
| `volume1yr` | Last 365 days |
| `volumeClob` | Total CLOB volume |
| `volume24hrClob` | 24h CLOB volume |
| `volume1wkClob` | 7d CLOB volume |
| `volume1moClob` | 30d CLOB volume |
| `volume1yrClob` | 1y CLOB volume |

### From Gamma API (Event Object)
Events aggregate volume across their child markets:
- `volume`, `volume24hr`, `volume1wk`, `volume1mo`, `volume1yr`

### Volume Data Quality Warning
Research (Columbia study, Nov 2025) found ~25% of Polymarket volume may be artificial/wash trading. Volume spikes should be cross-referenced with price movement and unique trader counts for validation.

---

## 6. Liquidity / Order Book Depth

### Order Book Snapshot
- **Endpoint**: `GET https://clob.polymarket.com/book?token_id={id}`
- **Returns**:
  ```json
  {
    "market": "0x...",
    "asset_id": "...",
    "timestamp": "2026-01-15T12:00:00Z",
    "hash": "...",
    "bids": [{"price": "0.64", "size": "500"}, ...],
    "asks": [{"price": "0.66", "size": "300"}, ...],
    "min_order_size": "5",
    "tick_size": "0.01",
    "neg_risk": false
  }
  ```
- **Note**: Known staleness issue -- `/book` may serve stale snapshots. Use `/price` for reliable best bid/ask, and WebSocket for real-time depth.

### Liquidity Metrics (from Gamma API)
| Field | Description |
|---|---|
| `liquidity` | Total liquidity across all sources |
| `liquidityNum` | Numeric version |
| `liquidityClob` | CLOB-specific liquidity |
| `liquidityAmm` | AMM-specific liquidity (legacy) |

### Market Configuration
| Field | Description |
|---|---|
| `orderPriceMinTickSize` | Minimum price increment |
| `orderMinSize` | Minimum order size |
| `enableOrderBook` | Whether CLOB trading is available |
| `acceptingOrders` | Whether orders are currently accepted |
| `rfqEnabled` | Request-for-quote enabled |
| `rewardsMinSize` | Minimum size for market-making rewards |
| `rewardsMaxSpread` | Maximum spread for market-making rewards |

### Real-Time Depth via WebSocket
The market WebSocket channel streams `book` events with full orderbook snapshots and incremental updates. Recommended to maintain a local orderbook copy with sequence number tracking.

---

## 7. Open Interest

### From Gamma API (Event Object)
- `openInterest` (number) -- available on event objects
- Represents total value of outstanding positions

### Limitations
- Open interest is at the event level, not per-market
- No historical open interest endpoint documented
- Must snapshot over time to build OI timeseries

---

## 8. User Activity / Trade Data (Publicly Available)

### Trades Endpoint
- **URL**: `GET https://data-api.polymarket.com/trades`
- **Parameters**: `limit` (max 500), `offset`, `market` (conditionId), `user` (wallet), `side` (BUY/SELL), `takerOnly` (default: true), `filterType` (CASH/TOKENS), `filterAmount`
- **Returns per trade**: proxyWallet, side, asset, conditionId, size, price, timestamp, title, slug, outcome, outcomeIndex, name, pseudonym, transactionHash

### Activity Endpoint
- **URL**: `GET https://data-api.polymarket.com/activity?user={wallet}`
- **Types**: TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION
- **Parameters**: market, type, start/end timestamps, side, sortBy (TIMESTAMP/TOKENS/CASH)
- **Returns per activity**: proxyWallet, timestamp, conditionId, type, size, usdcSize, transactionHash, price, asset, side, outcomeIndex, title, slug, outcome

### Positions Endpoint
- **URL**: `GET https://data-api.polymarket.com/positions?user={wallet}`
- **Returns per position**: asset, conditionId, size, avgPrice, initialValue, currentValue, cashPnl, percentPnl, totalBought, realizedPnl, curPrice, redeemable, title, outcome, endDate, negativeRisk

### Holders Endpoint
- **URL**: `GET https://data-api.polymarket.com/holders?market={conditionId}`
- **Returns**: Top holders with wallet, pseudonym, amount, outcomeIndex

### Value Endpoint
- **URL**: `GET https://data-api.polymarket.com/value?user={wallet}`
- **Returns**: Total USD value of user's positions

---

## 9. On-Chain Data (Polygon)

### Contract Addresses
| Contract | Address | Purpose |
|---|---|---|
| Conditional Tokens (CTF) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | ERC1155 token factory, core market logic |
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Trading venue for outcome tokens |
| Neg Risk Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` | Negative risk market trading |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` | Adapter for neg risk markets |
| USDC.e | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Collateral token |

### On-Chain Operations
- **Condition setup**: Oracle registers new conditions
- **Token minting/splitting**: Collateral -> Yes + No tokens
- **Merging**: Yes + No tokens -> Collateral
- **Redemption**: Winning tokens -> Collateral (post-resolution)
- **Trading**: ERC1155 token transfers via CTF Exchange

### Oracle
Polymarket uses UMA's Optimistic Oracle (V2) for market resolution. The oracle address is used in condition ID derivation.

### On-Chain Data Availability
All trades, mints, splits, merges, and redemptions are recorded as Polygon transactions. This data can be queried via:
- PolygonScan API
- Subgraph (The Graph) -- community-built Polymarket subgraphs exist
- Bitquery API (provides structured Polymarket on-chain data)
- Direct RPC node queries

---

## 10. WebSocket Real-Time Data

### Market Channel Events
Subscribe to `wss://ws-subscriptions-clob.polymarket.com/ws/market` with token IDs.

| Event Type | Description | Key Fields |
|---|---|---|
| `book` | Full orderbook snapshot | bids[], asks[] |
| `price_change` | Best bid/ask price changed | asset_id, price changes |
| `tick_size_change` | Tick size updated | new tick size |
| `last_trade_price` | New trade executed | asset_id, price, side, size, timestamp, fee_rate_bps |
| `best_bid_ask` | Best bid/ask updated | bid, ask prices |
| `new_market` | New market listed | market details |
| `market_resolved` | Market resolved | resolution details |

### Best Practices
- Implement automatic reconnection with exponential backoff
- Send ping messages to maintain connection
- Maintain local orderbook copy with incremental updates
- Track sequence numbers to detect missed messages
- ~100ms latency for market data

---

## 11. Third-Party Data Aggregators & Tools

### Data Aggregators
| Tool | Description |
|---|---|
| [Polymarket Analytics](https://polymarketanalytics.com) | Dashboards and analytics for Polymarket data |
| [Oddpool](https://oddpool.com) | Cross-venue odds aggregation (Polymarket, Kalshi, CME) with arb detection |
| [Bitquery](https://docs.bitquery.io/docs/examples/polymarket-api/) | On-chain Polymarket data via GraphQL |
| [Forcazt](https://forcazt.com) | AI-powered prediction market analytics |

### Open Source Tools
| Repository | Description |
|---|---|
| [Polymarket/agents](https://github.com/Polymarket/agents) | Official AI trading agents |
| [Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client) | Official Python CLOB client |
| [warproxxx/poly_data](https://github.com/warproxxx/poly_data) | Comprehensive data pipeline for Polymarket |
| [Jon-Becker/prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) | Largest public dataset (Polymarket + Kalshi) |
| [SII-WANGZJ/Polymarket_data](https://github.com/SII-WANGZJ/Polymarket_data) | 1.1B trading records dataset |
| [PaulieB14/polymarket-subgraph-analytics](https://github.com/PaulieB14/polymarket-subgraph-analytics) | Subgraph-based analytics |
| [HuakunShen/polymarket-kit](https://github.com/HuakunShen/polymarket-kit) | Fully typed SDK with OpenAPI schema |
| [PredictionXBT/PredictOS](https://github.com/PredictionXBT/PredictOS) | Open-source prediction market framework |

---

## 12. Complete Gamma API Market Object Fields

Full field list from a live API response:

### Market-Level Fields
```
id, question, conditionId, slug, resolutionSource, endDate, startDate,
image, icon, description, outcomes, outcomePrices, clobTokenIds,

-- Volume --
volume, volumeNum, volume24hr, volume1wk, volume1mo, volume1yr,
volumeClob, volume24hrClob, volume1wkClob, volume1moClob, volume1yrClob,

-- Liquidity --
liquidity, liquidityNum, liquidityClob,

-- Pricing --
lastTradePrice, bestBid, bestAsk, spread,
oneDayPriceChange, oneHourPriceChange, oneWeekPriceChange,
oneMonthPriceChange, oneYearPriceChange,

-- Status --
active, closed, archived, new, featured, restricted, ready, funded,
acceptingOrders, acceptingOrdersTimestamp, approved,
enableOrderBook, negRisk, negRiskMarketID, negRiskRequestID,

-- Configuration --
orderPriceMinTickSize, orderMinSize, marketMakerAddress,
umaBond, umaReward, rewardsMinSize, rewardsMaxSpread,
rfqEnabled, holdingRewardsEnabled, feesEnabled, feeType,

-- Metadata --
createdAt, updatedAt, submitted_by, resolvedBy,
groupItemTitle, groupItemThreshold, questionID,
competitive, cyom, pagerDutyNotificationEnabled,
clearBookOnStart, automaticallyActive, manualActivation,
seriesColor, showGmpSeries, showGmpOutcome,
negRiskOther, umaResolutionStatuses,
pendingDeployment, deploying, requiresTranslation
```

### Nested Event Fields (within market)
```
id, ticker, slug, title, description, resolutionSource,
startDate, creationDate, endDate, image, icon,
active, closed, archived, new, featured, restricted,
liquidity, volume, openInterest, competitive,
volume24hr, volume1wk, volume1mo, volume1yr,
enableOrderBook, liquidityClob, negRisk, negRiskMarketID,
commentCount, cyom, showAllOutcomes, showMarketImages,
enableNegRisk, negRiskAugmented, cumulativeMarkets,
pendingDeployment, deploying, requiresTranslation,
seriesSlug, gmpChartMode,
series: { id, ticker, slug, title, seriesType, recurrence, ... },
createdAt, updatedAt
```

---

## 13. Signals Useful for Betting

### Price-Based Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Price momentum (1h, 1d, 1w change) | Gamma API (`oneDayPriceChange`, etc.) | Trend following / mean reversion |
| Price history divergence from fundamentals | CLOB `/prices-history` | Identify mispriced markets |
| Spread tightening/widening | CLOB `/spread` or WS `best_bid_ask` | Liquidity confidence signal |
| Midpoint vs. last trade | CLOB `/midpoint` vs `/last_trade_price` | Detect directional pressure |

### Volume-Based Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Volume spike detection | Gamma API (`volume24hr` vs `volume1wk/7`) | Unusual activity = information arrival |
| CLOB vs. total volume ratio | `volume24hrClob` / `volume24hr` | Organic vs. AMM activity split |
| Volume-weighted price momentum | Trades API + price data | Volume confirms directional moves |
| Volume relative to open interest | `volume24hr` / event `openInterest` | High turnover = active repricing |

### Liquidity Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Order book depth imbalance | CLOB `/book` or WS `book` events | More bids than asks = bullish pressure |
| Liquidity withdrawal | Gamma API `liquidity` over time | Market makers pulling out = uncertainty |
| Spread vs. historical average | CLOB `/spread` snapshots | Wide spread = low confidence or event risk |
| Reward spread threshold proximity | `rewardsMaxSpread` vs actual spread | Market maker incentive boundaries |

### Activity Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Whale position tracking | Data API `/positions` + `/holders` | Follow smart money |
| New large positions | Data API `/trades` filtered by size | Informed money entering |
| Holder concentration | Data API `/holders` | Few large holders = manipulation risk |
| Trade flow imbalance (buy vs sell) | Data API `/trades` with side filter | Net directional pressure |

### Structural Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Time to resolution | Market `endDate` | Convergence pressure as deadline approaches |
| Market creation recency | Market `createdAt` | New markets often mispriced |
| Neg risk flag | Market `negRisk` | Different risk/reward dynamics |
| Competitive flag | Market `competitive` | Higher activity expected |
| GMP outcome correlation | Multiple markets in same event | Arbitrage across related outcomes |

### Cross-Market Signals
| Signal | Data Source | Rationale |
|---|---|---|
| Correlated market divergence | Multiple Gamma API markets | Related markets should move together |
| Cross-platform arbitrage | Polymarket vs Kalshi/Metaculus | Price gaps across platforms |
| Event-level OI vs market-level volume | Event `openInterest` + market volumes | Capital allocation within events |

---

## 14. Data Collection Strategy for a Pipeline

### Recommended Polling Intervals
| Data Type | Source | Interval | Priority |
|---|---|---|---|
| Active markets list | Gamma `/events?active=true` | Every 5 minutes | High |
| Market metadata + volume | Gamma `/markets` | Every 1 minute | High |
| Price snapshots | CLOB `/price` per token | Every 30 seconds | Critical |
| Orderbook depth | CLOB `/book` per token | Every 1 minute | Medium |
| Historical prices | CLOB `/prices-history` | Every 1 hour (backfill) | Medium |
| Trades | Data API `/trades` | Every 1 minute | High |
| Top holders | Data API `/holders` | Every 15 minutes | Low |
| Open interest | Gamma `/events` (OI field) | Every 5 minutes | Medium |

### Real-Time via WebSocket (preferred for latency-sensitive signals)
- Subscribe to market channel for all active token IDs
- Process `last_trade_price`, `price_change`, `best_bid_ask`, `book` events
- Maintain local orderbook state
- Stream into pipeline for real-time signal generation

### Data Gaps to Be Aware Of
1. **No native OHLCV** -- must construct from price snapshots or trade data
2. **No historical open interest** -- must snapshot and store over time
3. **No historical orderbook** -- must snapshot and store depth over time
4. **Volume may be inflated** (~25% wash trading per Columbia study)
5. **`/book` endpoint staleness** -- prefer `/price` and WebSocket for real-time
6. **Pagination limits** -- max 500 results per request on Data API
7. **No documented rate limits** -- but aggressive polling may get throttled
