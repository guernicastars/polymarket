# Polymarket CLOB API Research

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Base URLs](#base-urls)
3. [Authentication](#authentication)
4. [Rate Limits](#rate-limits)
5. [CLOB API Endpoints](#clob-api-endpoints)
6. [Gamma API Endpoints](#gamma-api-endpoints)
7. [Data API Endpoints](#data-api-endpoints)
8. [WebSocket Endpoints](#websocket-endpoints)
9. [Pagination Patterns](#pagination-patterns)
10. [Client Libraries](#client-libraries)
11. [Data Model & Key Concepts](#data-model--key-concepts)

---

## Architecture Overview

Polymarket operates a **hybrid-decentralized Central Limit Order Book (CLOB)** with three distinct API layers:

| API | Purpose | Base URL |
|-----|---------|----------|
| **CLOB API** | Order book, trading, pricing | `https://clob.polymarket.com` |
| **Gamma API** | Market discovery, metadata, search | `https://gamma-api.polymarket.com` |
| **Data API** | User positions, trades, activity | `https://data-api.polymarket.com` |

**How it works:**
- Orders are EIP-712 signed structured data
- Assets: Binary Outcome Tokens (CTF ERC1155 / ERC20 PToken) traded against USDC (ERC20 collateral)
- Off-chain matching by operator, atomic on-chain settlement via Exchange contract on Polygon (chain ID 137)
- Operator cannot set prices or execute unauthorized trades (non-custodial)

**Fee Structure:** Currently 0 bps (both maker and taker).

---

## Base URLs

### REST APIs

| Service | URL |
|---------|-----|
| CLOB API | `https://clob.polymarket.com` |
| Gamma API | `https://gamma-api.polymarket.com` |
| Data API | `https://data-api.polymarket.com` |

### WebSocket

| Service | URL |
|---------|-----|
| CLOB WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/` |
| Real-Time Data (RTDS) | `wss://ws-live-data.polymarket.com` |

---

## Authentication

### Three Authentication Levels

| Level | Requirements | Capabilities |
|-------|-------------|--------------|
| **L0 (Public)** | None | Read market data, prices, order books |
| **L1 (Wallet Signature)** | Private key (EIP-712 signing) | Create/derive API credentials, sign orders |
| **L2 (API Key + HMAC)** | API key + secret + passphrase | Post orders, cancel orders, query trades/balances |

### L1 Authentication (Wallet Signature)

Used to create or derive API credentials.

**Required HTTP Headers:**

| Header | Value |
|--------|-------|
| `POLY_ADDRESS` | Signing wallet address |
| `POLY_SIGNATURE` | EIP-712 signature |
| `POLY_TIMESTAMP` | Current UNIX timestamp |
| `POLY_NONCE` | Nonce (default: 0) |

**EIP-712 Domain:**
```
Domain: ClobAuthDomain v1, Chain 137 (Polygon)
Message: "This message attests that I control the given wallet"
Fields: address, timestamp, nonce
```

**Key Derivation Endpoints:**
- `POST /auth/api-key` -- Create new API credentials (L1 auth)
- `GET /auth/derive-api-key` -- Retrieve existing credentials (L1 auth)

**Response:** Returns `{ apiKey, secret, passphrase }`.

### L2 Authentication (HMAC-SHA256)

Used for all trading operations.

**Required HTTP Headers:**

| Header | Value |
|--------|-------|
| `POLY_ADDRESS` | Signer address |
| `POLY_SIGNATURE` | HMAC-SHA256 signature (signed with `secret`) |
| `POLY_TIMESTAMP` | Current UNIX timestamp |
| `POLY_API_KEY` | User's apiKey |
| `POLY_PASSPHRASE` | User's passphrase |

### Signature Types

| Type | ID | Use Case |
|------|-----|----------|
| EOA | 0 | Standard MetaMask/external wallets |
| POLY_PROXY | 1 | Magic Link / Google login users |
| GNOSIS_SAFE | 2 | Multisig wallets, embedded wallets (recommended default) |

### Authentication Flow

```
1. Create temp client with private key
2. Call createOrDeriveApiKey() -> returns { apiKey, secret, passphrase }
3. Initialize trading client with credentials + signature type
4. Execute authenticated operations
```

---

## Rate Limits

Polymarket uses Cloudflare-based throttling with **sliding time windows**. Requests over the limit are **delayed/queued** (not dropped). Short bursts are permitted on some endpoints.

### General Endpoints

| Endpoint | Limit |
|----------|-------|
| Standard requests | 15,000 per 10 seconds |
| Health check (`/ok`) | 100 per 10 seconds |

### CLOB API

| Endpoint | Burst Limit | Sustained Limit |
|----------|------------|-----------------|
| General | 9,000 / 10s | -- |
| Market data (`/book`, `/price`, `/midprice`) | 1,500 / 10s | -- |
| `POST /order` | 3,500 / 10s | 36,000 / 10 min |
| Cancel operations | 250-1,000 / 10s | -- |

### Gamma API

| Endpoint | Limit |
|----------|-------|
| General | 4,000 / 10s |
| `/markets` | 300 / 10s |
| `/events` | 500 / 10s |
| Search | 350 / 10s |
| Comments / Tags | 200 / 10s each |

### Data API

| Endpoint | Limit |
|----------|-------|
| General | 1,000 / 10s |
| `/trades` | 200 / 10s |
| `/positions` | 150 / 10s |
| `/closed-positions` | 150 / 10s |

### Other

| Endpoint | Limit |
|----------|-------|
| User PNL API | 200 / 10s |
| RELAYER `/submit` | 25 / 1 min |

---

## CLOB API Endpoints

Base URL: `https://clob.polymarket.com`

### Health & Server (L0 -- No Auth)

#### `GET /ok`
Health check.
- **Response:** Service health confirmation string.

#### `GET /time`
Server time.
- **Response:** `number` (UNIX timestamp in seconds).

### Markets (L0 -- No Auth)

#### `GET /markets`
Paginated list of all markets.
- **Query Params:** `next_cursor` (string, optional)
- **Response:** `PaginationPayload`
```json
{
  "data": [Market],
  "next_cursor": "string",
  "count": 0
}
```

**Market Object Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `condition_id` | string | Market identifier (condition ID) |
| `question_id` | string | Question identifier |
| `tokens` | Token[] | Array of outcome tokens with `token_id` and `outcome` |
| `minimum_order_size` | string | Min order size |
| `minimum_tick_size` | string | Min price increment (`"0.1"`, `"0.01"`, `"0.001"`, `"0.0001"`) |
| `description` | string | Market description |
| `question` | string | Market question |
| `market_slug` | string | URL slug |
| `end_date_iso` | string | End date (ISO 8601) |
| `active` | boolean | Is actively trading |
| `closed` | boolean | Is closed |
| `archived` | boolean | Is archived |
| `accepting_orders` | boolean | Currently accepting orders |
| `accepting_order_timestamp` | string | When orders were first accepted |
| `neg_risk` | boolean | Negative risk enabled |
| `neg_risk_market_id` | string | Neg risk market ID |
| `neg_risk_request_id` | string | Neg risk request ID |
| `maker_base_fee` | number | Maker fee in bps |
| `taker_base_fee` | number | Taker fee in bps |
| `notifications_enabled` | boolean | Notifications on |
| `icon` | string | Icon URL |
| `image` | string | Image URL |
| `rewards` | object | Reward configuration |
| `tags` | string[] | Tags/categories |
| `is_50_50_outcome` | boolean | Binary 50/50 market |
| `fpmm` | string | Fixed-product market maker address |
| `seconds_delay` | number | Settlement delay |
| `enable_order_book` | boolean | CLOB enabled |
| `game_start_time` | string | Sports game start time |

#### `GET /simplified-markets`
Lightweight market data (fewer fields).
- **Query Params:** `next_cursor` (string, optional)
- **Response:** `PaginationPayload` with `SimplifiedMarket[]`.

#### `GET /sampling-markets`
Representative sample of markets.
- **Query Params:** `next_cursor` (string, optional)
- **Response:** `PaginationPayload` with `Market[]`.

#### `GET /sampling-simplified-markets`
Lightweight sampled markets.
- **Query Params:** `next_cursor` (string, optional)
- **Response:** `PaginationPayload` with `SimplifiedMarket[]`.

#### `GET /market/{condition_id}`
Single market by condition ID.
- **Path Params:** `condition_id` (string, required)
- **Response:** `Market` object.

### Order Book & Pricing (L0 -- No Auth)

#### `GET /book`
Order book for a single token.
- **Query Params:** `token_id` (string, required)
- **Response:** `OrderBookSummary`
```json
{
  "market": "0x1b6f76e5b8587ee896c35847e12d11e75290a8c3934c5952e8a9d6e4c6f03cfa",
  "asset_id": "1234567890",
  "timestamp": "2023-10-01T12:00:00Z",
  "hash": "0xabc123def456...",
  "bids": [{"price": "1800.50", "size": "10.5"}],
  "asks": [{"price": "1800.75", "size": "8.2"}],
  "min_order_size": "0.001",
  "tick_size": "0.01",
  "neg_risk": false
}
```

**OrderBookSummary Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `market` | string | Condition ID |
| `asset_id` | string | Token ID |
| `timestamp` | string | ISO 8601 snapshot timestamp |
| `hash` | string | Order book state hash |
| `bids` | OrderSummary[] | Bid levels `[{price, size}]` |
| `asks` | OrderSummary[] | Ask levels `[{price, size}]` |
| `min_order_size` | string | Minimum order size |
| `tick_size` | string | Minimum price increment |
| `neg_risk` | boolean | Negative risk flag |

**Errors:**
- `400`: `{"error": "Invalid token id"}`
- `404`: `{"error": "No orderbook exists for the requested token id"}`

#### `POST /books`
Multiple order books in a single request.
- **Body:** `[{"token_id": "..."}]` (array of token IDs)
- **Response:** `OrderBookSummary[]`

#### `GET /price`
Best price for a token on a given side.
- **Query Params:**
  - `token_id` (string, required)
  - `side` (string, required: `BUY` or `SELL`)
- **Response:**
```json
{"price": "0.65"}
```
**Errors:** `400` (invalid params), `404` (no orderbook), `500` (server error).

#### `POST /prices`
Multiple token prices.
- **Body:** `[{"token_id": "...", "side": "BUY"}]`
- **Response:**
```json
{
  "<token_id>": {"BUY": "0.65", "SELL": "0.35"}
}
```

#### `GET /midpoint`
Midpoint price (average of best bid and best ask).
- **Query Params:** `token_id` (string, required)
- **Response:** `{"mid": "0.50"}`

#### `POST /midpoints`
Multiple token midpoints.
- **Body:** `[{"token_id": "..."}]`
- **Response:** `{"<token_id>": "0.50"}`

#### `GET /spread`
Spread (best ask - best bid).
- **Query Params:** `token_id` (string, required)
- **Response:** `{"spread": "0.02"}`

#### `POST /spreads`
Multiple token spreads.
- **Body:** `[{"token_id": "..."}]`
- **Response:** `{"<token_id>": "0.02"}`

### Historical Pricing (L0 -- No Auth)

#### `GET /prices-history`
Historical price timeseries for a token.
- **Query Params:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `market` | string | Yes | CLOB token ID |
| `startTs` | number | No | Start time (UNIX UTC) |
| `endTs` | number | No | End time (UNIX UTC) |
| `interval` | string | No | Duration: `1m`, `1h`, `6h`, `1d`, `1w`, `max` (mutually exclusive with startTs/endTs) |
| `fidelity` | number | No | Resolution in minutes (e.g., 60) |

- **Response:** `PriceHistoryResponse`
```json
{
  "history": [
    {"t": 1697875200, "p": 0.65},
    {"t": 1697878800, "p": 0.67}
  ]
}
```

**Errors:** `400`, `404` (market not found), `500`.

### Trades (L0 for public, L2 for user-specific)

#### `GET /last-trade-price`
Most recent trade for a token.
- **Query Params:** `token_id` (string, required)
- **Response:** `{"price": "0.65", "side": "BUY"}`

#### `POST /last-trades-prices`
Batch last trade prices.
- **Body:** `[{"token_id": "..."}]`
- **Response:** `[{"price": "0.65", "side": "BUY", "token_id": "..."}]`

#### `GET /market-trades-events/{condition_id}`
Trade history for a market.
- **Path Params:** `condition_id` (string, required)
- **Response:** Array with event type, market details, user info, side, size, fee rate, price, outcome.

### Market Parameters (L0 -- No Auth)

#### `GET /fee-rate-bps`
Fee rate in basis points.
- **Query Params:** `token_id` (string, required)
- **Response:** `number` (e.g., `0`)

#### `GET /tick-size`
Minimum tick size for a token.
- **Query Params:** `token_id` (string, required)
- **Response:** `"0.1"` | `"0.01"` | `"0.001"` | `"0.0001"`

#### `GET /neg-risk`
Negative risk flag.
- **Query Params:** `token_id` (string, required)
- **Response:** `boolean`

### Orders (L2 -- Authenticated)

#### `POST /order`
Submit a signed order.
- **Auth:** L2 headers required
- **Body:** Signed order object with `OrderType` (`GTC`, `FOK`, `GTD`, `FAK`), optional `postOnly`
- **Response:** `OrderResponse`
```json
{
  "success": true,
  "orderID": "0xabc...",
  "transactionsHashes": ["0x..."]
}
```

#### `POST /orders`
Batch post orders (up to 15).
- **Auth:** L2 headers required
- **Body:** Array of `PostOrdersArgs`
- **Response:** `OrderResponse[]`

#### `GET /data/order/{order_hash}`
Get a specific order.
- **Auth:** L2 headers required
- **Path Params:** `order_hash` (string)
- **Response:** `OpenOrder`
```json
{
  "id": "0xb816...",
  "status": "LIVE",
  "market": "0x1b6f...",
  "original_size": "100",
  "outcome": "Yes",
  "maker_address": "0x...",
  "owner": "api-key-id",
  "price": "0.65",
  "side": "BUY",
  "size_matched": "25",
  "asset_id": "token-id",
  "expiration": "0",
  "type": "GTC",
  "created_at": "1697875200",
  "associate_trades": ["trade-id-1"]
}
```

**OpenOrder Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Order ID |
| `status` | string | Current status |
| `market` | string | Condition ID |
| `original_size` | string | Size at placement |
| `outcome` | string | Human-readable outcome |
| `maker_address` | string | Funder address |
| `owner` | string | API key |
| `price` | string | Order price |
| `side` | string | BUY or SELL |
| `size_matched` | string | Filled amount |
| `asset_id` | string | Token ID |
| `expiration` | string | UNIX expiry (0 = no expiry) |
| `type` | string | GTC, FOK, GTD |
| `created_at` | string | UNIX creation timestamp |
| `associate_trades` | string[] | Related trade IDs |

#### `GET /data/orders`
List open orders.
- **Auth:** L2 headers required
- **Query Params:** `id`, `market`, `asset_id` (all optional)
- **Response:** `OpenOrder[]`

#### `DELETE /order`
Cancel a single order.
- **Auth:** L2 headers required
- **Body:** `{ "orderID": "0x..." }`
- **Response:** `CancelOrdersResponse`
```json
{
  "canceled": ["0x..."],
  "not_canceled": []
}
```

#### `DELETE /orders`
Cancel multiple orders.
- **Auth:** L2 headers required
- **Body:** `{ "orderIDs": ["0x...", "0x..."] }`
- **Response:** `CancelOrdersResponse`

#### `DELETE /cancel-all`
Cancel all open orders.
- **Auth:** L2 headers required
- **Response:** `CancelOrdersResponse`

#### `DELETE /cancel-market-orders`
Cancel orders for specific market/asset.
- **Auth:** L2 headers required
- **Body:** `{ "market": "...", "asset_id": "..." }` (both optional)
- **Response:** `CancelOrdersResponse`

### User Data (L2 -- Authenticated)

#### `GET /trades`
User's trade history.
- **Auth:** L2 headers required
- **Query Params:** Optional `TradeParams` for filtering
- **Response:** `Trade[]`

#### `GET /trades-paginated`
Paginated trade history.
- **Auth:** L2 headers required
- **Query Params:** `TradeParams`
- **Response:** `TradesPaginatedResponse` with `trades[]`, `limit`, `count`

#### `GET /balance-allowance`
Balance and token allowances.
- **Auth:** L2 headers required
- **Query Params:** `asset_type`, optional `token_id`
- **Response:** `{ "balance": "1000.00", "allowance": "5000.00" }`

#### `GET /api-keys`
List API keys.
- **Auth:** L2 headers required
- **Response:** `ApiKeysResponse` with array of `ApiKeyCreds`

#### `DELETE /api-key`
Revoke current API key.
- **Auth:** L2 headers required

#### `GET /notifications`
User notifications.
- **Auth:** L2 headers required
- **Response:** `Notification[]`

**Notification Types:**
- `1` -- Order Cancellation
- `2` -- Order Fill
- `4` -- Market Resolved

#### `DELETE /notifications`
Dismiss notifications.
- **Auth:** L2 headers required
- **Body:** `{ "ids": [1, 2, 3] }`

### Calculate Market Price (L0 -- No Auth)

#### `GET /calculate-market-price`
Estimate execution price for a hypothetical order.
- **Query Params:**
  - `token_id` (string, required)
  - `side` (string, required: `BUY` or `SELL`)
  - `amount` (number, required)
  - `order_type` (string, optional: `GTC`, `FOK`, `GTD`, `FAK`)
- **Response:** `number` (estimated price)

---

## Gamma API Endpoints

Base URL: `https://gamma-api.polymarket.com`

The Gamma API is a **read-only** hosted indexing service for market metadata, categorization, and volume metrics. It also exposes a **GraphQL** endpoint at `/query` (with a GraphiQL IDE).

### Data Hierarchy

```
Event (contains 1+ markets)
  |-- Single Market Event (SMP): 1 market
  |-- Group Market Event (GMP): 2+ markets
      |
      Market (fundamental trading unit)
        |-- condition_id
        |-- token_ids (pair)
        |-- market_address
        |-- question_id
```

### `GET /events`
List events with nested markets.

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `limit` | integer (>=0) | Max results |
| `offset` | integer (>=0) | Pagination offset |
| `order` | string | Comma-separated sort fields |
| `ascending` | boolean | Sort direction |
| `id` | integer[] | Filter by event IDs |
| `slug` | string[] | Filter by event slugs |
| `tag_id` | integer | Filter by tag |
| `exclude_tag_id` | integer[] | Exclude tags |
| `related_tags` | boolean | Include related tags |
| `featured` | boolean | Featured only |
| `cyom` | boolean | Create Your Own Market filter |
| `include_chat` | boolean | Include chat data |
| `include_template` | boolean | Include template data |
| `recurrence` | string | Recurrence pattern |
| `closed` | boolean | Include closed events |
| `start_date_min` | string (ISO 8601) | Min start date |
| `start_date_max` | string (ISO 8601) | Max start date |
| `end_date_min` | string (ISO 8601) | Min end date |
| `end_date_max` | string (ISO 8601) | Max end date |

**Response:** Array of Event objects.

**Event Object Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Event ID |
| `ticker` | string | Ticker symbol |
| `slug` | string | URL slug |
| `title` | string | Event title |
| `subtitle` | string | Subtitle |
| `description` | string | Full description |
| `resolutionSource` | string | Resolution source |
| `startDate` | string | Start date |
| `endDate` | string | End date |
| `creationDate` | string | Creation date |
| `image` | string | Image URL |
| `icon` | string | Icon URL |
| `active` | boolean | Is active |
| `closed` | boolean | Is closed |
| `archived` | boolean | Is archived |
| `new` | boolean | Is new |
| `featured` | boolean | Is featured |
| `liquidity` | string | Total liquidity |
| `volume` | string | Total volume |
| `openInterest` | string | Open interest |
| `category` | string | Category |
| `subcategory` | string | Subcategory |
| `enableOrderBook` | boolean | CLOB enabled |
| `liquidityAmm` | string | AMM liquidity |
| `liquidityClob` | string | CLOB liquidity |
| `negRisk` | boolean | Negative risk |
| `negRiskMarketID` | string | Neg risk market ID |
| `negRiskFeeBips` | number | Neg risk fee bps |
| `markets` | Market[] | Nested markets |
| `series` | object | Series info |
| `categories` | Category[] | Categories |
| `collections` | object[] | Collections |
| `tags` | Tag[] | Tags |

### `GET /events/{id}`
Single event by ID.
- **Path Params:** `id` (integer, required)
- **Response:** Event object.

### `GET /markets`
List markets.

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `limit` | integer (>=0) | Max results |
| `offset` | integer (>=0) | Pagination offset |
| `order` | string | Comma-separated sort fields |
| `ascending` | boolean | Sort direction |
| `id` | integer[] | Filter by market IDs |
| `slug` | string[] | Filter by market slugs |
| `clob_token_ids` | string[] | CLOB token IDs |
| `condition_ids` | string[] | Condition IDs |
| `market_maker_address` | string[] | Market maker addresses |
| `liquidity_num_min` | number | Min liquidity |
| `liquidity_num_max` | number | Max liquidity |
| `volume_num_min` | number | Min volume |
| `volume_num_max` | number | Max volume |
| `start_date_min` | string (ISO 8601) | Min start date |
| `start_date_max` | string (ISO 8601) | Max start date |
| `end_date_min` | string (ISO 8601) | Min end date |
| `end_date_max` | string (ISO 8601) | Max end date |
| `tag_id` | integer | Tag filter |
| `related_tags` | boolean | Include related tags |
| `cyom` | boolean | CYOM filter |
| `uma_resolution_status` | string | UMA resolution status |
| `game_id` | string | Sports game ID |
| `sports_market_types` | string[] | Sport market types |
| `rewards_min_size` | number | Min reward size |
| `question_ids` | string[] | Question IDs |
| `include_tag` | boolean | Include tag data |
| `closed` | boolean | Closed filter |

**Response:** Array of Market objects.

**Gamma Market Object Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Market ID |
| `conditionId` | string | Condition ID |
| `slug` | string | URL slug |
| `question` | string | Market question |
| `creator` | string | Creator address |
| `outcomes` | string | JSON-encoded outcomes `'["Yes", "No"]'` |
| `outcomePrices` | string | JSON-encoded prices `'[0.65, 0.35]'` |
| `denominationToken` | string | Collateral token |
| `fee` | string | Fee rate |
| `marketType` | string | Market type |
| `formatType` | string | Format type |
| `enableOrderBook` | boolean | CLOB enabled |
| `orderPriceMinTickSize` | string | Min tick size |
| `orderMinSize` | string | Min order size |
| `liquidity` | string | Total liquidity |
| `liquidityNum` | number | Liquidity (numeric) |
| `volume` | string | Total volume |
| `volumeNum` | number | Volume (numeric) |
| `volume24hr` | number | 24h volume |
| `volume1wk` | number | 1 week volume |
| `volume1mo` | number | 1 month volume |
| `volume1yr` | number | 1 year volume |
| `volumeAmm` | string | AMM volume |
| `volumeClob` | string | CLOB volume |
| `liquidityAmm` | string | AMM liquidity |
| `liquidityClob` | string | CLOB liquidity |
| `bestBid` | number | Best bid price |
| `bestAsk` | number | Best ask price |
| `lastTradePrice` | number | Last trade price |
| `startDate` | string | Start date |
| `endDate` | string | End date |
| `createdAt` | string | Creation timestamp |
| `updatedAt` | string | Update timestamp |
| `closedTime` | string | Close timestamp |
| `active` | boolean | Is active |
| `closed` | boolean | Is closed |
| `archived` | boolean | Is archived |
| `restricted` | boolean | Is restricted |
| `ready` | boolean | Is ready |
| `funded` | boolean | Is funded |
| `image` | string | Image URL |
| `icon` | string | Icon URL |
| `twitterCardImage` | string | Twitter card image |
| `description` | string | Description |
| `umaEndDate` | string | UMA end date |
| `umaResolutionStatus` | string | UMA resolution status |
| `umaBond` | string | UMA bond |
| `umaReward` | string | UMA reward |
| `events` | Event[] | Parent events |
| `categories` | Category[] | Categories |
| `tags` | Tag[] | Tags |

### `GET /markets/{id}`
Single market by ID.

### `GET /search`
Search markets, events, and profiles.
- **Query Params:** Query string (implementation varies)

---

## Data API Endpoints

Base URL: `https://data-api.polymarket.com`

All Data API endpoints are **public** (no auth required) but return per-user data based on wallet address.

### `GET /positions`
User's current positions.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `user` | string | Yes | Wallet address |
| `market` | string | No | Condition ID(s), comma-separated |
| `sizeThreshold` | number | No | Min position size (default: 1.0) |
| `redeemable` | boolean | No | Redeemable filter |
| `mergeable` | boolean | No | Mergeable filter |
| `title` | string | No | Market title filter |
| `limit` | integer | No | Max results (default: 100, max: 500) |
| `offset` | integer | No | Pagination offset |
| `sortBy` | string | No | TOKENS, CURRENT, INITIAL, CASHPNL, PERCENTPNL, TITLE, RESOLVING, PRICE |
| `sortDirection` | string | No | ASC or DESC (default: DESC) |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `proxyWallet` | string | Proxy wallet address |
| `asset` | string | Asset ID |
| `conditionId` | string | Condition ID |
| `size` | string | Position size |
| `avgPrice` | string | Average entry price |
| `initialValue` | string | Initial value |
| `currentValue` | string | Current value |
| `cashPnl` | string | Cash P&L |
| `percentPnl` | string | Percent P&L |
| `curPrice` | string | Current price |
| `redeemable` | boolean | Can redeem |
| `title` | string | Market title |
| `outcome` | string | Outcome name |
| `endDate` | string | End date |
| `negativeRisk` | boolean | Negative risk |

### `GET /trades`
User's trade history.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `user` | string | No | Wallet address |
| `limit` | integer | No | Max results (default: 100, max: 500) |
| `offset` | integer | No | Pagination offset |
| `takerOnly` | boolean | No | Taker trades only (default: true) |
| `filterType` | string | No | CASH or TOKENS |
| `filterAmount` | number | No | Amount threshold |
| `market` | string | No | Condition ID(s) |
| `side` | string | No | BUY or SELL |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `proxyWallet` | string | Proxy wallet address |
| `side` | string | BUY or SELL |
| `asset` | string | Asset ID |
| `conditionId` | string | Condition ID |
| `size` | string | Trade size |
| `price` | string | Trade price |
| `timestamp` | string | Trade timestamp |
| `title` | string | Market title |
| `outcome` | string | Outcome name |
| `transactionHash` | string | On-chain tx hash |

### `GET /activity`
On-chain activity (trades, splits, merges, redeems).

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `user` | string | Yes | Wallet address |
| `market` | string | No | Condition ID(s), comma-separated |
| `type` | string | No | TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION |
| `limit` | integer | No | Max results (default: 100, max: 500) |
| `offset` | integer | No | Pagination offset |
| `start` | number | No | Start timestamp (seconds) |
| `end` | number | No | End timestamp (seconds) |
| `side` | string | No | BUY or SELL |
| `sortBy` | string | No | TIMESTAMP, TOKENS, CASH (default: TIMESTAMP) |
| `sortDirection` | string | No | ASC or DESC (default: DESC) |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `proxyWallet` | string | Proxy wallet address |
| `timestamp` | string | Activity timestamp |
| `conditionId` | string | Condition ID |
| `type` | string | Activity type |
| `size` | string | Size |
| `usdcSize` | string | USDC value |
| `transactionHash` | string | On-chain tx hash |
| `price` | string | Price |
| `asset` | string | Asset ID |
| `side` | string | Side |
| `outcome` | string | Outcome name |

### `GET /holders`
Top position holders for a market.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `market` | string | Yes | Condition ID |
| `limit` | integer | No | Max holders (default: 100) |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `token` | string | Token ID |
| `holders` | array | Array of holder objects |
| `holders[].proxyWallet` | string | Wallet address |
| `holders[].amount` | string | Position size |
| `holders[].pseudonym` | string | Display name |
| `holders[].outcomeIndex` | number | Outcome index |
| `holders[].name` | string | Profile name |
| `holders[].profileImage` | string | Profile image URL |

### `GET /value`
Total USD value of a user's positions.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `user` | string | Yes | Wallet address |
| `market` | string | No | Condition ID(s), optional |

**Response:**
```json
{"user": "0x...", "value": 12345.67}
```

---

## WebSocket Endpoints

### CLOB WebSocket

**URL:** `wss://ws-subscriptions-clob.polymarket.com/ws/`

#### Channels

| Channel | Auth Required | Purpose |
|---------|--------------|---------|
| `market` | No | Order book updates, price changes, trade events |
| `user` | Yes (L2) | Order status updates, fills, cancellations |

#### Subscription Message Format

```json
{
  "auth": { ... },
  "type": "MARKET",
  "assets_ids": ["<token_id_1>", "<token_id_2>"]
}
```

For user channel:
```json
{
  "auth": {
    "apiKey": "...",
    "secret": "...",
    "passphrase": "..."
  },
  "type": "USER",
  "markets": ["<condition_id_1>", "<condition_id_2>"]
}
```

#### Dynamic Subscription (post-connection)

```json
{
  "type": "subscribe",
  "assets_ids": ["<new_token_id>"]
}
```

```json
{
  "type": "unsubscribe",
  "assets_ids": ["<token_id_to_remove>"]
}
```

**Limit:** Max 500 instruments per WebSocket connection.

#### Market Channel Message Types

##### 1. `book`
Full order book snapshot. Emitted on subscription and when trades affect the book.

```json
{
  "event_type": "book",
  "asset_id": "string (token ID)",
  "market": "string (condition ID)",
  "timestamp": "string (UNIX ms)",
  "hash": "string (orderbook hash)",
  "buys": [{"price": "string", "size": "string"}],
  "sells": [{"price": "string", "size": "string"}]
}
```

##### 2. `price_change`
Emitted when orders are placed or cancelled.

```json
{
  "event_type": "price_change",
  "market": "string (condition ID)",
  "timestamp": "string",
  "price_changes": [
    {
      "asset_id": "string",
      "price": "string",
      "size": "string",
      "side": "BUY | SELL",
      "hash": "string",
      "best_bid": "string",
      "best_ask": "string"
    }
  ]
}
```

##### 3. `tick_size_change`
Emitted when minimum tick size changes (price >0.96 or <0.04).

```json
{
  "event_type": "tick_size_change",
  "asset_id": "string",
  "market": "string",
  "old_tick_size": "string",
  "new_tick_size": "string",
  "side": "string",
  "timestamp": "string"
}
```

##### 4. `last_trade_price`
Emitted when maker and taker orders match.

```json
{
  "event_type": "last_trade_price",
  "asset_id": "string",
  "market": "string",
  "price": "string",
  "side": "string",
  "size": "string",
  "fee_rate_bps": "string",
  "timestamp": "string"
}
```

##### 5. `best_bid_ask` (feature-flagged)
Emitted when best bid/ask prices change.

```json
{
  "event_type": "best_bid_ask",
  "market": "string",
  "asset_id": "string",
  "best_bid": "string",
  "best_ask": "string",
  "spread": "string",
  "timestamp": "string"
}
```

##### 6. `new_market` (feature-flagged)
Emitted when a new market is created.

```json
{
  "event_type": "new_market",
  "id": "string",
  "question": "string",
  "market": "string",
  "slug": "string",
  "description": "string",
  "assets_ids": ["string"],
  "outcomes": ["string"],
  "timestamp": "string"
}
```

##### 7. `market_resolved` (feature-flagged)
Emitted when a market resolves. Extends `new_market` with:

```json
{
  "event_type": "market_resolved",
  "winning_asset_id": "string",
  "winning_outcome": "string",
  ...
}
```

### Real-Time Data Service (RTDS)

**URL:** `wss://ws-live-data.polymarket.com`

Provides cryptocurrency price information and comment feeds. Separate from CLOB WebSocket.

---

## Pagination Patterns

### CLOB API -- Cursor-Based Pagination

The CLOB API uses **cursor-based pagination** for market listings:

```python
# Python example
client = ClobClient("https://clob.polymarket.com")
all_markets = []
next_cursor = None

while True:
    if next_cursor is None:
        response = client.get_markets()
    else:
        response = client.get_markets(next_cursor=next_cursor)

    all_markets.extend(response["data"])
    next_cursor = response.get("next_cursor")

    if not next_cursor:
        break
```

**Response format:**
```json
{
  "data": [...],
  "next_cursor": "string or null",
  "count": 100
}
```

### Gamma API -- Offset-Based Pagination

The Gamma API uses traditional `limit`/`offset` pagination:

```
GET /events?limit=100&offset=0
GET /events?limit=100&offset=100
GET /markets?limit=50&offset=200
```

### Data API -- Offset-Based Pagination

```
GET /positions?user=0x...&limit=100&offset=0
GET /trades?user=0x...&limit=500&offset=0
```

Max limit: 500 for all Data API endpoints.

---

## Client Libraries

### Official Libraries

| Language | Package | Install |
|----------|---------|---------|
| **Python** | `py-clob-client` | `pip install py-clob-client` |
| **TypeScript** | `@polymarket/clob-client` | `npm install @polymarket/clob-client ethers` |
| **Rust** | `polymarket-client-sdk` | `cargo add polymarket-client-sdk` |

### Python: `py-clob-client`

**PyPI:** https://pypi.org/project/py-clob-client/
**GitHub:** https://github.com/Polymarket/py-clob-client
**Version:** v0.34.5+ (latest as of Jan 2026)
**Requires:** Python 3.9+

#### Initialization by Auth Level

**L0 (Read-only, no auth):**
```python
from py_clob_client.client import ClobClient

client = ClobClient("https://clob.polymarket.com")
```

**L1 (Key derivation):**
```python
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key="0x_YOUR_PRIVATE_KEY"
)
creds = client.create_or_derive_api_creds()
```

**L2 (Trading):**
```python
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key="0x_YOUR_PRIVATE_KEY",
    signature_type=2,  # GNOSIS_SAFE
    funder="0x_YOUR_PROXY_WALLET"
)
client.set_api_creds(client.create_or_derive_api_creds())
```

#### Key Methods

**Public (L0):**
- `get_ok()` -- Health check
- `get_server_time()` -- Server timestamp
- `get_markets(next_cursor=None)` -- Paginated markets
- `get_simplified_markets(next_cursor=None)` -- Lightweight markets
- `get_sampling_markets(next_cursor=None)` -- Sampled markets
- `get_market(condition_id)` -- Single market
- `get_order_book(token_id)` -- Order book
- `get_order_books([BookParams])` -- Multiple order books
- `get_price(token_id, side)` -- Best price
- `get_midpoint(token_id)` -- Midpoint
- `get_spread(token_id)` -- Spread
- `get_last_trade_price(token_id)` -- Last trade
- `get_prices_history(PriceHistoryFilterParams)` -- Historical prices
- `calculate_market_price(token_id, side, amount)` -- Price estimate

**Authenticated (L2):**
- `create_order(OrderArgs)` -- Create limit order
- `create_market_order(MarketOrderArgs)` -- Create market order
- `post_order(signed_order, OrderType)` -- Submit signed order
- `create_and_post_order(UserOrder, options, order_type)` -- Create + submit
- `get_order(order_id)` -- Get order details
- `get_orders(OpenOrderParams)` -- List open orders
- `cancel(order_id)` -- Cancel single order
- `cancel_orders(order_ids)` -- Cancel multiple
- `cancel_all()` -- Cancel all
- `cancel_market_orders(params)` -- Cancel by market
- `get_trades()` -- Trade history
- `get_last_trade_price(token_id)` -- Last trade
- `get_balance_allowance(params)` -- Balance info
- `get_api_keys()` -- List API keys
- `delete_api_key()` -- Revoke key
- `get_notifications()` -- Get notifications
- `drop_notifications(ids)` -- Dismiss notifications

### TypeScript: `@polymarket/clob-client`

**npm:** `@polymarket/clob-client`
**GitHub:** https://github.com/Polymarket/clob-client

```typescript
import { ClobClient } from "@polymarket/clob-client";
import { ethers } from "ethers";

// L0
const client = new ClobClient("https://clob.polymarket.com", 137);

// L2
const signer = new ethers.Wallet(PRIVATE_KEY);
const tradingClient = new ClobClient(
  "https://clob.polymarket.com",
  137,
  signer,
  apiCreds,
  SignatureType.POLY_GNOSIS_SAFE,
  funderAddress
);
```

### Third-Party Libraries

| Library | Language | Package |
|---------|----------|---------|
| `polymarket-apis` | Python | `pip install polymarket-apis` |
| `@dicedhq/polymarket` | TypeScript (JSR) | JSR registry |
| NautilusTrader | Python | Full algo trading framework with Polymarket adapter |

---

## Data Model & Key Concepts

### Key Identifiers

| Identifier | Format | Description | Where Used |
|------------|--------|-------------|------------|
| `condition_id` | Hex string | Unique market identifier | CLOB + Gamma |
| `token_id` | Long numeric string | Unique outcome token ID | CLOB (order book, prices) |
| `question_id` | String | Links to question metadata | CLOB + Gamma |
| `asset_id` | String | Alias for token_id in some contexts | WebSocket, Data API |

### Market Structure

Each market has exactly **2 outcome tokens** (binary: Yes/No):
```
Market (condition_id)
  |-- Token A (token_id_yes) -- "Yes" outcome
  |-- Token B (token_id_no)  -- "No" outcome
```

Prices are complementary: `price_yes + price_no ~= 1.00`

### Negative Risk Markets

Some multi-outcome events use `neg_risk = true`, where individual markets within the event share a common neg_risk_market_id. This affects settlement mechanics.

### Order Types

| Type | Description |
|------|-------------|
| `GTC` | Good Till Cancelled -- stays on book until filled or cancelled |
| `GTD` | Good Till Date -- expires at specified timestamp |
| `FOK` | Fill Or Kill -- must fill entirely immediately or cancel |
| `FAK` | Fill And Kill -- fills what it can immediately, cancels remainder |

### Tick Sizes

Tick size varies by price proximity to extremes:
- Standard: `0.01`
- Near extremes (price >0.96 or <0.04): `0.001` or `0.0001`

---

## Quick Reference: Common Data Pipeline Patterns

### Fetch All Active Markets

```python
from py_clob_client.client import ClobClient

client = ClobClient("https://clob.polymarket.com")
all_markets = []
cursor = None

while True:
    resp = client.get_markets(next_cursor=cursor) if cursor else client.get_markets()
    all_markets.extend(resp["data"])
    cursor = resp.get("next_cursor")
    if not cursor:
        break

active = [m for m in all_markets if m["active"] and not m["closed"]]
```

### Fetch Market + Order Book + Price History

```python
import requests

# Get event with nested markets from Gamma
event = requests.get(
    "https://gamma-api.polymarket.com/events",
    params={"slug": "will-trump-win-2024", "limit": 1}
).json()[0]

# Get order book from CLOB
token_id = event["markets"][0]["clobTokenIds"][0]
book = client.get_order_book(token_id)

# Get price history from CLOB
history = client.get_prices_history({
    "market": token_id,
    "interval": "max",
    "fidelity": 60
})
```

### Stream Real-Time Prices via WebSocket

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data.get("event_type") == "last_trade_price":
        print(f"Trade: {data['asset_id']} @ {data['price']} ({data['side']})")

ws = websocket.WebSocketApp(
    "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    on_message=on_message
)
# Subscribe after connect
ws.send(json.dumps({
    "type": "MARKET",
    "assets_ids": ["<token_id>"]
}))
ws.run_forever()
```

---

## Sources

- Official Docs: https://docs.polymarket.com/
- CLOB Introduction: https://docs.polymarket.com/developers/CLOB/introduction
- Endpoints Reference: https://docs.polymarket.com/quickstart/reference/endpoints
- Rate Limits: https://docs.polymarket.com/quickstart/introduction/rate-limits
- Authentication: https://docs.polymarket.com/developers/CLOB/authentication
- Public Methods: https://docs.polymarket.com/developers/CLOB/clients/methods-public
- L2 Methods: https://docs.polymarket.com/developers/CLOB/clients/methods-l2
- WebSocket Market Channel: https://docs.polymarket.com/developers/CLOB/websocket/market-channel
- WSS Overview: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
- Timeseries: https://docs.polymarket.com/developers/CLOB/timeseries
- Gamma Overview: https://docs.polymarket.com/developers/gamma-markets-api/overview
- Gamma Events: https://docs.polymarket.com/developers/gamma-markets-api/get-events
- Gamma Markets: https://docs.polymarket.com/developers/gamma-markets-api/get-markets
- Gamma Structure: https://docs.polymarket.com/developers/gamma-markets-api/gamma-structure
- py-clob-client GitHub: https://github.com/Polymarket/py-clob-client
- py-clob-client PyPI: https://pypi.org/project/py-clob-client/
- Data API Gist: https://gist.github.com/shaunlebron/0dd3338f7dea06b8e9f8724981bb13bf
