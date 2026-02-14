# Research: Polymarket Public APIs for User/Wallet Data

## Summary

Polymarket exposes comprehensive user-level data through **three public APIs** (no authentication required for read-only access). All user endpoints are keyed by **proxy wallet address** (0x-prefixed, 40 hex chars). The Data API is the primary source for user trading data; the Gamma API handles profiles and search.

---

## API Base URLs

| API | Base URL | Purpose |
|-----|----------|---------|
| **Gamma API** | `https://gamma-api.polymarket.com` | Market metadata, profiles, search |
| **Data API** | `https://data-api.polymarket.com` | User positions, trades, activity, leaderboard |
| **CLOB API** | `https://clob.polymarket.com` | Prices, orderbook (user orders require L2 auth) |

---

## Public User Endpoints (No Auth Required)

### 1. GET /public-profile (Gamma API)

**URL:** `https://gamma-api.polymarket.com/public-profile?address={wallet}`

**Purpose:** Look up a user's public profile by wallet address.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `address` | string | Yes | Wallet address (`0x` + 40 hex chars) |

**Response fields:**
- `createdAt` (datetime) -- profile creation timestamp
- `proxyWallet` (string) -- proxy wallet address
- `profileImage` (URI) -- profile picture URL
- `displayUsernamePublic` (boolean) -- whether username is public
- `bio` (string) -- user bio
- `pseudonym` (string) -- display name / pseudonym
- `name` (string) -- real name (if set)
- `users` (array) -- associated user objects with `id`, `creator`, `mod` properties
- `xUsername` (string) -- X/Twitter handle
- `verifiedBadge` (boolean) -- verification status

**Error responses:** 400 (invalid address format), 404 (profile not found)

---

### 2. GET /public-search (Gamma API)

**URL:** `https://gamma-api.polymarket.com/public-search?q={query}&search_profiles=true`

**Purpose:** Unified search across markets, events, and user profiles.

**Key Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `q` | string | Yes | Search query |
| `search_profiles` | boolean | No | Include profiles in results |
| `limit_per_type` | integer | No | Results limit per entity type |
| `page` | integer | No | Pagination |
| `cache` | boolean | No | Enable/disable caching |
| `events_status` | string | No | Filter by event status |
| `events_tag` | array | No | Filter by event tags |
| `sort` | string | No | Sort field |
| `ascending` | boolean | No | Sort direction |

**Response structure:**
```json
{
  "events": [...],
  "tags": [...],
  "profiles": [
    {
      "id": "...",
      "name": "...",
      "pseudonym": "...",
      "bio": "...",
      "proxyWallet": "0x...",
      "profileImage": "...",
      "walletActivated": true
    }
  ],
  "pagination": { "hasMore": true, "totalResults": 100 }
}
```

---

### 3. GET /v1/leaderboard (Data API)

**URL:** `https://data-api.polymarket.com/v1/leaderboard`

**Purpose:** Ranked trader leaderboard with PnL and volume data.

**Parameters:**
| Param | Type | Default | Values |
|-------|------|---------|--------|
| `category` | string | OVERALL | OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, MENTIONS, WEATHER, ECONOMICS, TECH, FINANCE |
| `timePeriod` | string | DAY | DAY, WEEK, MONTH, ALL |
| `orderBy` | string | PNL | PNL, VOL |
| `limit` | integer | 25 | 1-50 |
| `offset` | integer | 0 | 0-1000 |
| `user` | string | -- | Specific wallet address to look up |
| `userName` | string | -- | Search by username |

**Response (array of objects):**
```json
[
  {
    "rank": "1",
    "proxyWallet": "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
    "userName": "trader123",
    "vol": 1500000.0,
    "pnl": 250000.0,
    "profileImage": "https://...",
    "xUsername": "trader_x",
    "verifiedBadge": true
  }
]
```

**Notes:**
- Max 50 results per page, offset up to 1000 (so max ~1050 traders accessible)
- Can filter to a specific user's rank with `user` param
- Categories match Gamma API tag labels
- Supports both PnL and volume ranking

---

### 4. GET /positions (Data API)

**URL:** `https://data-api.polymarket.com/positions?user={wallet}`

**Purpose:** Current open positions for a user with full PnL breakdown.

**Parameters:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user` | string | Yes | -- | Wallet address |
| `market` | array | No | -- | Comma-separated condition IDs |
| `eventId` | array | No | -- | Comma-separated event IDs (mutually exclusive with `market`) |
| `sizeThreshold` | number | No | 1 | Min position size |
| `redeemable` | boolean | No | false | Filter redeemable positions |
| `mergeable` | boolean | No | false | Filter mergeable positions |
| `title` | string | No | -- | Search market title (max 100 chars) |
| `limit` | integer | No | 100 | Max 500 |
| `offset` | integer | No | 0 | Max 10,000 |
| `sortBy` | string | No | TOKENS | CURRENT, INITIAL, TOKENS, CASHPNL, PERCENTPNL, TITLE, RESOLVING, PRICE, AVGPRICE |
| `sortDirection` | string | No | DESC | ASC, DESC |

**Response fields per position:**
- `proxyWallet`, `asset`, `conditionId` -- identifiers
- `size` -- number of tokens held
- `avgPrice` -- average entry price
- `initialValue` -- total cost basis
- `currentValue`, `curPrice` -- current mark-to-market
- `cashPnl`, `percentPnl` -- unrealized PnL
- `realizedPnl`, `percentRealizedPnl` -- realized PnL
- `totalBought` -- total tokens purchased
- `title`, `slug`, `outcome`, `outcomeIndex` -- market metadata
- `endDate` -- market end date
- `redeemable`, `mergeable`, `negativeRisk` -- status flags

---

### 5. GET /trades (Data API)

**URL:** `https://data-api.polymarket.com/trades?user={wallet}`

**Purpose:** Trade history for a user or specific markets.

**Parameters:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user` | string | No* | -- | Wallet address |
| `market` | array | No* | -- | Condition IDs (comma-separated) |
| `eventId` | array | No | -- | Event IDs (mutually exclusive with `market`) |
| `limit` | integer | No | 100 | Max 10,000 |
| `offset` | integer | No | 0 | Max 10,000 |
| `takerOnly` | boolean | No | true | Filter to taker trades |
| `filterType` | string | No | -- | CASH or TOKENS (requires filterAmount) |
| `filterAmount` | number | No | -- | Min trade size (requires filterType) |
| `side` | string | No | -- | BUY or SELL |

*At least `user` or `market` should be provided.

**Response fields per trade:**
- `proxyWallet` -- trader wallet
- `side` -- BUY or SELL
- `asset`, `conditionId` -- market identifiers
- `size` -- trade size (tokens)
- `price` -- execution price
- `timestamp` -- trade timestamp
- `transactionHash` -- on-chain tx hash
- `title`, `slug`, `icon`, `eventSlug` -- market metadata
- `outcome`, `outcomeIndex` -- which outcome was traded
- `name`, `pseudonym`, `bio`, `profileImage`, `profileImageOptimized` -- trader profile

---

### 6. GET /activity (Data API)

**URL:** `https://data-api.polymarket.com/activity?user={wallet}`

**Purpose:** Full on-chain activity history (trades, splits, merges, redeems, rewards, conversions, maker rebates).

**Parameters:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user` | string | Yes | -- | Wallet address |
| `market` | array | No | -- | Condition IDs |
| `eventId` | array | No | -- | Event IDs (mutually exclusive with market) |
| `type` | array | No | -- | TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE |
| `start` | integer | No | -- | Unix timestamp filter |
| `end` | integer | No | -- | Unix timestamp filter |
| `limit` | integer | No | 100 | Max 500 |
| `offset` | integer | No | 0 | Max 10,000 |
| `sortBy` | string | No | TIMESTAMP | TIMESTAMP, TOKENS, CASH |
| `sortDirection` | string | No | DESC | ASC, DESC |
| `side` | string | No | -- | BUY or SELL |

**Response fields per activity:**
- `proxyWallet` -- wallet address
- `timestamp` -- activity timestamp
- `conditionId` -- market identifier
- `type` -- activity type (TRADE, SPLIT, MERGE, etc.)
- `size` -- token amount
- `usdcSize` -- USDC equivalent
- `transactionHash` -- on-chain tx
- `price` -- execution price
- `asset` -- token ID
- `side` -- BUY/SELL
- `outcomeIndex` -- outcome traded
- `title`, `slug`, `icon`, `eventSlug` -- market metadata
- `outcome` -- outcome name
- `name`, `pseudonym`, `bio`, `profileImage`, `profileImageOptimized` -- profile

**Key difference from /trades:** Activity includes non-trade events (SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE) and always requires a `user` parameter.

---

### 7. GET /holders (Data API)

**URL:** `https://data-api.polymarket.com/holders?market={conditionId}`

**Purpose:** Top holders of a specific market token.

**Parameters:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `market` | string | Yes | -- | Condition ID (0x + 64 hex chars) |
| `limit` | integer | No | 20 | Max 20 |
| `minBalance` | integer | No | 1 | Min token balance (0-999999) |

**Response structure:**
```json
[
  {
    "token": "token_id",
    "holders": [
      {
        "proxyWallet": "0x...",
        "amount": 1500.0,
        "pseudonym": "whale_trader",
        "name": "...",
        "bio": "...",
        "profileImage": "...",
        "profileImageOptimized": "...",
        "displayUsernamePublic": true,
        "outcomeIndex": 0,
        "asset": "..."
      }
    ]
  }
]
```

**Note:** Limited to top 20 holders per request.

---

### 8. GET /value (Data API)

**URL:** `https://data-api.polymarket.com/value?user={wallet}`

**Purpose:** Total USD value of a user's positions (portfolio value).

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `user` | string | Yes | Wallet address |
| `market` | array | No | Optional condition IDs to filter |

**Response:**
```json
[
  {
    "user": "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
    "value": 1500.50
  }
]
```

---

## Authenticated Endpoints (CLOB API - L2 Auth Required)

These require API key/secret/passphrase generated via L1 (wallet signature) authentication:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/order/{order_hash}` | GET | Get a specific order by hash |
| `/data/orders` | GET | Get active/open orders (filter by market, asset) |
| `/order` | POST | Place an order |
| `/order` | DELETE | Cancel an order |

**Not useful for public user tracking** -- these require the user's own credentials.

---

## CLOB Authentication Levels

| Level | Auth | Capabilities |
|-------|------|-------------|
| **L0** | None | Market data, prices, orderbooks, spreads, midpoints |
| **L1** | Wallet signature (EIP-712) | Create/derive API credentials, sign orders |
| **L2** | HMAC-SHA256 (apiKey, secret, passphrase) | Post orders, cancel orders, view own orders/trades |

---

## Proxy Wallet Architecture

- Every Polymarket user gets a **proxy wallet** (1-of-1 multisig) deployed on Polygon
- **MetaMask users:** Gnosis Safe contracts via factory `0xaacfeea03eb1561c4e67d661e40682bd20e3541b`
- **MagicLink users:** Custom proxy via factory `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052`
- Proxy addresses are **deterministically derived** from the user's EOA using CREATE2
- All user-facing API endpoints use the **proxy wallet** address (not EOA)
- The `/public-profile` endpoint can look up a profile by either proxy or base wallet address

---

## Rate Limits

- CLOB market data endpoints: **1500 requests / 10 seconds**
- Data API: No documented rate limits, but pagination limits apply (max offset 10,000)
- Gamma API: No documented rate limits

---

## Key Observations for User Tracking

1. **No bulk user enumeration:** There is no `/users` or `/profiles/all` endpoint. Users must be discovered via:
   - Leaderboard (top ~1050 by PnL/volume, across categories)
   - Market holders (top 20 per market)
   - Trade history (per market)
   - Search (by username/name)
   - On-chain event logs (Polygon)

2. **Rich PnL data available:** The `/positions` endpoint provides full portfolio breakdown with entry prices, current values, realized/unrealized PnL -- all pre-computed by Polymarket.

3. **Activity types beyond trades:** The `/activity` endpoint captures SPLIT, MERGE, REDEEM, REWARD, CONVERSION, and MAKER_REBATE in addition to trades.

4. **Profile data is opt-in:** Users may not have `pseudonym`, `name`, `bio`, or `xUsername` set. `displayUsernamePublic` controls visibility.

5. **Leaderboard categories match market tags:** POLITICS, SPORTS, CRYPTO, CULTURE, MENTIONS, WEATHER, ECONOMICS, TECH, FINANCE.

6. **Transaction hashes link to Polygon:** Every trade/activity has a `transactionHash` for on-chain verification via PolygonScan.

7. **Pagination ceiling:** Most endpoints cap at offset=10,000 which limits deep historical queries. For comprehensive history, on-chain indexing is needed.

---

## Recommended Ingestion Strategy

| Data Type | Endpoint | Frequency | Key |
|-----------|----------|-----------|-----|
| Top traders | `/v1/leaderboard` | Every 5-15 min | proxyWallet |
| Trader profiles | `/public-profile` | On discovery / daily refresh | address |
| Open positions | `/positions` | Every 5-15 min per tracked user | user + conditionId |
| Trade history | `/trades` | Every 1-5 min per tracked user | user + timestamp |
| Full activity | `/activity` | Every 5 min per tracked user | user + timestamp |
| Market whale holders | `/holders` | Every 15-30 min per market | market + proxyWallet |
| Portfolio value | `/value` | Every 5 min per tracked user | user |
| User search | `/public-search` | On demand | q |
