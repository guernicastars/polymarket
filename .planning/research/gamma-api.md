# Polymarket Gamma (Markets) API -- Research

## Overview

Gamma is Polymarket's hosted service that indexes market data and provides additional market metadata (categorization, indexed volume, tags, images, etc.) through a read-only REST API. It is the primary interface for discovering and browsing markets -- separate from the CLOB API which handles order placement and trading.

**Base URL:** `https://gamma-api.polymarket.com`

**Authentication:** Public endpoints (events, markets, tags, series, teams, sports) require no authentication. The `/search` endpoint requires authentication (returns 401 without valid token/cookies).

**Rate Limits:** Not documented; no rate-limit headers observed in responses.

---

## Data Model & Hierarchy

```
Series (optional grouping)
  └── Events (organizational container)
        └── Markets (fundamental tradeable unit)
              └── Outcomes (Yes/No or multi-outcome)
```

### Key Relationships

- **Market** is the fundamental element -- each market corresponds to a single question with binary (Yes/No) or multi-outcome resolution. Each market has a unique `conditionId`, `questionID`, and CLOB token IDs.
- **Event** groups one or more related markets. A single-market event produces a "Single Market Page" (SMP); a multi-market event produces a "Group Market Page" (GMP). Example: "Who will Trump nominate as Fed Chair?" is one event containing many markets (one per candidate).
- **Series** groups recurring events (e.g., "March Madness Games", "SG Earnings"). Has a `recurrence` field (daily, weekly, monthly).
- **Tags** provide categorization (e.g., "Finance", "AI", "Earnings"). Tags have related-tag relationships with ranking.
- **Categories** are separate from tags -- structured label/slug pairs with optional parent categories.

### NegRisk (Multi-Outcome) Markets

Multi-outcome events use a "negative risk" model where all outcomes share a single `negRiskMarketID`. Each individual market within the event represents one outcome, and all share the same `questionID` prefix. Fields:
- `negRisk: true` -- indicates this market is part of a neg-risk group
- `negRiskMarketID` -- shared identifier across all markets in the group
- `negRiskRequestID` -- per-market request ID
- `negRiskFeeBips` -- fee in basis points for neg-risk markets

---

## Endpoints

### Events

#### `GET /events` -- List Events
Returns an array of Event objects with nested markets.

**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `limit` | integer (min: 0) | Results per page |
| `offset` | integer (min: 0) | Pagination offset |
| `order` | string | Comma-separated fields to order by (e.g., `volume24hr`, `id`, `startDate`) |
| `ascending` | boolean | Sort direction (default varies) |
| `id` | array[integer] | Filter by event IDs |
| `slug` | array[string] | Filter by event slugs |
| `tag_id` | integer | Filter by tag ID |
| `exclude_tag_id` | array[integer] | Exclude specific tag IDs |
| `tag_slug` | string | Filter by tag slug |
| `related_tags` | boolean | Include related tags in filtering |
| `active` | boolean | Filter by active status |
| `archived` | boolean | Filter by archived status |
| `featured` | boolean | Filter featured events only |
| `closed` | boolean | Filter by closed status |
| `cyom` | boolean | Create-your-own-market filter |
| `recurrence` | string | Event recurrence type |
| `liquidity_min` | number | Minimum liquidity threshold |
| `liquidity_max` | number | Maximum liquidity threshold |
| `volume_min` | number | Minimum volume threshold |
| `volume_max` | number | Maximum volume threshold |
| `start_date_min` | datetime (ISO 8601) | Earliest start date |
| `start_date_max` | datetime (ISO 8601) | Latest start date |
| `end_date_min` | datetime (ISO 8601) | Earliest end date |
| `end_date_max` | datetime (ISO 8601) | Latest end date |
| `include_chat` | boolean | Include chat data |
| `include_template` | boolean | Include template data |

#### `GET /events/{id}` -- Get Event by ID
**Path Params:** `id` (integer, required)
**Query Params:** `include_chat`, `include_template` (boolean, optional)
**Returns:** Single Event object. 404 if not found.

#### `GET /events/slug/{slug}` -- Get Event by Slug
**Path Params:** `slug` (string, required)
**Query Params:** `include_chat`, `include_template` (boolean, optional)
**Returns:** Single Event object. 404 if not found.

#### `GET /events/{id}/tags` -- Get Event Tags
**Path Params:** `id` (integer, required)
**Returns:** Array of Tag objects. 404 if event not found.

### Markets

#### `GET /markets` -- List Markets
Returns an array of Market objects.

**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `limit` | integer (min: 0) | Results per page |
| `offset` | integer (min: 0) | Pagination offset |
| `order` | string | Comma-separated fields to order by |
| `ascending` | boolean | Sort direction |
| `id` | array[integer] | Filter by market IDs |
| `slug` | array[string] | Filter by market slugs |
| `clob_token_ids` | array[string] | Filter by CLOB token IDs |
| `condition_ids` | array[string] | Filter by condition IDs |
| `question_ids` | array[string] | Filter by question IDs |
| `market_maker_address` | array[string] | Filter by market maker address |
| `liquidity_num_min` | number | Minimum liquidity |
| `liquidity_num_max` | number | Maximum liquidity |
| `volume_num_min` | number | Minimum volume |
| `volume_num_max` | number | Maximum volume |
| `start_date_min` | datetime | Earliest start date |
| `start_date_max` | datetime | Latest start date |
| `end_date_min` | datetime | Earliest end date |
| `end_date_max` | datetime | Latest end date |
| `tag_id` | integer | Filter by tag ID |
| `related_tags` | boolean | Include related tags |
| `cyom` | boolean | Create-your-own-market filter |
| `uma_resolution_status` | string | UMA resolution status filter |
| `game_id` | string | Filter by sports game ID |
| `sports_market_types` | array[string] | Sports market type filter |
| `rewards_min_size` | number | Minimum rewards size |
| `include_tag` | boolean | Include tag data in response |
| `closed` | boolean | Filter by closed status |

#### `GET /markets/{id}` -- Get Market by ID
**Path Params:** `id` (integer, required)
**Query Params:** `include_tag` (boolean, optional)
**Returns:** Single Market object. 404 if not found.

#### `GET /markets/slug/{slug}` -- Get Market by Slug
**Path Params:** `slug` (string, required)
**Query Params:** `include_tag` (boolean, optional)
**Returns:** Single Market object. 404 if not found.

#### `GET /markets/{id}/tags` -- Get Market Tags
**Path Params:** `id` (integer, required)
**Returns:** Array of Tag objects. 404 if not found.

### Series

#### `GET /series` -- List Series
**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `limit` | integer | Results per page |
| `offset` | integer | Pagination offset |
| `order` | string | Comma-separated sort fields |
| `ascending` | boolean | Sort direction |
| `slug` | array[string] | Filter by slugs |
| `categories_ids` | array[integer] | Filter by category IDs |
| `categories_labels` | array[string] | Filter by category labels |
| `closed` | boolean | Filter by closed status |
| `include_chat` | boolean | Include chat data |
| `recurrence` | string | Filter by recurrence type |

#### `GET /series/{id}` -- Get Series by ID
**Path Params:** `id` (integer, required)
**Query Params:** `include_chat` (boolean, optional)
**Returns:** Single Series object.

### Tags

#### `GET /tags` -- List Tags
**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `limit` | integer | Results per page |
| `offset` | integer | Pagination offset |
| `order` | string | Sort fields |
| `ascending` | boolean | Sort direction |
| `include_template` | boolean | Include template data |
| `is_carousel` | boolean | Filter carousel tags only |

#### `GET /tags/{id}` -- Get Tag by ID
**Query Params:** `include_template` (boolean, optional)

#### `GET /tags/slug/{slug}` -- Get Tag by Slug
**Query Params:** `include_template` (boolean, optional)

#### `GET /tags/{id}/related-tags` -- Related Tags by ID
**Query Params:**
- `omit_empty` (boolean) -- Omit empty relationships
- `status` (string) -- `"active"`, `"closed"`, or `"all"`

#### `GET /tags/slug/{slug}/related-tags` -- Related Tags by Slug
Same params as above.

#### `GET /tags/{id}/related-tags/tags` -- Detailed Related Tags by ID
Returns full Tag objects for related tags.

#### `GET /tags/slug/{slug}/related-tags/tags` -- Detailed Related Tags by Slug
Returns full Tag objects for related tags.

### Sports & Teams

#### `GET /sports` -- Sports Metadata
Returns array of sport configurations with image, resolution source, ordering, tag IDs, and series references.

#### `GET /teams` -- List Teams
**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `limit` | integer | Results per page |
| `offset` | integer | Pagination offset |
| `order` | string | Sort fields |
| `ascending` | boolean | Sort direction |
| `league` | array[string] | Filter by league(s) |
| `name` | array[string] | Filter by name(s) |
| `abbreviation` | array[string] | Filter by abbreviation(s) |

### Search (Requires Authentication)

#### `GET /search` -- Search Markets, Events, and Profiles
Returns 401 without valid authentication.

**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `q` | string (required) | Search query |
| `cache` | boolean | Use cache |
| `events_status` | string | Filter event status |
| `limit_per_type` | integer | Results per entity type |
| `page` | integer | Page number |
| `events_tag` | array[string] | Tag filter for events |
| `keep_closed_markets` | integer (0/1) | Include closed markets |
| `sort` | string | Sort field |
| `ascending` | boolean | Sort direction |
| `search_tags` | boolean | Include tags in search |
| `search_profiles` | boolean | Include profiles in search |
| `recurrence` | string | Recurrence filter |
| `exclude_tag_id` | array[integer] | Exclude tags |
| `optimized` | boolean | Use optimized search |

### Health

#### `GET /status` -- Health Check
Returns plain text `"OK"` with 200 status.

---

## Response Schemas

### Event Object (Complete)

```json
{
  "id": "35908",
  "ticker": "who-will-trump-nominate-as-fed-chair",
  "slug": "who-will-trump-nominate-as-fed-chair",
  "title": "Who will Trump nominate as Fed Chair?",
  "description": "This market will resolve according to...",
  "resolutionSource": "",
  "startDate": "2025-08-05T17:33:33.863763Z",
  "creationDate": "2025-08-05T17:33:33.863732Z",
  "endDate": "2026-12-31T00:00:00Z",
  "image": "https://polymarket-upload.s3.us-east-2.amazonaws.com/...",
  "icon": "https://polymarket-upload.s3.us-east-2.amazonaws.com/...",
  "active": true,
  "closed": false,
  "archived": false,
  "new": false,
  "featured": false,
  "restricted": true,
  "liquidity": 70493960.6881,
  "volume": 476808899.117802,
  "openInterest": 0,
  "sortBy": "price",
  "createdAt": "2025-08-05T17:13:18.083346Z",
  "updatedAt": "2026-02-14T14:12:05.851211Z",
  "competitive": 0.8231589176201076,
  "volume24hr": 11018294.309006998,
  "volume1wk": 55750741.00236403,
  "volume1mo": 293573206.2130419,
  "volume1yr": 475338366.2483143,
  "enableOrderBook": true,
  "liquidityClob": 70493960.6881,
  "negRisk": true,
  "negRiskMarketID": "0x4714f4189125bba4cb9e6f9e8b5757ebd34a5be31379c33a665e4b0ca9738600",
  "commentCount": 1553,
  "cyom": false,
  "showAllOutcomes": true,
  "showMarketImages": true,
  "automaticallyResolved": false,
  "enableNegRisk": true,
  "automaticallyActive": true,
  "gmpChartMode": "default",
  "negRiskAugmented": true,
  "estimateValue": false,
  "cumulativeMarkets": false,
  "pendingDeployment": false,
  "deploying": false,
  "requiresTranslation": false,
  "markets": [ /* array of Market objects */ ],
  "tags": [ /* array of Tag objects */ ]
}
```

**All Event Fields:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique event ID |
| `ticker` | string | URL-friendly ticker |
| `slug` | string | URL slug |
| `title` | string | Display title |
| `subtitle` | string? | Optional subtitle |
| `description` | string | Full description with resolution criteria |
| `resolutionSource` | string | URL or description of resolution source |
| `startDate` | datetime | When trading starts |
| `endDate` | datetime | When trading ends |
| `creationDate` | datetime | When event was created |
| `closedTime` | datetime? | When event was closed (null if open) |
| `finishedTimestamp` | datetime? | When event finished |
| `publishedAt` | datetime? | When event was published |
| `createdAt` | datetime | DB creation timestamp |
| `updatedAt` | datetime | DB update timestamp |
| `image` | string | Full-size image URL |
| `icon` | string | Icon image URL |
| `featuredImage` | string? | Featured/hero image URL |
| `active` | boolean | Currently active |
| `closed` | boolean | Trading is closed |
| `archived` | boolean | Event is archived |
| `new` | boolean | Recently created |
| `featured` | boolean | Featured on homepage |
| `restricted` | boolean | Geo-restricted |
| `live` | boolean? | Live event (sports) |
| `ended` | boolean? | Event has ended (sports) |
| `liquidity` | number | Total liquidity (USD) |
| `liquidityAmm` | number? | AMM liquidity |
| `liquidityClob` | number | CLOB liquidity |
| `volume` | number | Total volume (USD) |
| `volume24hr` | number | 24-hour volume |
| `volume1wk` | number | 1-week volume |
| `volume1mo` | number | 1-month volume |
| `volume1yr` | number | 1-year volume |
| `openInterest` | number | Open interest |
| `competitive` | number | Competitiveness score (0-1) |
| `commentCount` | integer | Number of comments |
| `tweetCount` | integer? | Number of tweets |
| `category` | string? | Primary category |
| `subcategory` | string? | Subcategory |
| `sortBy` | string | Default sort for markets |
| `negRisk` | boolean | Uses neg-risk model |
| `negRiskMarketID` | string? | Shared neg-risk market ID |
| `negRiskFeeBips` | integer? | Neg-risk fee (basis points) |
| `enableNegRisk` | boolean? | Neg-risk enabled flag |
| `negRiskAugmented` | boolean? | Neg-risk augmented flag |
| `enableOrderBook` | boolean | Order book enabled |
| `cyom` | boolean | Create-your-own-market |
| `showAllOutcomes` | boolean | Show all outcomes in UI |
| `showMarketImages` | boolean | Show market images in UI |
| `automaticallyResolved` | boolean | Auto-resolved |
| `automaticallyActive` | boolean | Auto-activated |
| `gmpChartMode` | string | Chart mode (default/manual) |
| `estimateValue` | boolean | Allow value estimation |
| `cumulativeMarkets` | boolean | Markets are cumulative |
| `commentsEnabled` | boolean? | Comments enabled |
| `pendingDeployment` | boolean | Awaiting deployment |
| `deploying` | boolean | Currently deploying |
| `requiresTranslation` | boolean | Needs translation |
| `isTemplate` | boolean? | Is an event template |
| `templateVariables` | string? | Template variables |
| **Nested Objects** | | |
| `markets` | Market[] | Array of child markets |
| `series` | Series[]? | Associated series |
| `tags` | Tag[] | Associated tags |
| `categories` | Category[]? | Associated categories |
| `collections` | Collection[]? | Associated collections |
| `eventCreators` | EventCreator[]? | Creator info |
| `chats` | Chat[]? | Chat channels (if requested) |
| `templates` | Template[]? | Templates (if requested) |
| `imageOptimized` | object? | Optimized image metadata |
| `iconOptimized` | object? | Optimized icon metadata |
| `featuredImageOptimized` | object? | Optimized featured image |
| **Sports-specific** | | |
| `score` | string? | Current score |
| `elapsed` | string? | Elapsed time |
| `period` | string? | Current period |
| `gameStatus` | string? | Game status |
| `eventDate` | string? | Event date |
| `eventWeek` | string? | Event week |
| `seriesSlug` | string? | Parent series slug |
| `startTime` | datetime? | Game start time |

### Market Object (Complete)

```json
{
  "id": "572469",
  "question": "Will Trump nominate Kevin Warsh as the next Fed chair?",
  "conditionId": "0x61b66d02793b4a68ab0cc25be60d65f517fe18c7d654041281bb130341244fcc",
  "slug": "will-trump-nominate-kevin-warsh-as-the-next-fed-chair",
  "resolutionSource": "",
  "endDate": "2026-12-31T00:00:00Z",
  "liquidity": "725490.77341",
  "startDate": "2025-08-05T17:28:15.516Z",
  "image": "https://polymarket-upload.s3.us-east-2.amazonaws.com/...",
  "icon": "https://polymarket-upload.s3.us-east-2.amazonaws.com/...",
  "description": "This market will resolve according to...",
  "outcomes": "[\"Yes\", \"No\"]",
  "outcomePrices": "[\"0.9635\", \"0.0365\"]",
  "volume": "38853330.006589",
  "active": true,
  "closed": false,
  "marketMakerAddress": "",
  "createdAt": "2025-08-05T17:13:19.02272Z",
  "updatedAt": "2026-02-14T14:11:25.841883Z",
  "new": false,
  "featured": false,
  "submitted_by": "0x91430CaD2d3975766499717fA0D66A78D814E5c5",
  "archived": false,
  "resolvedBy": "0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d",
  "restricted": true,
  "groupItemTitle": "Kevin Warsh",
  "groupItemThreshold": "0",
  "questionID": "0x4714f4189125bba4cb9e6f9e8b5757ebd34a5be31379c33a665e4b0ca9738600",
  "enableOrderBook": true,
  "orderPriceMinTickSize": 0.001,
  "orderMinSize": 5,
  "umaResolutionStatus": "disputed",
  "volumeNum": 38853330.006589,
  "liquidityNum": 725490.77341,
  "endDateIso": "2026-12-31",
  "startDateIso": "2025-08-05",
  "hasReviewedDates": true,
  "volume24hr": 1009555.243004,
  "volume1wk": 6001408.542862,
  "volume1mo": 33834932.514202,
  "volume1yr": 38713932.546478,
  "clobTokenIds": "[\"51338236787729560681434534660841415073585974762690814047670810862722808070955\", \"18289842382539867639079362738467334752951741961393928566628307174343542320349\"]",
  "umaBond": "500",
  "umaReward": "5",
  "volume24hrClob": 1009555.243004,
  "volume1wkClob": 6001408.542862,
  "volume1moClob": 33834932.514202,
  "volume1yrClob": 38713932.546478,
  "volumeClob": 38853330.006589,
  "liquidityClob": 725490.77341,
  "customLiveness": 0,
  "acceptingOrders": true,
  "negRisk": true,
  "negRiskMarketID": "0x4714f4189125bba4cb9e6f9e8b5757ebd34a5be31379c33a665e4b0ca9738600",
  "negRiskRequestID": "0x0c8d400454512a35df3a2c2e59af4fd4ae2d4d48b6ae35bd0763c20800df6bbb",
  "ready": false,
  "funded": false,
  "acceptingOrdersTimestamp": "2025-08-05T17:27:53Z",
  "cyom": false,
  "competitive": 0.823158917620,
  "pagerDutyNotificationEnabled": false,
  "approved": true,
  "rewardsMinSize": 100,
  "rewardsMaxSpread": 3.5,
  "spread": 0.001,
  "oneDayPriceChange": 0.014,
  "oneWeekPriceChange": 0.0075,
  "oneMonthPriceChange": 0.5385,
  "lastTradePrice": 0.964,
  "bestBid": 0.963,
  "bestAsk": 0.964,
  "automaticallyActive": true,
  "clearBookOnStart": true,
  "negRiskOther": false,
  "pendingDeployment": false,
  "deploying": false,
  "rfqEnabled": false,
  "holdingRewardsEnabled": false,
  "feesEnabled": false,
  "requiresTranslation": false,
  "feeType": null,
  "events": [ /* parent Event objects (lightweight) */ ]
}
```

**All Market Fields:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique market ID (numeric string) |
| `question` | string | Market question text |
| `conditionId` | string | On-chain condition ID (0x-prefixed hex) |
| `slug` | string | URL slug |
| `questionID` | string | On-chain question ID (shared in neg-risk groups) |
| `description` | string | Full description with resolution criteria |
| `resolutionSource` | string | Resolution source URL |
| `category` | string? | Category label |
| `marketType` | string? | Market type ("normal", etc.) |
| `formatType` | string? | Format type |
| `ammType` | string? | AMM type |
| **Outcomes & Pricing** | | |
| `outcomes` | string | JSON-encoded array: `'["Yes", "No"]'` |
| `outcomePrices` | string | JSON-encoded array: `'["0.9635", "0.0365"]'` |
| `lastTradePrice` | number | Last trade price (0-1) |
| `bestBid` | number | Best bid price |
| `bestAsk` | number | Best ask price |
| `spread` | number | Bid-ask spread |
| `oneDayPriceChange` | number | 24h price change |
| `oneHourPriceChange` | number? | 1h price change |
| `oneWeekPriceChange` | number | 7d price change |
| `oneMonthPriceChange` | number | 30d price change |
| `oneYearPriceChange` | number? | 1y price change |
| **Volume** | | |
| `volume` | string | Total volume (string, USD) |
| `volumeNum` | number | Total volume (number, USD) |
| `volume24hr` | number | 24-hour volume |
| `volume1wk` | number | 1-week volume |
| `volume1mo` | number | 1-month volume |
| `volume1yr` | number | 1-year volume |
| `volumeClob` | number | CLOB-only volume |
| `volume24hrClob` | number | CLOB 24h volume |
| `volume1wkClob` | number | CLOB 1-week volume |
| `volume1moClob` | number | CLOB 1-month volume |
| `volume1yrClob` | number | CLOB 1-year volume |
| `volume1wkAmm` | number? | AMM 1-week volume |
| `volume1moAmm` | number? | AMM 1-month volume |
| `volume1yrAmm` | number? | AMM 1-year volume |
| **Liquidity** | | |
| `liquidity` | string | Total liquidity (string, USD) |
| `liquidityNum` | number | Total liquidity (number, USD) |
| `liquidityClob` | number | CLOB liquidity |
| **Status** | | |
| `active` | boolean | Market is active |
| `closed` | boolean | Market is closed |
| `archived` | boolean | Market is archived |
| `new` | boolean | Recently created |
| `featured` | boolean | Featured on homepage |
| `restricted` | boolean | Geo-restricted |
| `approved` | boolean | Approved for trading |
| `ready` | boolean | Ready flag |
| `funded` | boolean | Funded flag |
| `acceptingOrders` | boolean | Currently accepting orders |
| `pendingDeployment` | boolean | Awaiting deployment |
| `deploying` | boolean | Currently deploying |
| **Dates** | | |
| `startDate` | datetime | Trading start |
| `endDate` | datetime | Trading end |
| `startDateIso` | string | Start date (YYYY-MM-DD) |
| `endDateIso` | string | End date (YYYY-MM-DD) |
| `createdAt` | datetime | DB creation |
| `updatedAt` | datetime | DB update |
| `closedTime` | datetime? | When closed |
| `acceptingOrdersTimestamp` | datetime? | When orders started |
| `deployingTimestamp` | datetime? | Deployment timestamp |
| `eventStartTime` | datetime? | Event start (sports) |
| `umaEndDate` | datetime? | UMA resolution deadline |
| **Trading Config** | | |
| `enableOrderBook` | boolean | Order book enabled |
| `orderPriceMinTickSize` | number | Min price tick (e.g., 0.001) |
| `orderMinSize` | number | Min order size (e.g., 5) |
| `fee` | string? | Fee rate |
| `denominationToken` | string? | Denomination token |
| `makerBaseFee` | number? | Maker fee |
| `takerBaseFee` | number? | Taker fee |
| `feesEnabled` | boolean | Fees enabled |
| `feeType` | string? | Fee type |
| `rfqEnabled` | boolean | RFQ enabled |
| `rewardsMinSize` | number | Min size for rewards |
| `rewardsMaxSpread` | number | Max spread for rewards |
| `holdingRewardsEnabled` | boolean | Holding rewards |
| **Identifiers** | | |
| `marketMakerAddress` | string | Market maker contract address |
| `submitted_by` | string | Submitter address |
| `resolvedBy` | string | Resolver address |
| `clobTokenIds` | string | JSON-encoded array of CLOB token IDs (large integers) |
| **Resolution** | | |
| `umaResolutionStatus` | string? | "proposed", "disputed", "resolved", etc. |
| `umaResolutionStatuses` | string? | JSON array of historical statuses |
| `umaBond` | string? | UMA bond amount |
| `umaReward` | string? | UMA reward amount |
| `automaticallyResolved` | boolean? | Auto-resolved |
| **Group/Multi-outcome** | | |
| `groupItemTitle` | string? | Display title within group (e.g., "Kevin Warsh") |
| `groupItemThreshold` | string? | Threshold value |
| `negRisk` | boolean | Part of neg-risk group |
| `negRiskMarketID` | string? | Shared neg-risk market ID |
| `negRiskRequestID` | string? | Per-market neg-risk request ID |
| `negRiskOther` | boolean | Is the "Other" outcome in neg-risk |
| **Sports-specific** | | |
| `gameId` | string? | Sports game ID |
| `sportsMarketType` | string? | Sports market type |
| `teamAID` | string? | Team A ID |
| `teamBID` | string? | Team B ID |
| `line` | string? | Betting line |
| **Display** | | |
| `image` | string | Market image URL |
| `icon` | string | Market icon URL |
| `twitterCardImage` | string? | Twitter card image |
| `competitive` | number | Competitiveness score (0-1) |
| `seriesColor` | string? | Display color for series |
| `showGmpSeries` | boolean | Show in GMP series |
| `showGmpOutcome` | boolean | Show as GMP outcome |
| `clearBookOnStart` | boolean | Clear order book on start |
| `automaticallyActive` | boolean | Auto-activated |
| `manualActivation` | boolean | Requires manual activation |
| `cyom` | boolean | Create-your-own-market |
| `pagerDutyNotificationEnabled` | boolean | PagerDuty alerts |
| `hasReviewedDates` | boolean | Dates reviewed |
| `requiresTranslation` | boolean | Needs translation |
| **Nested** | | |
| `events` | Event[] | Parent event(s) |
| `tags` | Tag[]? | Associated tags |
| `categories` | Category[]? | Associated categories |
| `imageOptimized` | object? | Optimized image |
| `iconOptimized` | object? | Optimized icon |

### Tag Object

```json
{
  "id": "120",
  "label": "Finance",
  "slug": "finance",
  "forceShow": false,
  "publishedAt": "2023-11-02 21:22:21.615+00",
  "createdAt": "2023-11-02T21:22:21.62Z",
  "updatedAt": "2026-02-06T20:12:40.097768Z",
  "isCarousel": false,
  "requiresTranslation": false
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique tag ID |
| `label` | string | Display label |
| `slug` | string | URL slug |
| `forceShow` | boolean? | Force show in UI |
| `forceHide` | boolean? | Force hide in UI |
| `isCarousel` | boolean? | Show in carousel |
| `publishedAt` | string? | Publication date |
| `createdBy` | integer? | Creator user ID |
| `updatedBy` | integer? | Updater user ID |
| `createdAt` | datetime | Creation timestamp |
| `updatedAt` | datetime | Update timestamp |
| `requiresTranslation` | boolean | Needs translation |

### TagRelationship Object

```json
{
  "id": "21832",
  "tagID": 120,
  "relatedTagID": 1013,
  "rank": 1
}
```

### Series Object

```json
{
  "id": "39",
  "ticker": "march-madness",
  "slug": "march-madness",
  "title": "March Madness Games",
  "seriesType": "single",
  "recurrence": "daily",
  "image": "https://polymarket-upload.s3.us-east-2.amazonaws.com/marchmadness.jpeg",
  "icon": "https://polymarket-upload.s3.us-east-2.amazonaws.com/marchmadness.jpeg",
  "layout": "default",
  "active": true,
  "closed": false,
  "archived": false,
  "new": false,
  "featured": false,
  "restricted": true,
  "publishedAt": "2023-03-14 21:21:30.319+00",
  "createdBy": "15",
  "updatedBy": "15",
  "createdAt": "2023-03-14T14:04:50.064Z",
  "updatedAt": "2026-02-14T14:09:57.053233Z",
  "commentsEnabled": false,
  "competitive": "0",
  "commentCount": 0,
  "requiresTranslation": false,
  "events": [ /* array of Event objects */ ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique series ID |
| `ticker` | string | URL-friendly ticker |
| `slug` | string | URL slug |
| `title` | string | Display title |
| `subtitle` | string? | Subtitle |
| `description` | string? | Description |
| `seriesType` | string | Type ("single", etc.) |
| `recurrence` | string | Recurrence pattern ("daily", "weekly", "monthly") |
| `image` | string | Image URL |
| `icon` | string | Icon URL |
| `layout` | string | Layout type ("default", etc.) |
| `active` | boolean | Currently active |
| `closed` | boolean | Closed |
| `archived` | boolean | Archived |
| `new` | boolean | Recently created |
| `featured` | boolean | Featured |
| `restricted` | boolean | Geo-restricted |
| `publishedAt` | string? | Publication date |
| `createdBy` | string? | Creator ID |
| `updatedBy` | string? | Updater ID |
| `createdAt` | datetime | Creation timestamp |
| `updatedAt` | datetime | Update timestamp |
| `commentsEnabled` | boolean | Comments enabled |
| `competitive` | string | Competitiveness score |
| `commentCount` | integer | Comment count |
| `volume24hr` | number? | 24h volume |
| `volume` | number? | Total volume |
| `liquidity` | number? | Total liquidity |
| `score` | string? | Score (sports) |
| `pythTokenID` | string? | Pyth token ID |
| `cgAssetName` | string? | CoinGecko asset name |
| `events` | Event[] | Child events |
| `collections` | Collection[]? | Collections |
| `categories` | Category[]? | Categories |
| `tags` | Tag[]? | Tags |
| `chats` | Chat[]? | Chat channels |

### Team Object (Sports)

```json
{
  "id": 178024,
  "name": "Learner Tien",
  "league": "atp",
  "record": "0-0",
  "logo": "https://polymarket-upload.s3.us-east-2.amazonaws.com/country-flags/usa.png",
  "abbreviation": "tien",
  "createdAt": "2025-10-25T04:00:03.451204Z",
  "updatedAt": "2025-11-07T22:35:56.143285Z",
  "providerId": 1359543,
  "color": "#0749A0"
}
```

### Sport Object

```json
{
  "id": 1,
  "sport": "ncaab",
  "image": "https://polymarket-upload.s3.us-east-2.amazonaws.com/marchmadness.jpeg",
  "resolution": "https://www.ncaa.com/march-madness-live/bracket",
  "ordering": "home",
  "tags": "1,100149,100639",
  "series": "39",
  "createdAt": "2025-11-05T19:27:45.399303Z"
}
```

---

## Pagination

Gamma uses **offset-based pagination** across all list endpoints.

```
Page 1: ?limit=100&offset=0
Page 2: ?limit=100&offset=100
Page 3: ?limit=100&offset=200
```

There is no `total_count` or `has_more` field in the list endpoint responses -- you must paginate until you receive an empty array or fewer results than the `limit`.

**Performance note:** Offset-based pagination is O(n) per page for the backend, so deep pagination (very high offsets) may be slower. For large-scale ingestion, consider filtering by date ranges to limit result sets.

---

## Sorting

The `order` parameter accepts comma-separated field names. Combined with `ascending` (boolean):

**Common sort fields for events:**
- `id`, `volume24hr`, `volume`, `liquidity`, `startDate`, `endDate`, `createdAt`

**Common sort fields for markets:**
- `id`, `volumeNum`, `liquidityNum`, `volume24hr`, `startDate`, `endDate`, `createdAt`, `lastTradePrice`

Example: `?order=volume24hr&ascending=false` (highest 24h volume first)

---

## Key Data Points for Pipeline

### Identifying Resolved Markets
- `closed: true` -- market trading is closed
- `umaResolutionStatus: "resolved"` -- UMA oracle has resolved
- `automaticallyResolved: true` -- resolved automatically
- `outcomePrices: '["1", "0"]'` -- resolved to Yes (first outcome price = 1)
- `outcomePrices: '["0", "1"]'` -- resolved to No

### Volume & Liquidity (all USD-denominated)
- Event-level: `volume`, `volume24hr`, `volume1wk`, `volume1mo`, `volume1yr`, `liquidity`, `liquidityClob`
- Market-level: `volumeNum`, `volume24hr`, `volume1wk`, `volume1mo`, `volume1yr`, `liquidityNum`, `liquidityClob` (plus separate AMM/CLOB breakdowns)

### Price Data
- `lastTradePrice` -- most recent trade price (0-1 range)
- `bestBid`, `bestAsk` -- current order book top-of-book
- `spread` -- bid-ask spread
- `outcomePrices` -- JSON-encoded current prices for all outcomes
- `oneDayPriceChange`, `oneWeekPriceChange`, `oneMonthPriceChange` -- price deltas

### CLOB Token IDs
- `clobTokenIds` is a JSON-encoded array of stringified large integers
- First token = "Yes" outcome, second token = "No" outcome
- These IDs are needed to query the CLOB API for order book data and trade history

### Recommended Ingestion Strategy
1. **Full sync:** Paginate `/events?order=id&ascending=true&limit=100` to get all events with nested markets
2. **Incremental updates:** Use `/events?order=updatedAt&ascending=false&limit=100` to get recently changed events
3. **Active-only:** Add `closed=false&active=true` to focus on live markets
4. **By category:** Use `tag_id` parameter to fetch specific categories (e.g., Finance=120, AI=439)

---

## Important Notes

1. **String-encoded fields:** `outcomes`, `outcomePrices`, and `clobTokenIds` are JSON-encoded strings within the JSON response -- they require double-parsing. Example: `JSON.parse(market.outcomePrices)` yields `["0.9635", "0.0365"]`.

2. **Dual numeric fields:** Many fields exist in both string and numeric forms (e.g., `volume` as string, `volumeNum` as number; `liquidity` as string, `liquidityNum` as number). Use the numeric versions for filtering and computation.

3. **Events contain markets:** When fetching events, the full nested `markets` array is included by default. This is the most efficient way to get both event metadata and market data in one call.

4. **No WebSocket/streaming:** Gamma is REST-only. For real-time price updates, use the CLOB API's WebSocket endpoints instead. Gamma data has some update lag.

5. **Image optimization:** Both events and markets include `imageOptimized` and `iconOptimized` objects with pre-processed image variants at different sizes.

6. **UMA Resolution:** Markets use UMA (Universal Market Access) oracle for resolution. Status progression: proposed -> disputed -> resolved. The `umaResolutionStatuses` field tracks the full history as a JSON array.

---

## Sources

- [Polymarket Gamma API Documentation](https://docs.polymarket.com/developers/gamma-markets-api/overview)
- [Gamma Structure Documentation](https://docs.polymarket.com/developers/gamma-markets-api/gamma-structure)
- [Fetch Markets Guide](https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide)
- [List Markets API Reference](https://docs.polymarket.com/api-reference/markets/list-markets)
- [List Events API Reference](https://docs.polymarket.com/api-reference/events/list-events)
- [Get Market by ID](https://docs.polymarket.com/api-reference/markets/get-market-by-id)
- [Get Event by ID](https://docs.polymarket.com/api-reference/events/get-event-by-id)
- [Go Gamma Client (ivanzzeth)](https://pkg.go.dev/github.com/ivanzzeth/polymarket-go-gamma-client) -- comprehensive endpoint and schema reference
- [Polymarket Agents (Python)](https://github.com/Polymarket/agents/blob/main/agents/polymarket/gamma.py) -- official Python client
- Live API responses verified 2026-02-14
