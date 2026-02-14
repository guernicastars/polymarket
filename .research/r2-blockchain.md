# Research: Polymarket On-Chain / Blockchain User Tracking

## 1. Smart Contract Architecture

Polymarket operates on the **Polygon (PoS)** network. All trading is settled on-chain through two primary exchange contracts:

### Core Exchange Contracts

| Contract | Address | Purpose |
|---|---|---|
| **CTF Exchange** | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary market settlement (YES/NO) |
| **NegRisk CTF Exchange** | `0xC5d563A36AE78145C45a50134d48A1215220f80a` | Multi-outcome market settlement |
| **NegRisk Adapter** | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` | Converts portfolio positions between outcome types |

### Supporting Contracts

| Contract | Address | Purpose |
|---|---|---|
| **Conditional Tokens (CTF)** | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | ERC1155 conditional token framework (Gnosis) |
| **USDC.e** | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Collateral token (bridged USDC on Polygon) |
| **Gnosis Safe Factory** | `0xaacfeea03eb1561c4e67d661e40682bd20e3541b` | Creates Safe proxy wallets for browser wallet users |
| **Polymarket Proxy Factory** | `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052` | Creates proxy wallets for MagicLink/email users |
| **UMA CTF Adapter** | `0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74` | Oracle resolution via UMA |

### How Trading Works

Polymarket uses a **hybrid-decentralized** order book: orders are matched off-chain by an operator, but settlement/execution happens on-chain via the exchange contracts. Users sign EIP-712 typed messages representing limit orders, and the exchange contract performs atomic swaps between outcome tokens (ERC1155) and USDC (ERC20).

---

## 2. On-Chain Events (Trade Events)

The exchange contracts emit specific events that can be used to track all trading activity:

### OrderFilled Event

Emitted for each individual order execution. This is the primary event for tracking trades.

**Fields:**
- `orderHash` (indexed) -- Unique hash of the order being filled
- `maker` (indexed) -- Liquidity provider placing the limit order
- `taker` (indexed) -- User filling the order, OR the Exchange contract if matching multiple orders
- `makerAssetId` -- Asset ID (0 = USDC; large integer = conditional token ID)
- `takerAssetId` -- Inverse of makerAssetId
- `makerAmountFilled` -- Quantity provided by maker
- `takerAmountFilled` -- Quantity provided by taker
- `fee` -- Fee charged

**Trade direction:** If `makerAssetId == 0`, it's a BUY order (maker gives USDC, receives tokens). Otherwise, it's a SELL order.

### OrdersMatched Event

Emitted when multiple orders are matched together in a single transaction.

**Fields:**
- `takerOrderHash` -- Identifier for the taker's order
- `takerOrderMaker` -- Address of taker's order creator
- `makerAssetId`, `takerAssetId` -- Assets involved
- `makerAmountFilled`, `takerAmountFilled` -- Settlement amounts

### Position Management Events (from CTF / NegRiskAdapter)

- **PositionsSplit** -- When collateral is converted into YES+NO token pairs (token minting)
- **PositionsMerge** -- When YES+NO token pairs are destroyed to release collateral (token burning)
- **PositionsConverted** -- Multi-outcome portfolio rebalancing (uses indexSet bitmask)

### Volume Double-Counting Warning

**Critical methodology note** (per Paradigm research, Dec 2025): Most dashboards double-count Polymarket volume by summing ALL `OrderFilled` events. Each trade generates TWO sets of OrderFilled events -- one maker-focused, one taker-focused -- representing the SAME economic activity.

**Correct approach:** Filter to one side only:
- **Taker-side volume:** Sum `OrderFilled` events where `taker` equals either exchange contract address
- **Maker-side volume:** Sum `OrderFilled` events where `taker` does NOT equal exchange contract addresses

Both approaches yield ~50% of commonly reported figures.

---

## 3. Proxy Wallet Architecture (Linking Wallets to Users)

### How User Wallets Work

When a user first trades on Polymarket.com, a **1-of-1 multisig smart contract wallet** is deployed to Polygon. This "proxy wallet" holds all user positions (ERC1155 tokens) and USDC.

Two types exist depending on authentication method:

| Type | Factory Address | Auth Method | Users |
|---|---|---|---|
| **Gnosis Safe** | `0xaacfeea03eb1561c4e67d661e40682bd20e3541b` | Browser wallets (MetaMask, Rainbow, Coinbase) | 1-of-1 modified Gnosis Safe |
| **Polymarket Proxy** | `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052` | MagicLink / Email accounts | EIP-1167 minimal proxy |

### Deterministic Address Derivation

Both wallet types use **CREATE2** for deterministic address derivation from the user's EOA:
- Safe address = `getCreate2Address(factory, salt=keccak256(EOA), initCodeHash)`
- Proxy address = similar CREATE2 derivation from EOA via Proxy Factory

This means: **given an EOA address, you can deterministically compute the proxy wallet address** without querying the chain.

### Linking Wallet to Profile

The Polymarket Gamma API provides a profile lookup endpoint:
```
GET https://gamma-api.polymarket.com/profiles/{wallet_address}
```
This returns profile data including: name, bio, profile image, proxy wallet address, and pseudonym.

### On-Chain Wallet Tracking

- All user trading activity happens through their proxy wallet
- The proxy wallet's EOA owner is the signer (visible in transaction `from` field or Safe execution events)
- Factory creation events on the two factory contracts can enumerate all Polymarket user wallets
- ERC1155 `TransferSingle`/`TransferBatch` events on the Conditional Tokens contract show position changes

---

## 4. The Graph Subgraphs

Polymarket maintains an open-source subgraph repository with **7 specialized subgraphs** for querying on-chain data via GraphQL.

### GitHub Repository
https://github.com/Polymarket/polymarket-subgraph

### The Graph Decentralized Network Endpoint
```
https://gateway.thegraph.com/api/{api-key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp
```
Free tier: 100K queries/month via The Graph Studio.

### Goldsky-Hosted Endpoints (Public, No Auth Required)

| Subgraph | Endpoint URL | Key Data |
|---|---|---|
| **Activity** | `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn` | Trades, events, transaction history |
| **Positions** | `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn` | User positions, balance, average price, realized PnL |
| **Orderbook** | `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn` | Orderbook depth, spreads, last trade price |
| **Open Interest** | `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/oi-subgraph/0.0.6/gn` | Market and global open interest |
| **PnL** | `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn` | Profit/loss calculations |

Additional subgraphs in the repo: `fpmm-subgraph` (fixed product market maker), `sports-oracle-subgraph`, `polymarket-subgraph` (primary).

### Example GraphQL Queries

**Get user positions by wallet:**
```graphql
{
  positions(where: { user: "0xYourWalletAddress" }, first: 10) {
    id
    condition
    outcomeIndex
    balance
    averagePrice
    realizedPnl
  }
}
```

**Top 5 highest payouts (redemptions):**
```graphql
{
  redemptions(orderBy: payout, orderDirection: desc, first: 5) {
    payout
    redeemer
    id
    timestamp
  }
}
```

**Orderbook data:**
```graphql
{
  orderBooks(first: 10, orderBy: timestamp, orderDirection: desc) {
    marketId
    currentSpread
    lastTradePrice
    totalBidDepth
    totalAskDepth
  }
}
```

---

## 5. Dune Analytics

### Key Dashboards

| Dashboard | URL | Focus |
|---|---|---|
| **Polymarket** (rchen8) | https://dune.com/rchen8/polymarket | Comprehensive volume, users, activity |
| **Activity and Volume** (filarm) | https://dune.com/filarm/polymarket-activity | Weekly volume breakdowns |
| **Polymarket on Polygon** (petertherock) | https://dune.com/petertherock/polymarket-on-polygon | Polygon-specific metrics |
| **CLOB Stats** (lifewillbeokay) | https://dune.com/lifewillbeokay/polymarket-clob-stats | Order book statistics |
| **Trade Activity Tracker** (0xclark_kent) | https://dune.com/0xclark_kent/polymarket-trade-activity-tracker | Per-trade analysis |
| **User Activity Analyzer** (genejp999) | https://dune.com/genejp999/polymarket-user-activity-analyzer | User behavior analysis |
| **Address Tracker / Airdrop** (seoul) | https://dune.com/seoul/poly | Address-level tracking |
| **Polymarket Overview** (datadashboards) | https://dune.com/datadashboards/polymarket-overview | High-level overview |

### Dune Table Structure

Polymarket data on Dune is available through **decoded event tables** on Polygon:

- `polymarket_polygon.CTFExchange_evt_OrderFilled` -- Every order fill on CTF Exchange
- `polymarket_polygon.CTFExchange_evt_OrdersMatched` -- Matched orders
- `polymarket_polygon.NegRiskCTFExchange_evt_OrderFilled` -- NegRisk order fills
- `polymarket_polygon.NegRiskCTFExchange_evt_OrdersMatched` -- NegRisk matched orders

Raw tables also available: `polygon.logs`, `polygon.transactions`, `polygon.traces`

### Example Dune SQL Pattern

```sql
-- Daily unique traders (correct one-sided counting)
SELECT
  date_trunc('day', evt_block_time) AS day,
  COUNT(DISTINCT maker) AS unique_makers,
  COUNT(DISTINCT taker) AS unique_takers
FROM polymarket_polygon.CTFExchange_evt_OrderFilled
WHERE taker != 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E  -- maker-side only
GROUP BY 1
ORDER BY 1 DESC
```

### Dune LiveFetch Functions

Dune also supports **LiveFetch** functions that can pull data directly from Polymarket's REST APIs in real-time, which was highlighted as a success story for combining on-chain and off-chain data.

---

## 6. Bitquery GraphQL API

Bitquery provides an alternative GraphQL API for querying Polymarket on-chain data on Polygon, with decoded event data.

### OrderFilled Events Query
```graphql
{
  EVM(dataset: realtime, network: matic) {
    Events(
      orderBy: { descending: Block_Time }
      where: {
        Log: { Signature: { Name: { in: ["OrderFilled"] } } }
        LogHeader: {
          Address: {
            in: [
              "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
              "0xC5d563A36AE78145C45a50134d48A1215220f80a"
            ]
          }
        }
      }
      limit: { count: 20 }
    ) {
      Block { Time Number }
      Transaction { Hash From To }
      Arguments {
        Name
        Value {
          ... on EVM_ABI_Integer_Value_Arg { integer }
          ... on EVM_ABI_Address_Value_Arg { address }
          ... on EVM_ABI_BigInt_Value_Arg { bigInteger }
        }
      }
    }
  }
}
```

### DEX Trades Query (Higher-Level Abstraction)
Bitquery also provides a `DEXTradeByTokens` entity with `Dex: {ProtocolName: {is: "polymarket"}}` filter, giving trade amounts in USD, token symbols, and order IDs.

---

## 7. Direct RPC / Web3 Approach

For custom analysis, you can query Polygon RPC directly:

```python
from web3 import Web3

web3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com/"))
web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

# Get OrderFilled logs from CTF Exchange
logs = web3.eth.get_logs({
    'fromBlock': start_block,
    'toBlock': end_block,
    'address': '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E'
})

# Decode using ABI
# Load ABI from PolygonScan verified contract
# Create event signature mapping: keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
```

**Recommended RPC providers:** Alchemy, Infura, QuickNode (all support Polygon).

---

## 8. Summary: What Can Be Tracked On-Chain

### Per-User Data Available
- **All trades**: Every buy/sell with exact amounts, prices, token IDs, timestamps (via OrderFilled events)
- **Current positions**: Which markets a wallet holds, outcome side, balance (via Positions subgraph or ERC1155 balance queries)
- **PnL**: Realized profit/loss per position (via PnL subgraph)
- **Trade history**: Full activity timeline (via Activity subgraph)
- **Wallet creation**: When a user first created their Polymarket wallet (factory contract events)
- **Profile data**: Username, bio, profile image linked to wallet (via Gamma API `/profiles/{address}`)

### What Cannot Be Tracked On-Chain
- **Order placement/cancellation**: Orders are placed off-chain; only fills/matches are on-chain
- **Unfilled limit orders**: Only visible via the CLOB API, not on-chain
- **User identity beyond profile**: No KYC data is on-chain
- **IP addresses / geographic data**: Not blockchain data

### Recommended Approach for User Tracking Pipeline

1. **Enumerate wallets**: Query factory contract `ProxyCreation` events to build a wallet registry
2. **Link to profiles**: Batch-query Gamma API `/profiles/{address}` for each wallet
3. **Track trades**: Index `OrderFilled` events from both CTF Exchange and NegRisk Exchange contracts
4. **Compute positions**: Use the Positions subgraph (Goldsky) or compute from trade history
5. **Calculate PnL**: Use the PnL subgraph or compute from entry/exit prices + redemptions
6. **Monitor in real-time**: Subscribe to Bitquery streaming API or use The Graph subgraph for near-real-time updates
