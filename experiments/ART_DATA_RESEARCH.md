# Art Market Data Research

Comprehensive catalog of all art auction data available in the Maynard Metrics repository for the Orth-SVAE embedding experiment.

## Executive Summary

| Metric | Sotheby's | Christie's | Phillips | Total |
|--------|-----------|------------|----------|-------|
| Total lots | 99,971 | 691,634 | 101,378 | 892,983 |
| Auctions | 2,394 | 4,447 | 0 (metadata) | 6,841 |
| Silver extractions | 88,620 | 388,368 | 81,876 | 558,864 |
| Gold features | 88,620 | 388,368 | 81,876 | 558,864 |
| With hammer price | 39,860 | 667,912 | 2,957 | 710,729 |
| Sold lots | 38,995 | 655,865 | 2,032 | 696,892 |
| Unique artists | 17,768 | 55,773 | 7,065 | ~80,606 |

**Best source for ML experiment: Christie's** -- 388K silver extractions, 264K with artist + medium + price, and 667K with hammer prices.

---

## Data Source 1: ClickHouse Cloud (Primary)

**Connection:** `https://ch.bloomsburytech.com:443`
**JDBC:** `jdbc:clickhouse://ch.bloomsburytech.com:443/{db}?user=default&password=clickhouse_admin_2026&ssl=true`
**Credentials:** user=`default`, password=`clickhouse_admin_2026`

### Databases

| Database | Purpose | Key Tables |
|----------|---------|------------|
| `sothebys` | Sotheby's auction data (full pipeline) | lots, sales, silver_extractions, gold_features, raw_lots, auctions |
| `christies` | Christie's auction data (full pipeline) | lots, sales, silver_extractions, gold_features, raw_lots, auctions |
| `phillips` | Phillips auction data (partial pipeline) | lots, sales, silver_extractions, gold_features, raw_lots |
| `auction_19th_century_american_and_western_art` | Single auction (110 lots) | lots, auctions, lot_images, pages_raw |
| `collector_connoisseur_the_max_n_berry_collections_american_art_d` | Single auction (75 lots) | lots, auctions |
| `polymarket` | Prediction market data (Polymarket) | markets, market_trades, etc. |
| `domains` | Harmonic company data (unrelated) | harmonic_* tables |

---

## Sotheby's Database (`sothebys`)

### Tables & Row Counts

| Table | Rows | Engine | Purpose |
|-------|------|--------|---------|
| `lots` | 99,971 | MergeTree | Lot metadata (title, creator, estimates) |
| `sales` | 99,971 | MergeTree | Sale results (hammer, final, bids) |
| `raw_lots` | 99,742 | MergeTree | Full raw data (descriptions, provenance, images) |
| `silver_extractions` | 88,620 | MergeTree | LLM-enriched structured data |
| `gold_features` | 88,620 | MergeTree | ML-ready features |
| `auctions` | 2,394 | MergeTree | Auction metadata |
| `raw_auctions` | 2,390 | MergeTree | Raw auction data |
| `artist_mappings` | 2,103 | MergeTree | Artist ID encoding |
| `fx_rates` | 16,854 | MergeTree | Currency conversion rates |
| `bronze_lot_pages` | 99,745 | MergeTree | Raw HTML pages |

### Schema Details

#### `lots`
```sql
lot_uuid String (PK)
auction_uuid String
lot_number Nullable(Int32)
title Nullable(String)
subtitle Nullable(String)
creator Nullable(String)
slug Nullable(String)
estimate_low Nullable(Float64)
estimate_high Nullable(Float64)
tags Nullable(String)
accepts_crypto UInt8
withdrawn_state Nullable(String)
```

#### `sales`
```sql
lot_uuid String (PK)
is_closed UInt8
reserve_met Nullable(UInt8)
num_bids Nullable(Int32)
starting_bid Nullable(Float64)
hammer_price Nullable(Float64)
final_price Nullable(Float64)
currency Nullable(String)
is_sold UInt8
bid_method Nullable(String)
closing_time Nullable(DateTime)
```

#### `silver_extractions` (Sotheby's schema)
```sql
lot_uuid String (PK)
auction_uuid String
-- Artist
artist_name Nullable(String)
artist_nationality Nullable(String)
artist_birth_year Nullable(Int32)
artist_death_year Nullable(Int32)
artist_name_confidence Float64
artist_nationality_confidence Float64
artist_dates_confidence Float64
-- Dimensions
height_cm Nullable(Float64)
width_cm Nullable(Float64)
depth_cm Nullable(Float64)
dimensions_confidence Float64
-- Medium
medium Nullable(String)
support Nullable(String)
medium_confidence Float64
-- Creation
creation_year Nullable(Int32)
creation_is_approximate UInt8
creation_period Nullable(String)
creation_confidence Float64
creation_source String  -- 'regex' or 'llm'
-- Enrichment
tax_bonded UInt8
tax_vat UInt8
is_guaranteed UInt8
regime_1031 UInt8
hammer_price_usd Nullable(Float64)
raw_description Nullable(String)
```

#### `gold_features` (Sotheby's schema)
```sql
lot_uuid String (PK)
surface_area_cm2 Nullable(Float64)
log_hammer_price Nullable(Float64)
log_surface_area Nullable(Float64)
artist_id Nullable(Int64)
is_rare_artist UInt8
vital_status Nullable(String)  -- 'alive' or 'dead'
```

#### `raw_lots`
```sql
lot_uuid String (PK)
auction_uuid Nullable(String)
lot_number Nullable(String)
title Nullable(String)
subtitle Nullable(String)
description_html Nullable(String)
description_text Nullable(String)
creators_display Nullable(String)
designation_line Nullable(String)
object_type Nullable(String)
estimate_low Nullable(Float64)
estimate_high Nullable(Float64)
currency Nullable(String)
provenance Nullable(String)
literature Nullable(String)
exhibition Nullable(String)
catalogue_note Nullable(String)
saleroom_notice Nullable(String)
images_json Nullable(String)
raw_json Nullable(String)
```

### Data Coverage (Sotheby's silver_extractions: 88,620 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| artist_name | 58,688 | 66.2% |
| artist_nationality | 13,926 | 15.7% |
| artist_birth_year | 40,665 | 45.9% |
| artist_death_year | 29,173 | 32.9% |
| height_cm | 46,657 | 52.6% |
| width_cm | 40,754 | 46.0% |
| depth_cm | 4,958 | 5.6% |
| medium | 78,053 | 88.1% |
| support | 52,123 | 58.8% |
| creation_year | 68,785 | 77.6% |
| hammer_price_usd | 29,515 | 33.3% |

### Data Coverage (Sotheby's raw_lots: 99,742 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| description_text | 99,735 | 99.99% |
| provenance | 51,148 | 51.3% |
| literature | 19,537 | 19.6% |
| exhibition | 14,571 | 14.6% |
| catalogue_note | 21,871 | 21.9% |
| object_type | 99,742 | 100% |
| images_json | 99,737 | 99.99% |
| creators_display | 39,074 | 39.2% |
| designation_line | 37,158 | 37.2% |

### Date Range
- Sales closing_time: 2018-05-24 to 2026-01-08
- Peak years: 2021 (3,041), 2022 (2,766), 2023 (3,458), 2024 (2,708), 2025 (2,599)

### Locations
New York (916), London (646), Hong Kong (386), Paris (335), Milan (35), Geneva (27), Cologne (26)

### Currencies
USD (918), GBP (646), EUR (397), HKD (386), CHF (36), SGD (9), CNY (2)

### Price Distribution (USD, Sotheby's)
- Min: $0.97, Max: $21,000,000
- Mean: $22,680, Median: $6,384
- Q25: $2,495, Q75: $18,000

### Departments (top 10)
Wine (276), Contemporary Art (264), Chinese Works of Art (162), Books & Manuscripts (160), Watches (82), Spirits (77), 20th Century Design (76), American Art (65), Jewellery (62), 19th Century Paintings (61)

### Top Media (Sotheby's)
oil (13,284), wine (2,999), bronze (2,964), ink (2,151), acrylic (1,843), porcelain (1,256), silver (1,074), lithograph (1,042), print (1,026), whisky (992)

---

## Christie's Database (`christies`)

### Tables & Row Counts

| Table | Rows | Engine | Purpose |
|-------|------|--------|---------|
| `lots` | 691,634 | ReplacingMergeTree | Lot metadata |
| `sales` | 691,634 | ReplacingMergeTree | Sale results |
| `raw_lots` | 691,634 | ReplacingMergeTree | Raw text data |
| `silver_extractions` | 388,368 | MergeTree | Structured extraction |
| `gold_features` | 388,368 | MergeTree | ML features |
| `auctions` | 4,447 | ReplacingMergeTree | Auction metadata |
| `artist_mappings` | (present) | -- | Artist encoding |
| `bronze_lot_pages` | (present) | -- | Raw HTML |
| `fx_rates` | (present) | -- | FX rates |

### Schema Differences from Sotheby's

#### `silver_extractions` (Christie's schema -- DIFFERENT columns)
```sql
lot_uuid String (PK)
url String
extracted_at DateTime
lot_category Nullable(String)        -- Category (painting, wine, jewelry, etc.)
lot_title Nullable(String)
creator_name Nullable(String)        -- vs artist_name in Sotheby's
creator_type Nullable(String)
creator_birth_year Nullable(Int32)
creator_death_year Nullable(Int32)
creator_nationality Nullable(String)
date_created Nullable(String)        -- Free text, not integer year
medium Nullable(String)
dimensions_cm Nullable(String)       -- Free text, not parsed h/w/d
dimensions_in Nullable(String)
signed_inscribed Nullable(String)
origin Nullable(String)
style_period Nullable(String)
provenance Array(String)             -- Array vs free text in Sotheby's
exhibitions Array(String)
literature Array(String)
condition_summary Nullable(String)
lot_essay_excerpt Nullable(String)   -- Additional rich text
special_notes Nullable(String)
```

#### `gold_features` (Christie's schema -- DIFFERENT columns)
```sql
lot_uuid String (PK)
surface_area_cm2 Nullable(Float64)
log_estimate_low Nullable(Float64)   -- Has estimate logs (NOT hammer price log)
log_estimate_high Nullable(Float64)
log_surface_area Nullable(Float64)
artist_id Nullable(Int64)            -- All 0 (not populated)
is_rare_artist UInt8
vital_status Nullable(String)
```

**IMPORTANT:** Christie's gold_features does NOT have `log_hammer_price` -- it has `log_estimate_low` and `log_estimate_high` instead. The `artist_id` column is entirely unpopulated (all 0).

### Data Coverage (Christie's silver_extractions: 388,368 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| creator_name | 279,508 | 72.0% |
| creator_birth_year | 128,955 | 33.2% |
| creator_death_year | 96,202 | 24.8% |
| creator_nationality | 39,810 | 10.3% |
| medium | 365,773 | 94.2% |
| dimensions_cm | 64,294 | 16.6% |
| date_created | 325,925 | 83.9% |
| lot_essay_excerpt | 58,599 | 15.1% |
| provenance (non-empty array) | 128,632 | 33.1% |
| condition_summary | 25,093 | 6.5% |

### Data Coverage (Christie's raw_lots: 691,634 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| description_text | 686,295 | 99.2% |
| provenance | 216,540 | 31.3% |
| literature | 92,170 | 13.3% |
| exhibition | 54,762 | 7.9% |
| images_json | 687,763 | 99.4% |

### Data Coverage (Christie's gold_features: 388,368 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| surface_area_cm2 | 44,120 | 11.4% |
| log_estimate_low | 386,535 | 99.5% |
| log_estimate_high | 386,537 | 99.5% |
| artist_id (populated) | 0 | 0% |

### Sale Results
- Total with hammer price: 667,912 (96.6%)
- Sold: 655,865 (94.8%)
- Currencies: USD (244,499), GBP (214,339), HKD (97,326), EUR (81,105), CHF (28,165), CNY (1,540)
- Price range: $1 to $463,600,000
- Mean: $202,491, Median: $8,750

### ML-Ready Record Counts (Christie's)
- Artist + medium + price: **264,178**
- Artist + price: 273,232
- Any price: 378,860

### Lot Categories (Christie's)
painting (106,763), wine (81,587), jewelry (51,137), decorative (37,194), furniture (31,115), asian_art (25,973), other (16,661), sculpture (14,940), book (14,734)

---

## Phillips Database (`phillips`)

### Tables & Row Counts

| Table | Rows | Engine | Purpose |
|-------|------|--------|---------|
| `lots` | 101,378 | ReplacingMergeTree | Lot metadata |
| `sales` | 101,378 | ReplacingMergeTree | Sale results |
| `raw_lots` | 101,378 | ReplacingMergeTree | Raw data |
| `silver_extractions` | 81,876 | MergeTree | Structured extraction |
| `gold_features` | 81,876 | MergeTree | ML features |
| `auctions` | 0 | -- | Empty |

### Schema
Phillips uses the **same schema as Christie's** for silver_extractions and gold_features (creator_name, dimensions_cm as text, etc.)

### Data Coverage (Phillips silver_extractions: 81,876 total)

| Field | Non-null count | Coverage % |
|-------|---------------|------------|
| creator_name | 63,452 | 77.5% |
| creator_birth_year | 14,457 | 17.7% |
| creator_death_year | 8,847 | 10.8% |
| medium | 56,439 | 68.9% |
| dimensions_cm | 12,992 | 15.9% |
| date_created | 53,406 | 65.2% |
| lot_essay_excerpt | 13,361 | 16.3% |
| provenance (non-empty array) | 1,173 | 1.4% |
| condition_summary | 88 | 0.1% |

### Sale Results (Phillips)
- **Mostly unpopulated**: only 2,957 with hammer price, 2,032 sold
- Most sales have NULL currency (94,585 of 101,378)
- Very limited price data for ML

### Lot Categories (Phillips)
painting (16,300), edition (12,094), watch (10,672), photograph (10,593), design (7,710), sculpture (3,055), jewelry (2,631)

---

## Data Source 2: SQLite Databases (Local)

Located at `/Users/ivrejchik/Desktop/art/data_enrich/data/`

### Available Files

| File | Size | Purpose |
|------|------|---------|
| `gold.db` | 15.5 MB | ML-ready features (Sotheby's) |
| `silver.db` | 110.5 MB | Enriched extractions (Sotheby's) |
| `bronze.db` | 19.2 GB | Raw HTML pages |
| `raw.db` | 2.6 GB | Extracted JSON data |
| `sothebys_lots.db` | 311 KB | Auction listings |
| `enrichment.db` | 37 KB | Enrichment metadata |
| `fx_rates.db` | 1.5 MB | FX rate history |

### SQLite Gold Features (88,620 rows)
Same data as ClickHouse `sothebys.gold_features` but with `log_hammer_price` (not estimates).

| Field | Non-null | Coverage % |
|-------|----------|------------|
| surface_area_cm2 | 40,644 | 45.9% |
| log_hammer_price | 29,515 | 33.3% |
| log_surface_area | 40,644 | 45.9% |
| artist_id (known, not rare) | 36,515 | 41.2% |
| vital_status | 88,620 | 100% |

### SQLite Silver Extractions (88,620 rows)
Same data as ClickHouse `sothebys.silver_extractions`.

---

## Data Source 3: Auction Parsers (Source Code)

### Sotheby's (`auction_parser/sothebys/`)
- Playwright-based scraper with cookie-based auth
- Scrapes auction listings and individual lot details
- Outputs to SQLite databases

### Phillips (`auction_parser/philips/`)
- Python scraper (`main.py`)

---

## Views (Pre-joined Queries)

### `sothebys.v_lot_with_sales`
Joins lots + sales + auctions (lot details with sale results and auction context)

### `sothebys.v_silver_with_auction`
Joins silver_extractions + auctions (enriched data with auction context)

### `sothebys.v_gold_with_artist`
Joins gold_features + artist_mappings (ML features with artist names)

---

## Recommendations for Orth-SVAE Experiment

### Primary Dataset: Christie's
- **264,178 lots** with artist + medium + hammer price (largest complete dataset)
- 388K silver extractions with 94.2% medium coverage, 72% artist coverage
- 667K sales with prices (96.6% of all lots)
- Structured `lot_category` field for stratification
- Rich text in `lot_essay_excerpt` (58K) and `provenance` arrays (128K)

### Secondary Dataset: Sotheby's
- **29,515 lots** with hammer_price_usd (normalized to USD)
- Better dimension parsing (separate height/width/depth vs free text in Christie's)
- Gold features include log_hammer_price, surface_area, artist_id
- Rich raw text: description (99.7K), provenance (51K)

### Feature Engineering Priorities

**Numeric features available:**
1. `hammer_price` / `hammer_price_usd` (log-transformed in gold)
2. `estimate_low`, `estimate_high` (log-transformed in Christie's gold)
3. `height_cm`, `width_cm` (Sotheby's parsed; Christie's as free text needing parsing)
4. `surface_area_cm2` (pre-computed in gold)
5. `creation_year` / `date_created`
6. `artist_birth_year`, `artist_death_year`
7. `num_bids`

**Categorical features available:**
1. `artist_name` / `creator_name` (~80K unique across houses)
2. `medium` (~10K unique in Sotheby's)
3. `support` (Sotheby's only)
4. `lot_category` (Christie's/Phillips: painting, wine, jewelry, etc.)
5. `object_type` (Sotheby's raw_lots: Painting, Sculpture, Wine, etc.)
6. `department` (Sotheby's auctions: Contemporary Art, etc.)
7. `vital_status` (alive/dead)
8. `location` (auction location: New York, London, etc.)
9. `currency`
10. `auction_house` (Sotheby's, Christie's, Phillips)

**Text features for embeddings:**
1. `description_text` (99%+ coverage across all houses)
2. `provenance` (31-51% coverage)
3. `lot_essay_excerpt` (Christie's/Phillips, 15-16% coverage)
4. `literature`, `exhibition`, `catalogue_note`

### Key Schema Differences to Harmonize

| Feature | Sotheby's | Christie's/Phillips |
|---------|-----------|-------------------|
| Artist name | `artist_name` | `creator_name` |
| Dimensions | Parsed `height_cm`, `width_cm`, `depth_cm` | Free text `dimensions_cm` |
| Creation date | Integer `creation_year` | Free text `date_created` |
| Price in USD | `hammer_price_usd` (normalized) | `hammer_price` + `currency` (needs FX) |
| Category | `object_type` (in raw_lots) | `lot_category` (in silver) |
| Gold price | `log_hammer_price` | `log_estimate_low`, `log_estimate_high` |
| Provenance | Free text string | Array of strings |

### Cross-House Data Summary

For a unified dataset joining all three houses:

| Completeness Level | Sotheby's | Christie's | Phillips | Total |
|-------------------|-----------|------------|----------|-------|
| Has any silver data | 88,620 | 388,368 | 81,876 | 558,864 |
| Has artist + medium | ~55,000 | ~265,000 | ~48,000 | ~368,000 |
| Has artist + medium + price | ~18,000 | ~264,000 | ~2,000 | ~284,000 |
| Has dimensions + price | ~12,000 | ~60,000* | ~2,000* | ~74,000 |

*Christie's/Phillips dimensions need parsing from free text

### Recommended Approach
1. Start with **Christie's as primary** (largest, best price coverage)
2. Parse Christie's `dimensions_cm` text to extract height/width
3. Normalize Christie's prices to USD using fx_rates
4. Add Sotheby's data (already has parsed dimensions and USD prices)
5. Phillips has minimal price data -- use mainly for artist/medium supplementation
6. Use `lot_category` / `object_type` to filter to fine art (painting, sculpture, drawing, photography) vs decorative/wine/jewelry
