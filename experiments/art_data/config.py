"""Configuration for art market feature extraction pipeline.

Primary data source: ClickHouse (Bloomsbury instance) with 3 auction houses.
Fallback: local SQLite (Sotheby's only, ~29K lots).

ClickHouse databases:
  - sothebys: ~120K lots, ~40K with hammer prices, ~120K with estimates
  - christies: ~691K lots, ~668K with hammer prices, ~686K with estimates
  - phillips: ~101K lots, ~3K with hammer prices, ~7K with estimates
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ClickHouse connection (primary data source)
# ---------------------------------------------------------------------------
CLICKHOUSE_HOST: str = os.getenv("ART_CH_HOST", "ch.bloomsburytech.com")
CLICKHOUSE_PORT: int = int(os.getenv("ART_CH_PORT", "443"))
CLICKHOUSE_USER: str = os.getenv("ART_CH_USER", "default")
CLICKHOUSE_PASSWORD: str = os.getenv("ART_CH_PASSWORD", "")

# Databases per auction house
AUCTION_HOUSE_DBS: dict[str, str] = {
    "sothebys": "sothebys",
    "christies": "christies",
    "phillips": "phillips",
}

# ---------------------------------------------------------------------------
# Fallback: local SQLite paths (Sotheby's only)
# ---------------------------------------------------------------------------
_DATA_ENRICH_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data_enrich" / "data"
SILVER_DB: Path = _DATA_ENRICH_DIR / "silver.db"
GOLD_DB: Path = _DATA_ENRICH_DIR / "gold.db"
SOTHEBYS_DB: Path = _DATA_ENRICH_DIR / "sothebys_lots.db"

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
MIN_HAMMER_USD: float = 100.0
MIN_ARTIST_LOTS: int = 2

# ---------------------------------------------------------------------------
# Medium categories -- group fine-grained media into macro categories
# ---------------------------------------------------------------------------
MEDIUM_CATEGORIES: dict[str, list[str]] = {
    "painting": ["oil", "acrylic", "gouache", "watercolour", "watercolor",
                  "tempera", "encaustic", "paint"],
    "works_on_paper": ["ink", "pencil", "charcoal", "pastel", "crayon",
                       "ink and colour", "mixed media", "collage"],
    "print": ["screenprint", "lithograph", "etching", "woodcut", "engraving",
              "woodblock print", "print", "offset lithograph",
              "screenprint in colours", "etching and aquatint", "printing",
              "printed text", "printed book"],
    "photograph": ["gelatin silver print", "chromogenic print", "gelatin silver",
                   "silver print", "photograph", "PNG"],
    "sculpture": ["bronze", "marble", "stainless steel", "steel", "iron",
                  "wood", "stone", "terracotta", "plaster"],
    "ceramic": ["porcelain", "ceramic", "stoneware", "pottery", "glaze",
                "celadon glaze", "terre de faïence", "underglaze blue",
                "famille-rose enamel", "enamel"],
    "decorative": ["silver", "gold", "gilt-bronze", "lacquer", "glass",
                   "textile", "leather", "jade", "diamonds", "yellow gold"],
    "wine_spirits": ["wine", "whisky", "spirit"],
}

# Reverse lookup: medium string -> category
MEDIUM_TO_CATEGORY: dict[str, str] = {}
for cat, media in MEDIUM_CATEGORIES.items():
    for m in media:
        MEDIUM_TO_CATEGORY[m.lower()] = cat

# ---------------------------------------------------------------------------
# Sale location mapping (from SAP sale number prefix -- Sotheby's specific)
# ---------------------------------------------------------------------------
SALE_LOCATION_PREFIXES: dict[str, str] = {
    "N1": "new_york", "N0": "new_york", "NV": "new_york",
    "NF": "new_york_online",
    "L2": "london", "L1": "london",
    "HK": "hong_kong",
    "PF": "paris",
    "MI": "milan",
    "DE": "dubai",
    "GE": "geneva", "G0": "geneva",
    "ZH": "zurich",
    "AM": "amsterdam",
}

# ---------------------------------------------------------------------------
# Train / validation / test split ratios (by sale date, temporal)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_EXPERIMENTS_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = _EXPERIMENTS_ROOT / "art_data" / "output"
FEATURES_FILE: Path = OUTPUT_DIR / "features.npz"
METADATA_FILE: Path = OUTPUT_DIR / "metadata.json"
