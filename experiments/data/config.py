"""Configuration for the Polymarket feature extraction pipeline.

Reads ClickHouse credentials from environment variables (matching pipeline/.env pattern)
and defines feature extraction parameters, dataset tiers, and output paths.

Aligned with the 27-feature spec in RESEARCH.md.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from pipeline directory (shared credentials)
_pipeline_env = Path(__file__).resolve().parent.parent.parent / "pipeline" / ".env"
load_dotenv(_pipeline_env)

# ---------------------------------------------------------------------------
# ClickHouse connection
# ---------------------------------------------------------------------------
CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "443"))
CLICKHOUSE_USER: str = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD: str = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE: str = os.getenv("CLICKHOUSE_DATABASE", "polymarket")

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
# Tier 2 (primary): binary resolved markets with >= MIN_TRADES trades
MIN_TRADES: int = 3

# ---------------------------------------------------------------------------
# Train / validation / test split ratios (by resolution date)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------------------------
# Output paths — align with config.yaml: results/data under experiments root
# ---------------------------------------------------------------------------
_EXPERIMENTS_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = _EXPERIMENTS_ROOT / "results" / "data"
FEATURES_FILE: Path = OUTPUT_DIR / "features.npz"
METADATA_FILE: Path = OUTPUT_DIR / "metadata.json"
