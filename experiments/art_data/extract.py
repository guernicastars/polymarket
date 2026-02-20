"""Main extraction script: queries ClickHouse, computes features, saves .npz.

Extracts 30 features from auction lot data across 3 houses (Sotheby's,
Christie's, Phillips) with estimates, bid data, and FX normalization.

Data sources (ClickHouse):
  - lots: lot_uuid, auction_uuid, lot_number, estimate_low, estimate_high
  - sales: hammer_price, final_price, num_bids, is_sold, currency
  - auctions: date, location, department, category, lot_count
  - silver_extractions: artist, medium, dimensions (parsed from HTML)
  - fx_rates: daily USD conversion rates (Sotheby's DB, shared)

Fallback: local SQLite (Sotheby's only, ~29K lots) if ClickHouse unreachable.

Usage:
    python -m art_data.extract                          # all 3 houses from ClickHouse
    python -m art_data.extract --source sqlite          # fallback to local SQLite
    python -m art_data.extract --min-price 500
    python -m art_data.extract --target over_estimate   # binary: hammer > estimate_mid
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd
from clickhouse_connect.driver.client import Client

from art_data.config import (
    AUCTION_HOUSE_DBS,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    GOLD_DB,
    MEDIUM_TO_CATEGORY,
    MIN_ARTIST_LOTS,
    MIN_HAMMER_USD,
    OUTPUT_DIR,
    SILVER_DB,
    SOTHEBYS_DB,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from art_data.features import build_feature_vector, extract_probe_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# ClickHouse connection
# ======================================================================

def get_ch_client() -> Client:
    """Create a ClickHouse client for the Bloomsbury instance."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        secure=True,
        compress="lz4",
        connect_timeout=30,
        send_receive_timeout=300,
    )


# ======================================================================
# ClickHouse data fetching
# ======================================================================

def fetch_fx_rates(client: Client) -> pd.DataFrame:
    """Fetch FX rates from Sotheby's database (shared rates table)."""
    query = """
    SELECT rate_date, currency, rate_to_usd
    FROM sothebys.fx_rates
    ORDER BY rate_date
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info("Fetched %d FX rates", len(df))
    return df


def _fx_convert_batch(
    amounts: pd.Series,
    currencies: pd.Series,
    sale_dates: pd.Series,
    fx_df: pd.DataFrame,
) -> pd.Series:
    """Vectorized FX conversion to USD using merge_asof per currency."""
    result = pd.Series(np.nan, index=amounts.index, dtype=float)

    # Direct USD pass-through
    cur_str = currencies.astype(str).str.upper().str.strip()
    is_usd = cur_str == "USD"
    valid_usd = is_usd & amounts.notna() & (amounts > 0)
    result[valid_usd] = amounts[valid_usd].astype(float)

    # Prepare FX reference
    fx = fx_df.copy()
    fx["currency"] = fx["currency"].str.upper()
    fx["rate_date"] = pd.to_datetime(fx["rate_date"])
    fx = fx.sort_values("rate_date")

    # Build fallback rates (latest per currency)
    latest_rates = fx.groupby("currency")["rate_to_usd"].last().to_dict()

    # Process non-USD: merge_asof per currency group
    needs_fx = (~is_usd) & amounts.notna() & (amounts > 0) & (cur_str != "") & (cur_str != "NAN")
    if not needs_fx.any():
        return result

    work = pd.DataFrame({
        "orig_idx": amounts.index[needs_fx],
        "amount": amounts[needs_fx].astype(float).values,
        "currency": cur_str[needs_fx].values,
        "sale_date": pd.to_datetime(sale_dates[needs_fx], errors="coerce").values,
    })

    for cur, grp in work.groupby("currency"):
        if cur not in latest_rates:
            continue
        cur_rates = fx[fx["currency"] == cur][["rate_date", "rate_to_usd"]].copy()
        if cur_rates.empty:
            continue

        sub = grp.copy()
        has_date = sub["sale_date"].notna()

        if has_date.any():
            dated = sub[has_date].sort_values("sale_date").copy()
            # Normalize datetime resolution to avoid merge_asof type mismatch
            # ClickHouse returns datetime64[us], FX rates are datetime64[s]
            dated["sale_date"] = dated["sale_date"].dt.floor("D").astype("datetime64[us]")
            rates_for_merge = cur_rates.rename(columns={"rate_date": "sale_date"}).copy()
            rates_for_merge["sale_date"] = rates_for_merge["sale_date"].dt.floor("D").astype("datetime64[us]")
            merged = pd.merge_asof(
                dated[["orig_idx", "amount", "sale_date"]],
                rates_for_merge,
                on="sale_date",
                direction="nearest",
            )
            for _, row in merged.iterrows():
                if pd.notna(row["rate_to_usd"]):
                    result[row["orig_idx"]] = row["amount"] * row["rate_to_usd"]

        # Fallback for rows without dates
        if (~has_date).any():
            rate = latest_rates[cur]
            for _, row in sub[~has_date].iterrows():
                result[row["orig_idx"]] = row["amount"] * rate

    return result


def _fx_to_usd(amount: float | None, currency: str | None, sale_date, fx_df: pd.DataFrame) -> float | None:
    """Convert an amount to USD using the closest FX rate by date."""
    if amount is None or pd.isna(amount) or float(amount) <= 0:
        return None
    if currency is None or not isinstance(currency, str) or not currency.strip():
        return None
    currency = currency.upper().strip()
    if currency == "USD":
        return amount

    rates = fx_df[fx_df["currency"] == currency]
    if rates.empty:
        return None

    if sale_date is not None and pd.notna(sale_date):
        try:
            target_date = pd.Timestamp(sale_date).date()
            rates_with_dist = rates.copy()
            rates_with_dist["dist"] = rates_with_dist["rate_date"].apply(
                lambda d: abs((pd.Timestamp(d).date() - target_date).days)
            )
            best = rates_with_dist.loc[rates_with_dist["dist"].idxmin()]
            return amount * float(best["rate_to_usd"])
        except Exception:
            pass

    # Fallback: use latest rate
    rate = float(rates.iloc[-1]["rate_to_usd"])
    return amount * rate


def _parse_dimensions_cm(dim_str: str | None) -> tuple[float | None, float | None, float | None]:
    """Parse dimension string into (h, w, d) in cm.

    Handles formats:
      - "70.1 x 59.9 cm" (standard H x W)
      - "35 w x 25 h x 18 d cm" (Christie's labeled format)
      - "70.1 x 59.9 x 30 cm" (H x W x D)
      - "16.5 cm" (single diameter/height)
    """
    if not dim_str or not isinstance(dim_str, str):
        return None, None, None

    s = dim_str.lower().strip().replace(',', '.')

    # Detect unit and set conversion factor
    if 'in' in s and 'cm' not in s:
        factor = 2.54  # inches to cm
    else:
        factor = 1.0

    # Christie's labeled format: "35 w x 25 h x 18 d cm"
    labeled = re.findall(r'([\d]+(?:\.[\d]+)?)\s*([whd])', s)
    if len(labeled) >= 2:
        dims = {label: float(val) * factor for val, label in labeled}
        return dims.get('h'), dims.get('w'), dims.get('d')

    # Standard format: extract all numbers
    nums = re.findall(r'([\d]+(?:\.[\d]+)?)', s)
    if not nums:
        return None, None, None

    vals = [float(v) * factor for v in nums]

    if len(vals) >= 3:
        return vals[0], vals[1], vals[2]
    elif len(vals) == 2:
        return vals[0], vals[1], None
    elif len(vals) == 1:
        return vals[0], None, None
    return None, None, None


def _parse_creation_year(date_str: str | None) -> tuple[int | None, int]:
    """Parse a date string like '2005' or '20TH CENTURY' into (year, is_approximate)."""
    if not date_str or not isinstance(date_str, str):
        return None, 0
    s = date_str.strip()
    # Direct year: "2005", "1890"
    if re.match(r'^\d{4}$', s):
        return int(s), 0
    # Circa/approximately: "c.1920", "circa 1920"
    m = re.search(r'(?:circa|c\.?)\s*(\d{4})', s, re.IGNORECASE)
    if m:
        return int(m.group(1)), 1
    # Century: "20TH CENTURY" -> midpoint
    m = re.search(r'(\d{1,2})(?:ST|ND|RD|TH)\s+CENTURY', s, re.IGNORECASE)
    if m:
        century = int(m.group(1))
        return (century - 1) * 100 + 50, 1
    # Year range: "1920-1925"
    m = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) // 2, 1
    # Any 4-digit year in string
    m = re.search(r'(\d{4})', s)
    if m:
        y = int(m.group(1))
        if 1000 <= y <= 2030:
            return y, 1
    return None, 0


def fetch_lots_clickhouse(client: Client, min_price: float) -> pd.DataFrame:
    """Fetch lots from all 3 auction houses via ClickHouse.

    Joins lots + sales + auctions + silver_extractions per house,
    then unions the results with FX normalization.
    """
    fx_df = fetch_fx_rates(client)
    all_dfs = []

    for house_name, db_name in AUCTION_HOUSE_DBS.items():
        logger.info("Fetching %s from ClickHouse...", house_name)

        try:
            if house_name == "sothebys":
                # Sotheby's: artist_name, height/width/depth as separate columns, creation_year
                query = f"""
                SELECT
                    l.lot_uuid AS lot_uuid, l.auction_uuid AS auction_uuid,
                    l.lot_number AS lot_number,
                    l.estimate_low AS estimate_low, l.estimate_high AS estimate_high,
                    s.hammer_price AS hammer_price, s.final_price AS final_price,
                    s.num_bids AS num_bids, s.is_sold AS is_sold,
                    s.currency AS sale_currency,
                    a.title AS auction_title, a.department AS department,
                    a.category AS auction_category, a.location AS location,
                    a.date_starts_closing AS sale_date, a.lot_count AS lot_count,
                    a.sap_sale_number AS sale_number,
                    se.artist_name AS artist_name,
                    se.artist_birth_year AS artist_birth_year,
                    se.artist_death_year AS artist_death_year,
                    se.artist_nationality AS artist_nationality,
                    se.medium AS medium,
                    se.height_cm AS height_cm, se.width_cm AS width_cm,
                    se.depth_cm AS depth_cm,
                    se.creation_year AS creation_year,
                    se.creation_is_approximate AS creation_is_approximate,
                    '' AS dimensions_cm,
                    '' AS date_created
                FROM {db_name}.lots l
                INNER JOIN {db_name}.sales s ON l.lot_uuid = s.lot_uuid
                LEFT JOIN {db_name}.auctions a ON l.auction_uuid = a.auction_uuid
                LEFT JOIN {db_name}.silver_extractions se ON l.lot_uuid = se.lot_uuid
                WHERE s.hammer_price > 0 AND s.is_sold = 1
                """
            else:
                # Christie's / Phillips: creator_name, dimensions_cm as string,
                # date_created as string, no reliable sale dates in auctions
                query = f"""
                SELECT
                    l.lot_uuid AS lot_uuid, l.auction_uuid AS auction_uuid,
                    l.lot_number AS lot_number,
                    l.estimate_low AS estimate_low, l.estimate_high AS estimate_high,
                    s.hammer_price AS hammer_price, s.final_price AS final_price,
                    s.num_bids AS num_bids, s.is_sold AS is_sold,
                    s.currency AS sale_currency,
                    a.title AS auction_title, a.department AS department,
                    a.category AS auction_category, a.location AS location,
                    l.created_at AS sale_date, a.lot_count AS lot_count,
                    a.sale_number AS sale_number,
                    se.creator_name AS artist_name,
                    se.creator_birth_year AS artist_birth_year,
                    se.creator_death_year AS artist_death_year,
                    se.creator_nationality AS artist_nationality,
                    se.medium AS medium,
                    0 AS height_cm, 0 AS width_cm, 0 AS depth_cm,
                    0 AS creation_year, 0 AS creation_is_approximate,
                    se.dimensions_cm AS dimensions_cm,
                    se.date_created AS date_created
                FROM {db_name}.lots l
                INNER JOIN {db_name}.sales s ON l.lot_uuid = s.lot_uuid
                LEFT JOIN {db_name}.auctions a ON l.auction_uuid = a.auction_uuid
                LEFT JOIN {db_name}.silver_extractions se ON l.lot_uuid = se.lot_uuid
                WHERE s.hammer_price > 0 AND s.is_sold = 1
                """

            result = client.query(query)
            df = pd.DataFrame(result.result_rows, columns=result.column_names)

            if df.empty:
                logger.info("  %s: 0 lots", house_name)
                continue

            df["auction_house"] = house_name

            # For Christie's/Phillips: parse dimension strings into h/w/d
            if house_name != "sothebys" and "dimensions_cm" in df.columns:
                parsed = df["dimensions_cm"].apply(_parse_dimensions_cm)
                df["height_cm"] = parsed.apply(lambda x: x[0])
                df["width_cm"] = parsed.apply(lambda x: x[1])
                df["depth_cm"] = parsed.apply(lambda x: x[2])

            # For Christie's/Phillips: parse date_created string into creation_year
            if house_name != "sothebys" and "date_created" in df.columns:
                parsed_dates = df["date_created"].apply(_parse_creation_year)
                df["creation_year"] = parsed_dates.apply(lambda x: x[0])
                df["creation_is_approximate"] = parsed_dates.apply(lambda x: x[1])

            # Replace 0/NaN placeholder values with None for clean NaN handling
            for col in ("height_cm", "width_cm", "depth_cm", "creation_year"):
                if col in df.columns:
                    df[col] = df[col].replace(0, None)

            # FX normalize hammer_price and estimates to USD (vectorized)
            df["hammer_price_usd"] = _fx_convert_batch(
                df["hammer_price"], df["sale_currency"], df["sale_date"], fx_df
            )
            df["estimate_low_usd"] = _fx_convert_batch(
                df["estimate_low"], df["sale_currency"], df["sale_date"], fx_df
            )
            df["estimate_high_usd"] = _fx_convert_batch(
                df["estimate_high"], df["sale_currency"], df["sale_date"], fx_df
            )

            # Filter by min USD price
            df = df[df["hammer_price_usd"].notna() & (df["hammer_price_usd"] >= min_price)]

            logger.info("  %s: %d lots after FX + min_price filter", house_name, len(df))
            all_dfs.append(df)

        except Exception as e:
            logger.warning("Failed to process %s: %s", house_name, e)
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Infer vital status from birth/death years
    def _vital_status(row):
        dy = row.get("artist_death_year")
        by = row.get("artist_birth_year")
        if dy and pd.notna(dy) and int(dy) > 0:
            return "dead"
        if by and pd.notna(by) and int(by) > 0:
            return "alive"
        return ""

    combined["vital_status"] = combined.apply(_vital_status, axis=1)
    combined["is_rare_artist"] = 0  # Will be computed during historical features

    # Sort by sale date
    combined = combined.sort_values("sale_date").reset_index(drop=True)

    logger.info("Total combined: %d lots across %d houses", len(combined), len(all_dfs))
    return combined


# ======================================================================
# SQLite fallback (Sotheby's only)
# ======================================================================

def fetch_lots_sqlite(min_price: float) -> pd.DataFrame:
    """Fetch lots from local SQLite (Sotheby's only). Fallback path."""
    conn = sqlite3.connect(str(SILVER_DB))
    conn.row_factory = sqlite3.Row

    conn.execute(f"ATTACH DATABASE '{GOLD_DB}' AS gold")
    conn.execute(f"ATTACH DATABASE '{SOTHEBYS_DB}' AS sothebys")

    query = """
    SELECT
        s.lot_uuid, s.auction_uuid,
        s.artist_name, s.artist_nationality,
        s.artist_birth_year, s.artist_death_year,
        s.height_cm, s.width_cm, s.depth_cm,
        s.medium, s.support,
        s.creation_year, s.creation_is_approximate,
        s.hammer_price_usd,
        g.is_rare_artist, g.vital_status,
        a.sap_sale_number AS sale_number,
        a.date_starts_closing AS sale_date,
        a.currency, a.lot_count
    FROM silver_extractions s
    LEFT JOIN gold.gold_features g ON s.lot_uuid = g.lot_uuid
    LEFT JOIN sothebys.auctions a ON s.auction_uuid = a.auction_uuid
    WHERE s.hammer_price_usd IS NOT NULL AND s.hammer_price_usd >= ?
    ORDER BY a.date_starts_closing, s.lot_uuid
    """
    rows = conn.execute(query, (min_price,)).fetchall()
    conn.close()

    data = [dict(row) for row in rows]
    df = pd.DataFrame(data)
    df["auction_house"] = "sothebys"
    df["estimate_low_usd"] = None
    df["estimate_high_usd"] = None
    df["num_bids"] = None
    df["lot_number"] = None
    df["department"] = None
    df["auction_category"] = None
    df["location"] = None

    logger.info("SQLite fallback: %d lots", len(df))
    return df


# ======================================================================
# Historical feature computation (avoids future leakage)
# ======================================================================

def compute_historical_features(lots_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-artist running historical features.

    Uses ONLY lots from the same artist with an earlier sale_date
    (strict temporal ordering to avoid leakage).
    """
    df = lots_df.copy()
    df["_sale_dt"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.sort_values("_sale_dt").reset_index(drop=True)
    df["_log_price"] = np.log(df["hammer_price_usd"].astype(float))

    # Artist lot counts (total, for market depth feature)
    artist_counts = df.groupby("artist_name").size().to_dict()
    df["artist_lot_count"] = df["artist_name"].map(artist_counts).fillna(0).astype(int)

    # Mark rare artists (< MIN_ARTIST_LOTS total appearances)
    df["is_rare_artist"] = (df["artist_lot_count"] < MIN_ARTIST_LOTS).astype(int)

    # Running historical stats per artist
    hist_avg = np.full(len(df), np.nan)
    hist_median = np.full(len(df), np.nan)
    hist_std = np.full(len(df), np.nan)
    hist_prior = np.full(len(df), np.nan)
    hist_trend = np.full(len(df), np.nan)

    artist_history: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for idx, row in df.iterrows():
        artist = row["artist_name"]
        if not artist or pd.isna(artist):
            continue

        history = artist_history[artist]
        n_prior = len(history)

        if n_prior >= MIN_ARTIST_LOTS:
            prices = [p for _, p in history]
            hist_avg[idx] = float(np.mean(prices))
            hist_median[idx] = float(np.median(prices))
            hist_std[idx] = float(np.std(prices)) if n_prior >= 3 else np.nan
            hist_prior[idx] = float(np.log1p(n_prior))

            if n_prior >= 3:
                times = np.array([t for t, _ in history])
                prices_arr = np.array(prices)
                t_mean = times.mean()
                t_std = times.std()
                if t_std > 0:
                    t_norm = (times - t_mean) / t_std
                    slope = float(np.polyfit(t_norm, prices_arr, 1)[0])
                    hist_trend[idx] = slope
        elif n_prior > 0:
            hist_prior[idx] = float(np.log1p(n_prior))

        sale_dt = row["_sale_dt"]
        t_ord = float(sale_dt.toordinal()) if pd.notna(sale_dt) else 0.0
        history.append((t_ord, float(row["_log_price"])))

    df["artist_avg_log_price"] = hist_avg
    df["artist_median_log_price"] = hist_median
    df["artist_price_std"] = hist_std
    df["artist_prior_lots"] = hist_prior
    df["artist_price_trend"] = hist_trend

    df.drop(columns=["_sale_dt", "_log_price"], inplace=True)
    return df


# ======================================================================
# Lot position computation
# ======================================================================

def compute_lot_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign lot position within each auction.

    Uses lot_number if available (Christie's/Phillips), otherwise row order.
    Groups by auction_uuid where available; assigns sequential positions otherwise.
    """
    df = df.copy()
    df["lot_position"] = np.nan

    group_col = "auction_uuid" if "auction_uuid" in df.columns else None
    has_lot_number = "lot_number" in df.columns

    if group_col is not None:
        # Handle rows with valid auction_uuid
        has_auction = df[group_col].notna() & (df[group_col].astype(str) != "")
        if has_auction.any():
            for auction_id, group in df[has_auction].groupby(group_col):
                if has_lot_number:
                    df.loc[group.index, "lot_position"] = group["lot_number"]
                    missing = df.loc[group.index, "lot_position"].isna()
                    if missing.any():
                        df.loc[group.index[missing], "lot_position"] = range(1, missing.sum() + 1)
                else:
                    df.loc[group.index, "lot_position"] = range(1, len(group) + 1)

        # Rows without auction_uuid: assign sequential positions
        no_auction = ~has_auction
        if no_auction.any():
            df.loc[no_auction, "lot_position"] = range(1, no_auction.sum() + 1)
    else:
        df["lot_position"] = range(1, len(df) + 1)

    return df


# ======================================================================
# Target encoding
# ======================================================================

def encode_target(df: pd.DataFrame, target: str = "log_price") -> np.ndarray:
    """Encode the prediction target."""
    if target == "log_price":
        return np.log(df["hammer_price_usd"].astype(float).values)

    elif target == "price_bucket":
        prices = df["hammer_price_usd"].astype(float).values
        quartiles = np.percentile(prices, [25, 50, 75])
        return np.digitize(prices, quartiles).astype(float)

    elif target == "over_estimate":
        # Binary: 1 if hammer > estimate midpoint, 0 otherwise
        hammer = df["hammer_price_usd"].astype(float).values
        est_low = pd.to_numeric(df["estimate_low_usd"], errors="coerce").values
        est_high = pd.to_numeric(df["estimate_high_usd"], errors="coerce").values
        est_mid = (est_low + est_high) / 2.0

        # Where estimates are available, use them; otherwise use median price
        has_est = np.isfinite(est_mid) & (est_mid > 0)
        y = np.full(len(hammer), np.nan)
        y[has_est] = (hammer[has_est] > est_mid[has_est]).astype(float)

        # For lots without estimates, fall back to median
        if (~has_est).any():
            median_price = np.median(hammer[has_est]) if has_est.any() else np.median(hammer)
            y[~has_est] = (hammer[~has_est] > median_price).astype(float)

        return y

    raise ValueError(f"Unknown target: {target}")


def assign_price_buckets(prices: np.ndarray) -> list[str]:
    """Assign price quartile labels for probe evaluation."""
    quartiles = np.percentile(prices, [25, 50, 75])
    labels = []
    for p in prices:
        if p <= quartiles[0]:
            labels.append("Q1_low")
        elif p <= quartiles[1]:
            labels.append("Q2_mid_low")
        elif p <= quartiles[2]:
            labels.append("Q3_mid_high")
        else:
            labels.append("Q4_high")
    return labels


# ======================================================================
# Temporal split
# ======================================================================

def temporal_split(
    df: pd.DataFrame, X: np.ndarray, y: np.ndarray, lot_ids: list[str],
) -> dict[str, np.ndarray]:
    """Split data temporally by sale date."""
    dates = pd.to_datetime(df["sale_date"], errors="coerce").values
    sort_idx = np.argsort(dates)

    n = len(sort_idx)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_idx = sort_idx[:n_train]
    val_idx = sort_idx[n_train:n_val]
    test_idx = sort_idx[n_val:]

    ids_arr = np.array(lot_ids)

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "lot_ids_train": ids_arr[train_idx],
        "lot_ids_val": ids_arr[val_idx],
        "lot_ids_test": ids_arr[test_idx],
    }


# ======================================================================
# Main pipeline
# ======================================================================

def main(
    min_price: float = MIN_HAMMER_USD,
    target: str = "log_price",
    source: str = "clickhouse",
    output_dir: Path | None = None,
) -> None:
    """Run the full extraction pipeline: query -> features -> split -> save."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1. Fetch raw lots
    if source == "clickhouse":
        logger.info("Connecting to ClickHouse at %s:%s", CLICKHOUSE_HOST, CLICKHOUSE_PORT)
        try:
            client = get_ch_client()
            lots_df = fetch_lots_clickhouse(client, min_price)
        except Exception as e:
            logger.warning("ClickHouse connection failed: %s. Falling back to SQLite.", e)
            lots_df = fetch_lots_sqlite(min_price)
    else:
        lots_df = fetch_lots_sqlite(min_price)

    if lots_df.empty:
        logger.warning("No lots found. Exiting.")
        return

    # 2. Compute lot positions
    logger.info("Computing lot positions...")
    lots_df = compute_lot_positions(lots_df)

    # 3. Compute historical artist features (temporal, no leakage)
    logger.info("Computing historical artist features (temporal, %d lots)...", len(lots_df))
    lots_df = compute_historical_features(lots_df)

    # 4. Assign price buckets for probe labels
    prices = lots_df["hammer_price_usd"].astype(float).values
    lots_df["price_bucket"] = assign_price_buckets(prices)

    # 5. Extract features and probe labels
    logger.info("Extracting features for %d lots...", len(lots_df))
    all_features: list[dict[str, float]] = []
    all_probe_labels: dict[str, list] = defaultdict(list)
    valid_lot_ids: list[str] = []
    valid_indices: list[int] = []

    for i, (idx, row) in enumerate(lots_df.iterrows()):
        if (i + 1) % 50000 == 0 or i == 0:
            logger.info("Processing %d/%d lots", i + 1, len(lots_df))

        lot = row.to_dict()
        fv = build_feature_vector(lot)
        all_features.append(fv)

        probe_labels = extract_probe_labels(lot)
        for key, val in probe_labels.items():
            all_probe_labels[key].append(val)

        valid_lot_ids.append(str(lot["lot_uuid"]))
        valid_indices.append(idx)

    if not all_features:
        logger.warning("No lots produced valid features. Exiting.")
        return

    # 6. Build aligned feature matrix
    all_keys: set[str] = set()
    for fv in all_features:
        all_keys.update(fv.keys())
    feature_names = sorted(all_keys)

    X = np.array(
        [[fv.get(fn, np.nan) for fn in feature_names] for fv in all_features],
        dtype=np.float64,
    )

    # 7. Encode target
    valid_df = lots_df.iloc[valid_indices].reset_index(drop=True)
    y = encode_target(valid_df, target)

    logger.info("Feature matrix: %s, target vector: %s", X.shape, y.shape)
    logger.info("Features: %d, NaN fraction: %.2f%%", len(feature_names), 100 * np.isnan(X).mean())

    # Per-feature NaN report
    for fi, fn in enumerate(feature_names):
        nan_frac = np.isnan(X[:, fi]).mean()
        if nan_frac > 0.01:
            logger.info("  %s: %.1f%% NaN", fn, 100 * nan_frac)

    # Target stats
    if target == "log_price":
        logger.info(
            "Target (log_price): mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
            y.mean(), y.std(), y.min(), y.max(),
        )
    else:
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info("  Class %.0f: %d (%.1f%%)", u, c, 100 * c / len(y))

    # Per-house breakdown
    if "auction_house" in valid_df.columns:
        for house, group in valid_df.groupby("auction_house"):
            logger.info("  %s: %d lots", house, len(group))

    # 8. Temporal split
    splits = temporal_split(valid_df, X, y, valid_lot_ids)

    # 9. Build probe label arrays
    probe_arrays = {}
    for key, vals in all_probe_labels.items():
        probe_arrays[key] = np.array(vals, dtype=object)

    # 10. Save .npz
    features_path = out / "features.npz"
    np.savez_compressed(features_path, feature_names=np.array(feature_names), **splits, **probe_arrays)
    logger.info("Saved features to %s", features_path)

    # 11. Compute multicollinearity preview
    high_corr_pairs = []
    max_corr = 0.0
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() < 50:
                continue
            xi, xj = X[mask, i], X[mask, j]
            if xi.std() < 1e-10 or xj.std() < 1e-10:
                continue
            r = float(np.corrcoef(xi, xj)[0, 1])
            if abs(r) > max_corr:
                max_corr = abs(r)
            if abs(r) > 0.5:
                high_corr_pairs.append((feature_names[i], feature_names[j], round(r, 3)))
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # 12. Save metadata
    # Count per-house lots
    house_counts = {}
    if "auction_house" in valid_df.columns:
        house_counts = valid_df["auction_house"].value_counts().to_dict()

    metadata = {
        "n_lots": len(valid_lot_ids),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_categories": {
            "A_artist": [
                "is_living", "artist_birth_year", "artist_career_length",
                "is_rare_artist", "artist_market_depth", "nationality_known",
            ],
            "B_physical": [
                "height_cm", "width_cm", "log_surface_area",
                "aspect_ratio", "has_depth", "creation_year",
                "creation_is_approximate",
            ],
            "C_medium": [
                "is_painting", "is_sculpture", "is_work_on_paper",
                "is_decorative", "medium_known",
            ],
            "D_estimate": [
                "log_estimate_low", "log_estimate_high",
                "log_estimate_mid", "estimate_spread",
            ],
            "E_sale_context": [
                "sale_month", "sale_year_numeric", "sale_day_of_week",
                "sale_size", "lot_position_pct",
            ],
            "F_historical": [
                "artist_avg_log_price", "artist_median_log_price",
                "artist_price_std", "artist_prior_lots", "artist_price_trend",
            ],
            "G_derived": [
                "log_depth_cm", "age_at_creation", "years_since_creation",
            ],
        },
        "target": target,
        "split_sizes": {
            "train": int(splits["X_train"].shape[0]),
            "val": int(splits["X_val"].shape[0]),
            "test": int(splits["X_test"].shape[0]),
        },
        "nan_fraction": float(np.isnan(X).mean()),
        "multicollinearity_preview": {
            "max_abs_correlation": max_corr,
            "high_correlation_pairs_gt_0.5": [
                {"feat_a": a, "feat_b": b, "r": r}
                for a, b, r in high_corr_pairs[:20]
            ],
        },
        "probe_labels": {
            key: {
                "n_valid": int(sum(1 for v in vals if v is not None and str(v) != "")),
                "n_classes": len(set(v for v in vals if v is not None and str(v) != "")),
            }
            for key, vals in all_probe_labels.items()
        },
        "data_source": {
            "type": source,
            "auction_houses": house_counts,
            "min_hammer_usd": min_price,
        },
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = out / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    logger.info(
        "Extraction complete: %d lots, %d features, split %d/%d/%d",
        len(valid_lot_ids), len(feature_names),
        splits["X_train"].shape[0], splits["X_val"].shape[0], splits["X_test"].shape[0],
    )
    if high_corr_pairs:
        logger.info("Top correlated pairs (|r| > 0.5):")
        for a, b, r in high_corr_pairs[:10]:
            logger.info("  %s <-> %s: r=%.3f", a, b, r)


# ======================================================================
# CLI entry point
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from auction lot data (Sotheby's + Christie's + Phillips)",
    )
    parser.add_argument(
        "--min-price", type=float, default=MIN_HAMMER_USD,
        help=f"Minimum hammer price USD (default: {MIN_HAMMER_USD})",
    )
    parser.add_argument(
        "--target", choices=["log_price", "over_estimate", "price_bucket"],
        default="log_price",
        help="Target variable (default: log_price)",
    )
    parser.add_argument(
        "--source", choices=["clickhouse", "sqlite"],
        default="clickhouse",
        help="Data source (default: clickhouse)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        min_price=args.min_price,
        target=args.target,
        source=args.source,
        output_dir=args.output_dir,
    )
