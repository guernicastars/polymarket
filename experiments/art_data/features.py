"""Feature engineering for art market auction lots.

35 features across 7 categories designed to exhibit natural multicollinearity
-- the ideal test case for Orth-SVAE vs PCA.

Feature categories:
  A. Artist Features (6): is_living, birth_year, career_length, rare, market_depth, nationality
  B. Artwork Physical (7): height, width, log_surface_area, aspect_ratio, has_depth, creation_year, approx
  C. Medium & Material (5): is_painting, is_sculpture, is_work_on_paper, is_decorative, medium_known
  D. Estimate / Valuation (4): log_estimate_low, log_estimate_high, log_estimate_mid, estimate_spread
  E. Sale Context (5): sale_month, sale_year_numeric, day_of_week, sale_size, lot_position_pct
  F. Historical Performance (5): artist_avg/median_log_price, price_std, prior_lots, price_trend
  G. Derived Ratios (3): log_depth_cm, age_at_creation, years_since_creation

Expected multicollinearity groups:
  - Dimension cluster: height_cm, width_cm, log_surface_area (r~0.8+)
  - Estimate cluster: log_estimate_low, log_estimate_high, log_estimate_mid (r~0.97+)
  - Medium-physical: is_sculpture <-> has_depth, log_depth_cm (r~0.7+)
  - Historical: artist_avg_log_price <-> artist_median_log_price (r~0.95+)
  - Age: creation_year, artist_birth_year, age_at_creation (near-linear)
  - Time: years_since_creation <-> creation_year (r = -1.0, exact)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from art_data.config import MEDIUM_CATEGORIES, MEDIUM_TO_CATEGORY, SALE_LOCATION_PREFIXES


# ======================================================================
# Category A: Artist Features (6)
# ======================================================================

def extract_artist_features(lot: dict) -> dict[str, float]:
    """Features 1-6: artist-level characteristics."""
    features: dict[str, float] = {}

    vital = lot.get("vital_status", "")
    if not isinstance(vital, str):
        vital = ""
    if vital == "alive":
        features["is_living"] = 1.0
    elif vital == "dead":
        features["is_living"] = 0.0
    else:
        features["is_living"] = np.nan

    birth = lot.get("artist_birth_year")
    features["artist_birth_year"] = (
        float(birth) if birth and not pd.isna(birth) and float(birth) > 0 else np.nan
    )

    birth_y = lot.get("artist_birth_year")
    death_y = lot.get("artist_death_year")
    if birth_y and not pd.isna(birth_y) and float(birth_y) > 0:
        end_y = float(death_y) if (death_y and not pd.isna(death_y) and float(death_y) > 0) else 2025
        career = end_y - float(birth_y)
        features["artist_career_length"] = float(career) if career > 0 else np.nan
    else:
        features["artist_career_length"] = np.nan

    features["is_rare_artist"] = float(lot.get("is_rare_artist", 0))

    prior = lot.get("artist_lot_count", 0)
    features["artist_market_depth"] = float(np.log1p(prior)) if prior and not pd.isna(prior) else 0.0

    nat = lot.get("artist_nationality", "")
    features["nationality_known"] = 1.0 if (nat and isinstance(nat, str) and nat.strip()) else 0.0

    return features


# ======================================================================
# Category B: Artwork Physical (7)
# ======================================================================

def extract_physical_features(lot: dict) -> dict[str, float]:
    """Features 7-13: physical characteristics."""
    features: dict[str, float] = {}

    h = lot.get("height_cm")
    w = lot.get("width_cm")
    d = lot.get("depth_cm")

    def _pos(v):
        return v is not None and not pd.isna(v) and float(v) > 0

    features["height_cm"] = float(h) if _pos(h) else np.nan
    features["width_cm"] = float(w) if _pos(w) else np.nan

    if _pos(h) and _pos(w):
        features["log_surface_area"] = float(np.log(float(h) * float(w)))
        features["aspect_ratio"] = float(float(w) / float(h))
    else:
        features["log_surface_area"] = np.nan
        features["aspect_ratio"] = np.nan

    features["has_depth"] = 1.0 if _pos(d) else 0.0

    year = lot.get("creation_year")
    features["creation_year"] = float(year) if _pos(year) else np.nan

    features["creation_is_approximate"] = float(lot.get("creation_is_approximate", 0))

    return features


# ======================================================================
# Category C: Medium & Material (5)
# ======================================================================

def extract_medium_features(lot: dict) -> dict[str, float]:
    """Features 14-18: medium and material encoding."""
    features: dict[str, float] = {}

    medium = lot.get("medium", "")
    if not isinstance(medium, str):
        medium = ""
    cat = MEDIUM_TO_CATEGORY.get(medium.lower().strip(), "") if medium.strip() else ""

    features["is_painting"] = 1.0 if cat == "painting" else 0.0
    features["is_sculpture"] = 1.0 if cat == "sculpture" else 0.0
    features["is_work_on_paper"] = 1.0 if cat in ("works_on_paper", "print", "photograph") else 0.0
    features["is_decorative"] = 1.0 if cat in ("decorative", "ceramic") else 0.0
    features["medium_known"] = 1.0 if cat else 0.0

    return features


# ======================================================================
# Category D: Estimate / Valuation (4)
# Creates a near-perfect collinearity cluster: low, high, mid (r~0.97+)
# ======================================================================

def extract_estimate_features(lot: dict) -> dict[str, float]:
    """Features 19-22: pre-sale estimate signals."""
    features: dict[str, float] = {}

    def _pos_val(v):
        return v is not None and not pd.isna(v) and float(v) > 0

    est_low = lot.get("estimate_low_usd")
    est_high = lot.get("estimate_high_usd")

    features["log_estimate_low"] = float(np.log1p(float(est_low))) if _pos_val(est_low) else np.nan
    features["log_estimate_high"] = float(np.log1p(float(est_high))) if _pos_val(est_high) else np.nan

    if _pos_val(est_low) and _pos_val(est_high):
        mid = (float(est_low) + float(est_high)) / 2.0
        features["log_estimate_mid"] = float(np.log1p(mid))
        features["estimate_spread"] = float((float(est_high) - float(est_low)) / float(est_low))
    else:
        features["log_estimate_mid"] = np.nan
        features["estimate_spread"] = np.nan

    return features


# ======================================================================
# Category E: Sale Context (5)
# ======================================================================

def extract_sale_features(lot: dict) -> dict[str, float]:
    """Features 23-27: contextual sale information."""
    from datetime import datetime as dt

    features: dict[str, float] = {}

    date_val = lot.get("sale_date", "")
    parsed = False
    if date_val is not None:
        try:
            if isinstance(date_val, str) and date_val.strip():
                d = dt.fromisoformat(str(date_val).replace("Z", "+00:00"))
                parsed = True
            elif hasattr(date_val, 'month'):
                d = date_val
                parsed = True
        except (ValueError, TypeError):
            pass

    if parsed:
        features["sale_month"] = float(d.month)
        features["sale_year_numeric"] = float(d.year)
        features["sale_day_of_week"] = float(d.weekday())
    else:
        features["sale_month"] = np.nan
        features["sale_year_numeric"] = np.nan
        features["sale_day_of_week"] = np.nan

    lc = lot.get("lot_count", 0)
    features["sale_size"] = float(lc) if lc and not pd.isna(lc) else 0.0

    pos = lot.get("lot_position")
    total = lot.get("lot_count", 1)
    if pos is not None and not pd.isna(pos) and total and not pd.isna(total) and float(total) > 0:
        features["lot_position_pct"] = float(float(pos) / float(total))
    else:
        features["lot_position_pct"] = np.nan

    return features


# ======================================================================
# Category F: Historical Performance (5)
# ======================================================================

def extract_historical_features(lot: dict) -> dict[str, float]:
    """Features 28-32: artist historical performance (pre-computed)."""
    features: dict[str, float] = {}
    features["artist_avg_log_price"] = float(lot.get("artist_avg_log_price", np.nan))
    features["artist_median_log_price"] = float(lot.get("artist_median_log_price", np.nan))
    features["artist_price_std"] = float(lot.get("artist_price_std", np.nan))
    features["artist_prior_lots"] = float(lot.get("artist_prior_lots", np.nan))
    features["artist_price_trend"] = float(lot.get("artist_price_trend", np.nan))
    return features


# ======================================================================
# Category G: Derived Ratios (3)
# ======================================================================

def extract_derived_features(lot: dict) -> dict[str, float]:
    """Features 33-35: derived ratios introducing collinearity."""
    features: dict[str, float] = {}

    d = lot.get("depth_cm")
    features["log_depth_cm"] = float(np.log1p(float(d))) if (d is not None and not pd.isna(d) and float(d) > 0) else 0.0

    birth = lot.get("artist_birth_year")
    creation = lot.get("creation_year")
    if (birth and not pd.isna(birth) and float(birth) > 0
            and creation and not pd.isna(creation) and float(creation) > 0):
        age = float(creation) - float(birth)
        features["age_at_creation"] = float(age) if age >= 0 else np.nan
    else:
        features["age_at_creation"] = np.nan

    if creation and not pd.isna(creation) and float(creation) > 0:
        features["years_since_creation"] = float(2025 - float(creation))
    else:
        features["years_since_creation"] = np.nan

    return features


# ======================================================================
# Aggregation
# ======================================================================

def build_feature_vector(lot: dict) -> dict[str, float]:
    """Build a single flat feature dict for one auction lot (35 features)."""
    features: dict[str, float] = {}
    features.update(extract_artist_features(lot))
    features.update(extract_physical_features(lot))
    features.update(extract_medium_features(lot))
    features.update(extract_estimate_features(lot))
    features.update(extract_sale_features(lot))
    features.update(extract_historical_features(lot))
    features.update(extract_derived_features(lot))
    return features


# ======================================================================
# Probe labels
# ======================================================================

def extract_probe_labels(lot: dict) -> dict[str, str | None]:
    """Extract categorical labels for downstream linear probe evaluation."""
    labels: dict[str, str | None] = {}

    # Medium category
    medium = lot.get("medium", "")
    if medium and isinstance(medium, str) and medium.strip():
        labels["medium"] = MEDIUM_TO_CATEGORY.get(medium.lower().strip())
    else:
        labels["medium"] = None

    # Auction house (now multi-house)
    house = lot.get("auction_house", "")
    labels["auction_house"] = str(house) if house and isinstance(house, str) else None

    # Artist vital status
    vital = lot.get("vital_status", "")
    if not isinstance(vital, str):
        vital = ""
    labels["artist_vital_status"] = vital if vital in ("alive", "dead") else None

    # Sale location (from auction location field or SAP prefix)
    location = lot.get("location", "")
    if location and isinstance(location, str) and location.strip():
        labels["sale_category"] = location.strip().lower()
    else:
        sap = lot.get("sale_number", "")
        if not isinstance(sap, str):
            sap = ""
        if sap and len(sap) >= 2:
            labels["sale_category"] = SALE_LOCATION_PREFIXES.get(sap[:2], "other")
        else:
            labels["sale_category"] = None

    # Price bucket (assigned during extraction)
    labels["price_bucket"] = lot.get("price_bucket")

    return labels
