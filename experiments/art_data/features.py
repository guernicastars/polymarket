"""Feature engineering for art market auction lots.

65 features across 15 categories designed to exhibit natural multicollinearity
-- the ideal test case for Orth-SVAE vs PCA.

Feature categories (original 35):
  A. Artist Features (6): is_living, birth_year, career_length, rare, market_depth, nationality
  B. Artwork Physical (7): height, width, log_surface_area, aspect_ratio, has_depth, creation_year, approx
  C. Medium & Material (5): is_painting, is_sculpture, is_work_on_paper, is_decorative, medium_known
  D. Estimate / Valuation (4): log_estimate_low, log_estimate_high, log_estimate_mid, estimate_spread
  E. Sale Context (5): sale_month, sale_year_numeric, day_of_week, sale_size, lot_position_pct
  F. Historical Performance (5): artist_avg/median_log_price, price_std, prior_lots, price_trend
  G. Derived Ratios (3): log_depth_cm, age_at_creation, years_since_creation

New categories (30 new features):
  H. Extraction Confidence (4): artist_name_conf, dimensions_conf, medium_conf, creation_conf
  I. Sale Mechanics (5): num_bids, log_starting_bid, is_online_sale, reserve_met, hammer_start_ratio
  J. Creator Attribution (2): is_attributed_artist, is_maker_not_artist
  K. Provenance & Literature (3): provenance_count, literature_count, exhibition_count
  L. Text Signals (4): title_length, has_description, has_signed_inscribed, has_condition_report
  M. Style & Period (2): has_style_period, has_origin
  N. Lot Classification (4): is_wine, is_jewelry, is_book, is_asian_art
  O. Sale & Tax Flags (3): has_crypto, is_guaranteed, final_hammer_ratio
  P. Estimate Accuracy (3): estimate_mid_usd, log_estimate_range_usd, estimate_relative_level

Expected NEW multicollinearity groups:
  - Confidence cluster: artist_name_conf, dims_conf, medium_conf, creation_conf (r~0.3-0.6)
  - Sale mechanics: num_bids <-> hammer_start_ratio <-> log_starting_bid (r~0.5+)
  - Provenance cluster: provenance_count, literature_count, exhibition_count (r~0.4-0.7)
  - Text cluster: title_length, has_description (r~0.3+)
  - Estimate accuracy <-> estimate features (r~0.7+)
  - Category: is_wine, is_jewelry, is_decorative, is_asian_art (sparse but correlated with estimates)
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
# Category H: Extraction Confidence (4) -- Sotheby's only, NaN for others
# Creates a soft cluster: confidence scores are correlated because
# well-documented lots tend to have high confidence across all fields.
# ======================================================================

def extract_confidence_features(lot: dict) -> dict[str, float]:
    """Features 36-39: extraction confidence scores (Sotheby's silver_extractions)."""
    features: dict[str, float] = {}

    def _conf(key: str) -> float:
        v = lot.get(key)
        if v is not None and not pd.isna(v):
            return float(v)
        return np.nan

    features["artist_name_confidence"] = _conf("artist_name_confidence")
    features["dimensions_confidence"] = _conf("dimensions_confidence")
    features["medium_confidence"] = _conf("medium_confidence")
    features["creation_confidence"] = _conf("creation_confidence")

    return features


# ======================================================================
# Category I: Sale Mechanics (5) -- Sotheby's has rich bid data
# Creates collinearity: num_bids <-> hammer_start_ratio <-> starting_bid
# ======================================================================

def extract_sale_mechanics_features(lot: dict) -> dict[str, float]:
    """Features 40-44: bid and sale mechanics."""
    features: dict[str, float] = {}

    def _pos_val(v):
        return v is not None and not pd.isna(v) and float(v) > 0

    nb = lot.get("num_bids")
    features["num_bids"] = float(nb) if _pos_val(nb) else np.nan

    sb = lot.get("starting_bid")
    features["log_starting_bid"] = float(np.log1p(float(sb))) if _pos_val(sb) else np.nan

    bid_method = lot.get("bid_method", "")
    if not isinstance(bid_method, str):
        bid_method = ""
    features["is_online_sale"] = 1.0 if bid_method.lower().strip() == "online" else 0.0

    rm = lot.get("reserve_met")
    if rm is not None and not pd.isna(rm):
        features["reserve_met"] = float(rm)
    else:
        features["reserve_met"] = np.nan

    # Ratio of hammer to starting bid (competitive bidding indicator)
    hp = lot.get("hammer_price")
    if _pos_val(hp) and _pos_val(sb):
        features["hammer_start_ratio"] = float(float(hp) / float(sb))
    else:
        features["hammer_start_ratio"] = np.nan

    return features


# ======================================================================
# Category J: Creator Attribution (2) -- Christie's has creator_type
# Correlated with estimates: attributed works fetch less than autograph
# ======================================================================

def extract_attribution_features(lot: dict) -> dict[str, float]:
    """Features 45-46: creator attribution type."""
    features: dict[str, float] = {}

    ct = lot.get("creator_type", "")
    if not isinstance(ct, str):
        ct = ""
    ct = ct.lower().strip()

    # "artist" = fully attributed; anything else = weaker attribution
    features["is_attributed_artist"] = 1.0 if ct == "artist" else (0.0 if ct else np.nan)
    features["is_maker_not_artist"] = 1.0 if ct in ("maker", "producer", "artisan") else (0.0 if ct else np.nan)

    return features


# ======================================================================
# Category K: Provenance & Literature (3) -- Christie's has arrays
# Strong multicollinearity: lots with provenance tend to have literature
# ======================================================================

def extract_provenance_features(lot: dict) -> dict[str, float]:
    """Features 47-49: provenance, literature, and exhibition counts."""
    features: dict[str, float] = {}

    def _array_len(key: str) -> float:
        v = lot.get(key)
        if v is not None:
            if isinstance(v, (list, tuple)):
                return float(len(v))
            if isinstance(v, str) and v.strip():
                return 1.0
        return 0.0

    features["provenance_count"] = _array_len("provenance")
    features["literature_count"] = _array_len("literature")
    features["exhibition_count"] = _array_len("exhibitions")

    return features


# ======================================================================
# Category L: Text Signals (4)
# Correlated cluster: longer titles go with richer descriptions
# ======================================================================

def extract_text_features(lot: dict) -> dict[str, float]:
    """Features 50-53: text-derived signals."""
    features: dict[str, float] = {}

    title = lot.get("title", "") or lot.get("lot_title", "")
    if not isinstance(title, str):
        title = ""
    features["title_length"] = float(len(title))

    desc = lot.get("raw_description", "") or lot.get("lot_essay_excerpt", "")
    features["has_description"] = 1.0 if (isinstance(desc, str) and len(desc.strip()) > 10) else 0.0

    signed = lot.get("signed_inscribed", "")
    features["has_signed_inscribed"] = 1.0 if (isinstance(signed, str) and signed.strip()) else 0.0

    condition = lot.get("condition_summary", "")
    features["has_condition_report"] = 1.0 if (isinstance(condition, str) and condition.strip()) else 0.0

    return features


# ======================================================================
# Category M: Style & Period (2)
# ======================================================================

def extract_style_features(lot: dict) -> dict[str, float]:
    """Features 54-55: style and period indicators."""
    features: dict[str, float] = {}

    style = lot.get("style_period", "")
    features["has_style_period"] = 1.0 if (isinstance(style, str) and style.strip()) else 0.0

    origin = lot.get("origin", "")
    features["has_origin"] = 1.0 if (isinstance(origin, str) and origin.strip()) else 0.0

    return features


# ======================================================================
# Category N: Lot Classification (4) -- from Christie's lot_category
# Sparse indicators correlated with estimates and medium features
# ======================================================================

def extract_lot_category_features(lot: dict) -> dict[str, float]:
    """Features 56-59: lot category indicators (Christie's lot_category)."""
    features: dict[str, float] = {}

    cat = lot.get("lot_category", "")
    if not isinstance(cat, str):
        cat = ""
    cat = cat.lower().strip()

    features["is_wine"] = 1.0 if cat == "wine" else 0.0
    features["is_jewelry"] = 1.0 if cat == "jewelry" else 0.0
    features["is_book"] = 1.0 if cat == "book" else 0.0
    features["is_asian_art"] = 1.0 if cat == "asian_art" else 0.0

    return features


# ======================================================================
# Category O: Sale & Tax Flags (3)
# ======================================================================

def extract_sale_flag_features(lot: dict) -> dict[str, float]:
    """Features 60-62: sale flags and premium ratio."""
    features: dict[str, float] = {}

    def _pos_val(v):
        return v is not None and not pd.isna(v) and float(v) > 0

    ac = lot.get("accepts_crypto")
    features["has_crypto"] = float(ac) if ac is not None and not pd.isna(ac) else 0.0

    g = lot.get("is_guaranteed")
    features["is_guaranteed"] = float(g) if g is not None and not pd.isna(g) else 0.0

    # Final price / hammer price = buyer's premium ratio
    hp = lot.get("hammer_price")
    fp = lot.get("final_price")
    if _pos_val(hp) and _pos_val(fp):
        features["final_hammer_ratio"] = float(float(fp) / float(hp))
    else:
        features["final_hammer_ratio"] = np.nan

    return features


# ======================================================================
# Category P: Estimate Accuracy (3)
# Creates additional collinearity with D (estimate cluster)
# ======================================================================

def extract_estimate_accuracy_features(lot: dict) -> dict[str, float]:
    """Features 63-65: estimate accuracy and relative level."""
    features: dict[str, float] = {}

    def _pos_val(v):
        return v is not None and not pd.isna(v) and float(v) > 0

    est_low = lot.get("estimate_low_usd")
    est_high = lot.get("estimate_high_usd")

    if _pos_val(est_low) and _pos_val(est_high):
        mid = (float(est_low) + float(est_high)) / 2.0
        features["estimate_mid_usd"] = float(np.log1p(mid))
        rng = float(est_high) - float(est_low)
        features["log_estimate_range_usd"] = float(np.log1p(max(0.0, rng)))
    else:
        features["estimate_mid_usd"] = np.nan
        features["log_estimate_range_usd"] = np.nan

    # Relative level: how this lot's estimate compares to artist average
    artist_avg = lot.get("artist_avg_log_price")
    if _pos_val(est_low) and _pos_val(est_high) and artist_avg and not pd.isna(artist_avg):
        mid_log = np.log1p((float(est_low) + float(est_high)) / 2.0)
        features["estimate_relative_level"] = float(mid_log - float(artist_avg))
    else:
        features["estimate_relative_level"] = np.nan

    return features


# ======================================================================
# Aggregation
# ======================================================================

def build_feature_vector(lot: dict) -> dict[str, float]:
    """Build a single flat feature dict for one auction lot (65 features)."""
    features: dict[str, float] = {}
    # Original 35 features (A-G)
    features.update(extract_artist_features(lot))
    features.update(extract_physical_features(lot))
    features.update(extract_medium_features(lot))
    features.update(extract_estimate_features(lot))
    features.update(extract_sale_features(lot))
    features.update(extract_historical_features(lot))
    features.update(extract_derived_features(lot))
    # New 30 features (H-P)
    features.update(extract_confidence_features(lot))
    features.update(extract_sale_mechanics_features(lot))
    features.update(extract_attribution_features(lot))
    features.update(extract_provenance_features(lot))
    features.update(extract_text_features(lot))
    features.update(extract_style_features(lot))
    features.update(extract_lot_category_features(lot))
    features.update(extract_sale_flag_features(lot))
    features.update(extract_estimate_accuracy_features(lot))
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

    # Lot category (Christie's)
    lot_cat = lot.get("lot_category", "")
    labels["lot_category"] = str(lot_cat) if (isinstance(lot_cat, str) and lot_cat.strip()) else None

    return labels
