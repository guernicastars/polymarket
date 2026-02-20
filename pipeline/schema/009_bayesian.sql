-- ============================================================
-- Polymarket ClickHouse Schema Migration 009 — Bayesian Predictions
-- Two-layer architecture: GNN-TCN base model + Bayesian combiner
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Bayesian Predictions (posterior output per market per cycle)
-- ============================================================
CREATE TABLE IF NOT EXISTS bayesian_predictions
(
    -- Market identity
    condition_id        LowCardinality(String),

    -- Posterior state (Beta distribution parameters)
    posterior_mean      Float64 DEFAULT 0,
    posterior_alpha     Float64 DEFAULT 1,
    posterior_beta      Float64 DEFAULT 1,
    credible_lo         Float64 DEFAULT 0,       -- 95% CI lower bound
    credible_hi         Float64 DEFAULT 1,       -- 95% CI upper bound

    -- Prior / market reference
    market_price        Float64 DEFAULT 0,
    edge                Float64 DEFAULT 0,       -- posterior_mean - market_price

    -- Signal quality
    confidence          Float64 DEFAULT 0,       -- Normalized concentration
    n_evidence_sources  UInt8 DEFAULT 0,
    evidence_agreement  Float64 DEFAULT 0,       -- Fraction of signals agreeing

    -- Trading signal
    direction           LowCardinality(String) DEFAULT 'HOLD',
    kelly_fraction      Float64 DEFAULT 0,
    position_size_usd   Float64 DEFAULT 0,

    -- Evidence breakdown (JSON: {source: {K, weight, direction}})
    evidence_detail     String DEFAULT '{}' CODEC(ZSTD(3)),

    -- Timestamps
    predicted_at        DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX cid_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX dir_idx direction TYPE set(5) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (condition_id, predicted_at)
TTL toDateTime(predicted_at) + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Bayesian State (per-market posterior for crash recovery)
-- ============================================================
CREATE TABLE IF NOT EXISTS bayesian_state
(
    condition_id        LowCardinality(String),
    alpha               Float64 DEFAULT 1,
    beta                Float64 DEFAULT 1,
    n_updates           UInt32 DEFAULT 0,
    last_market_price   Float64 DEFAULT 0.5,
    updated_at          DateTime64(3) CODEC(DoubleDelta, LZ4)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY condition_id
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. Calibration History (rolling metrics per source)
-- ============================================================
CREATE TABLE IF NOT EXISTS calibration_history
(
    source              LowCardinality(String),   -- 'bayesian', 'gnn', 'composite', 'market'
    brier_score         Float64 DEFAULT 0,
    n_predictions       UInt32 DEFAULT 0,
    n_resolved          UInt32 DEFAULT 0,
    calibration_bins    String DEFAULT '{}',       -- JSON: binned calibration curve
    reliability_adj     Float64 DEFAULT 1.0,
    computed_at         DateTime64(3) CODEC(DoubleDelta, LZ4)
)
ENGINE = MergeTree()
ORDER BY (source, computed_at)
TTL toDateTime(computed_at) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;
