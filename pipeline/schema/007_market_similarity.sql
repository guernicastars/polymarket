-- ============================================================
-- Polymarket ClickHouse Schema Migration 007 — Market Similarity Graph
-- Pre-computed pairwise market similarity for GNN graph construction
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Market Similarity Graph (pairwise edges)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_similarity_graph
(
    -- Edge endpoints (condition_ids)
    source_id          LowCardinality(String),
    target_id          LowCardinality(String),

    -- Composite similarity score [0, 1]
    similarity_score   Float64 DEFAULT 0,

    -- Component breakdown (JSON: {event, whale, correlation, signal, category})
    components         String DEFAULT '{}' CODEC(ZSTD(3)),

    -- Method used to compute
    method             LowCardinality(String) DEFAULT 'combined',

    -- Timestamps
    computed_at        DateTime64(3) DEFAULT now64(3),

    -- Indexes
    INDEX source_idx source_id TYPE bloom_filter GRANULARITY 4,
    INDEX target_idx target_id TYPE bloom_filter GRANULARITY 4,
    INDEX score_idx similarity_score TYPE minmax GRANULARITY 4
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (source_id, target_id)
TTL toDateTime(computed_at) + INTERVAL 30 DAY DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. Graph Metrics (per-refresh aggregate stats)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_graph_metrics
(
    -- Graph snapshot identity
    method             LowCardinality(String),
    refresh_time       DateTime64(3),

    -- Graph structure metrics
    node_count         UInt32 DEFAULT 0,
    edge_count         UInt32 DEFAULT 0,
    density            Float64 DEFAULT 0,
    avg_degree         Float64 DEFAULT 0,
    clustering_coeff   Float64 DEFAULT 0,

    -- Edge weight distribution
    min_weight         Float64 DEFAULT 0,
    max_weight         Float64 DEFAULT 0,
    median_weight      Float64 DEFAULT 0,

    -- Timestamps
    computed_at        DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (method, refresh_time)
TTL toDateTime(computed_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;
