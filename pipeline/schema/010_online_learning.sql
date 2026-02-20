-- ============================================================
-- Polymarket ClickHouse Schema Migration 010 — Online Learning State
-- Tracks incremental GNN-TCN model updates
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. Online Learning State (per update cycle metrics)
-- ============================================================
CREATE TABLE IF NOT EXISTS online_learning_state
(
    update_id           UInt64,
    model_version       String DEFAULT '',

    -- Update metrics
    n_samples           UInt32 DEFAULT 0,
    n_gradient_steps    UInt32 DEFAULT 0,
    avg_loss            Float64 DEFAULT 0,
    max_grad_norm       Float64 DEFAULT 0,
    learning_rate       Float64 DEFAULT 0,
    ema_decay           Float64 DEFAULT 0,

    -- Platt scaling state
    platt_a             Float64 DEFAULT 1,
    platt_b             Float64 DEFAULT 0,

    -- Timestamp
    updated_at          DateTime64(3) CODEC(DoubleDelta, LZ4)
)
ENGINE = MergeTree()
ORDER BY updated_at
TTL toDateTime(updated_at) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;
