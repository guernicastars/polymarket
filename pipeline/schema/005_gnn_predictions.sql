-- ============================================================
-- Polymarket ClickHouse Schema Migration 005 — GNN-TCN Predictions
-- Stores model outputs, calibrated probabilities, and backtest results
-- ============================================================

USE polymarket;

-- ============================================================
-- 1. GNN Model Predictions (per-market temporal predictions)
-- ============================================================
CREATE TABLE IF NOT EXISTS gnn_predictions
(
    -- Market identity
    condition_id       LowCardinality(String),
    settlement_id      String DEFAULT '',                  -- Donbas settlement node ID
    market_slug        String DEFAULT '',

    -- Model output
    raw_logit          Float64 DEFAULT 0,                  -- Pre-calibration model output
    calibrated_prob    Float64 DEFAULT 0,                  -- Post-Platt-scaling probability
    market_price       Float64 DEFAULT 0,                  -- Polymarket price at prediction time
    edge               Float64 DEFAULT 0,                  -- calibrated_prob - market_price

    -- Trading signal
    direction          LowCardinality(String) DEFAULT '',  -- BUY / SELL / HOLD
    kelly_fraction     Float64 DEFAULT 0,                  -- Raw Kelly criterion
    position_size_usd  Float64 DEFAULT 0,                  -- Recommended size after hurdle
    hurdle_rate        Float64 DEFAULT 0,                  -- Dynamic hurdle at prediction time

    -- Model metadata
    model_version      String DEFAULT '',                   -- Checkpoint identifier
    window_size        UInt32 DEFAULT 64,
    step_minutes       UInt32 DEFAULT 5,
    confidence         Float64 DEFAULT 0,                   -- Model confidence score

    -- Feature summary (for debugging / explainability)
    top_features       String DEFAULT '{}' CODEC(ZSTD(3)), -- JSON: top contributing features

    -- Timestamps
    predicted_at       DateTime64(3) CODEC(DoubleDelta, LZ4),
    target_time        DateTime64(3) DEFAULT now64(3),     -- What time the prediction targets

    -- Indexes
    INDEX condition_idx condition_id TYPE bloom_filter GRANULARITY 4,
    INDEX settlement_idx settlement_id TYPE bloom_filter GRANULARITY 4,
    INDEX direction_idx direction TYPE set(5) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (condition_id, predicted_at)
TTL predicted_at + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 2. GNN Backtest Results (per-run aggregate metrics)
-- ============================================================
CREATE TABLE IF NOT EXISTS gnn_backtest_runs
(
    -- Run identity
    run_id             String,                              -- UUID
    model_version      String DEFAULT '',

    -- Configuration
    train_start        DateTime64(3),
    train_end          DateTime64(3),
    test_start         DateTime64(3),
    test_end           DateTime64(3),
    window_size        UInt32 DEFAULT 64,
    step_minutes       UInt32 DEFAULT 5,
    n_epochs_trained   UInt32 DEFAULT 0,

    -- Aggregate metrics
    n_trades           UInt32 DEFAULT 0,
    win_rate           Float64 DEFAULT 0,
    sharpe_ratio       Float64 DEFAULT 0,
    max_drawdown       Float64 DEFAULT 0,
    total_return       Float64 DEFAULT 0,
    profit_factor      Float64 DEFAULT 0,
    calmar_ratio       Float64 DEFAULT 0,
    avg_edge           Float64 DEFAULT 0,

    -- Train/val loss
    final_train_loss   Float64 DEFAULT 0,
    final_val_loss     Float64 DEFAULT 0,
    best_val_loss      Float64 DEFAULT 0,

    -- Platt calibration
    platt_a            Array(Float64),                      -- Per-target scaling factors
    platt_b            Array(Float64),                      -- Per-target bias terms

    -- Timestamp
    created_at         DateTime64(3) DEFAULT now64(3),

    INDEX run_idx run_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
ORDER BY (run_id, created_at)
TTL created_at + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. GNN Backtest Trades (individual trade log per run)
-- ============================================================
CREATE TABLE IF NOT EXISTS gnn_backtest_trades
(
    -- Run identity
    run_id             String,
    trade_idx          UInt32 DEFAULT 0,

    -- Trade details
    market_idx         UInt8 DEFAULT 0,
    market_name        String DEFAULT '',
    direction          LowCardinality(String) DEFAULT '',
    model_prob         Float64 DEFAULT 0,
    market_price       Float64 DEFAULT 0,
    exit_price         Float64 DEFAULT 0,
    size_usd           Float64 DEFAULT 0,
    pnl                Float64 DEFAULT 0,
    costs              Float64 DEFAULT 0,

    -- Timestamp
    trade_time         DateTime64(3) CODEC(DoubleDelta, LZ4),

    INDEX run_idx run_id TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_time)
ORDER BY (run_id, trade_time, trade_idx)
TTL trade_time + INTERVAL 1 YEAR DELETE
SETTINGS index_granularity = 8192;
