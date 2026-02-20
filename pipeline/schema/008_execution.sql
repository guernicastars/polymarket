-- Phase 5: Execution layer tables
-- Tracks orders, positions, and portfolio snapshots for post-hoc analysis.

CREATE TABLE IF NOT EXISTS execution_orders (
    order_id        String,
    condition_id    String,
    token_id        String,
    side            String,
    price           Float64,
    size            Float64,
    edge            Float64,
    kelly_fraction  Float64,
    confidence      Float64,
    signal_source   String,
    status          String,
    error_msg       String DEFAULT '',
    fill_price      Nullable(Float64),
    filled_size     Nullable(Float64),
    latency_ms      Float64 DEFAULT 0,
    submitted_at    DateTime64(3, 'UTC'),
    created_at      DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (submitted_at, condition_id)
TTL toDateTime(created_at) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS execution_positions (
    condition_id    String,
    token_id        String,
    side            String,
    entry_price     Float64,
    size            Float64,
    cost_basis      Float64,
    current_price   Float64,
    unrealized_pnl  Float64,
    realized_pnl    Float64,
    signal_source   String,
    edge_at_entry   Float64,
    opened_at       DateTime64(3, 'UTC'),
    snapshot_at     DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (snapshot_at, condition_id)
TTL toDateTime(snapshot_at) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS execution_snapshots (
    timestamp           DateTime64(3, 'UTC'),
    capital             Float64,
    total_value         Float64,
    n_positions         UInt32,
    total_unrealized_pnl Float64,
    total_realized_pnl  Float64,
    total_cost_basis    Float64,
    max_position_value  Float64,
    high_water_mark     Float64,
    current_drawdown    Float64,
    mode                String DEFAULT 'DRY_RUN'
) ENGINE = MergeTree()
ORDER BY timestamp
TTL toDateTime(timestamp) + INTERVAL 90 DAY;
