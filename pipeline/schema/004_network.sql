-- Phase 4: Network model tables for Donbas settlement graph signals
-- Stores vulnerability scores, supply chain analysis, cascade results, and trading signals

CREATE TABLE IF NOT EXISTS network_vulnerability (
    settlement_id     String,
    connectivity_score Float64,
    supply_score       Float64,
    force_balance_score Float64,
    terrain_score      Float64,
    fortification_score Float64,
    assault_intensity_score Float64,
    frontline_score    Float64,
    composite          Float64,
    computed_at        DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (settlement_id)
TTL computed_at + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS network_supply_risk (
    settlement_id     String,
    origin            String DEFAULT 'dnipro',
    shortest_cost     Float64,
    path_redundancy   UInt32,
    min_cut_size      UInt32,
    min_cut_nodes     Array(String),
    supply_risk       Float64,
    computed_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (settlement_id, origin)
TTL computed_at + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS network_cascade_scenarios (
    trigger_node      String,
    fallen_nodes      Array(String),
    isolated_nodes    Array(String),
    supply_cut_nodes  Array(String),
    new_component_count UInt32,
    severity          Float64,
    computed_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (trigger_node)
TTL computed_at + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS network_signals (
    settlement_id     String,
    market_slug       String,
    model_probability Float64,
    market_probability Float64,
    edge              Float64,
    direction         String,
    kelly_fraction    Float64,
    confidence        Float64,
    computed_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (settlement_id, market_slug)
TTL computed_at + INTERVAL 90 DAY;
