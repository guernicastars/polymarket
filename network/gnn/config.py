"""Configuration for the GNN-TCN temporal prediction model.

Architecture follows Anastasiia's 12-feature TCN design adapted for
graph-structured data. Supports both geographic settlement graphs
(Ukraine, Middle East) and generic market similarity graphs.

In the OpenForage-inspired ensemble architecture, the GNN-TCN model is
one signal source among many — its predictions feed into the quality-gated
SignalEnsemble (see pipeline.causal.signal_ensemble) alongside composite
signals, causal analysis outputs, and other predictors. The model's
Platt-calibrated probability outputs are evaluated through the same
in-sample/out-of-sample/uniqueness pipeline as all other signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureConfig:
    """12 temporal features per node per timestep (Anastasiia's spec)."""

    # Timestep resolution — each of the 64 steps is this many minutes
    step_minutes: int = 5

    # Window size — must be power of 2 for TCN efficiency
    window_size: int = 64

    # Number of features per node per timestep
    n_features: int = 12

    # Feature names (ordered, matching tensor columns)
    feature_names: list[str] = field(default_factory=lambda: [
        "log_returns",            # F1: ln(price_t / price_{t-1})
        "high_low_spread",        # F2: high - low within the step
        "dist_from_ma",           # F3: price - SMA_12 (1-hour MA at 5-min steps)
        "bid_ask_spread",         # F4: best_ask - best_bid
        "obi",                    # F5: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        "depth_ratio",            # F6: top5_depth / top1_depth (spoof detection)
        "volume_delta",           # F7: vol_step / SMA_vol_last_2steps - 1
        "open_interest_change",   # F8: delta(market holders * amount) over step
        "sentiment_score",        # F9: from composite_signals.smart_money_score (proxy)
        "news_velocity",          # F10: count of trades > $5K in step (news proxy)
        "inv_time_to_expiry",     # F11: 1 / max(days_to_end_date, 1)
        "correlation_delta",      # F12: price change of most-correlated sibling market
    ])


@dataclass
class ModelConfig:
    """GNN-TCN architecture hyperparameters."""

    # --- Graph Attention Network ---
    gat_in_features: int = 12       # matches n_features
    gat_hidden: int = 32            # GAT hidden dimension
    gat_heads: int = 4              # multi-head attention
    gat_out: int = 32               # output per node after GAT
    gat_dropout: float = 0.1

    # --- Temporal Convolutional Network ---
    tcn_channels: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2

    # --- Prediction head ---
    fc_hidden: int = 64
    fc_dropout: float = 0.3
    n_targets: int = 10             # 10 Polymarket target settlements

    # --- Training ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    patience: int = 15              # early stopping patience
    grad_clip: float = 1.0

    # --- Platt scaling ---
    platt_lr: float = 0.01
    platt_epochs: int = 200

    # --- Window ---
    window_size: int = 64
    step_minutes: int = 5


@dataclass
class BacktestConfig:
    """Backtesting parameters (Anastasiia's Point 5)."""

    # Train/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Simulated execution costs
    latency_ms: int = 200           # simulated execution latency
    spread_penalty: float = 0.005   # 0.5% spread cost

    # Risk metrics
    risk_free_rate: float = 0.05    # annual risk-free rate for Sharpe
    max_position_pct: float = 0.25  # max 25% of capital per trade

    # Dynamic hurdle rate (Point 4)
    base_hurdle: float = 0.02       # minimum edge to trade
    impact_coefficient: float = 0.1 # price impact per $1K traded
    max_trade_size_usd: float = 5000.0

    # Kelly criterion
    kelly_fraction: float = 0.25    # fractional Kelly (quarter Kelly for safety)


@dataclass
class GraphConfig:
    """Graph construction configuration for market similarity graphs."""

    # Similarity method: 'event', 'whale', 'correlation', 'signal', 'category', 'combined'
    method: str = "combined"

    # Component weights (must sum to ~1.0 for combined method)
    event_weight: float = 0.30       # same event_slug → 1.0
    whale_weight: float = 0.20       # Jaccard of top-20 holders
    correlation_weight: float = 0.25  # Pearson of hourly log returns
    signal_weight: float = 0.15      # cosine sim of 8 composite_signals components
    category_weight: float = 0.10    # same category tag → 1.0

    # Graph construction thresholds
    min_similarity: float = 0.15     # edges below this become 0
    add_self_loops: bool = True
    symmetric: bool = True

    # Query tuning
    price_lookback_days: int = 7
    min_data_points: int = 50        # min hourly bars for correlation
    top_markets: int = 500           # max markets for similarity computation


@dataclass
class GNNConfig:
    """Top-level configuration aggregating all sub-configs."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Graph type: 'settlement' (geographic), 'market' (generic similarity)
    graph_type: str = "settlement"

    # ClickHouse connection (inherited from pipeline env)
    clickhouse_host: str = "ch.bloomsburytech.com"
    clickhouse_port: int = 443
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "polymarket"

    # Paths
    model_save_dir: str = "network/gnn/checkpoints"
    log_dir: str = "network/gnn/logs"


@dataclass
class OnlineLearningConfig:
    """Configuration for online/incremental GNN-TCN updates."""

    online_lr: float = 1e-4               # 10x lower than batch LR
    online_weight_decay: float = 1e-5
    ema_decay: float = 0.995              # EMA smoothing for weight stability
    min_samples_before_update: int = 16   # Minimum new samples to trigger SGD
    max_gradient_steps: int = 5           # Steps per update cycle
    online_grad_clip: float = 0.5         # Tighter than batch (1.0)
    platt_recalibrate_interval: int = 50  # Recalibrate Platt every N updates
    cold_start_prob: float = 0.5          # Default when no checkpoint exists
    checkpoint_save_interval: int = 10    # Save every N updates
    cold_start_max_steps: int = 50        # Extended initial training steps
    recalibration_buffer_size: int = 200  # Recent logit/label pairs for Platt refit

    # Features to use (indices into 12-feature vector)
    # Skip F8 (OI change=0), F9 (sentiment=0), F10 (news=0)
    feature_mask: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 11]
    )
