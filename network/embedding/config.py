"""Configuration for the embedding PoC.

Follows the dataclass pattern from network.gnn.config — one class per concern,
aggregated into a top-level EmbeddingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingFeatureConfig:
    """Summary features extracted per resolved market lifecycle."""

    # Minimum total volume (USD) to include a market
    min_volume_total: float = 1000.0

    # Minimum market lifespan in days
    min_duration_days: int = 1

    # Maximum markets to load (0 = unlimited)
    max_markets: int = 0

    # How far back to search for resolved markets
    lookback_months: int = 12

    # Lifetime cutoff ratio: 0.8 = use first 80% of life (leakage-safe), 1.0 = full
    lifetime_cutoff_ratio: float = 0.8

    # Feature group toggles (for ablation)
    include_price: bool = True        # 7 features
    include_volume: bool = True       # 5 features
    include_liquidity: bool = True    # 4 features
    include_participation: bool = True  # 4 features
    include_temporal: bool = True     # 3 features
    include_structure: bool = True    # 4 features

    # Total expected features (validated at runtime)
    n_features: int = 27


@dataclass
class AutoencoderConfig:
    """Autoencoder architecture and training hyperparameters."""

    # Architecture
    latent_dim: int = 64
    encoder_hidden: list[int] = field(default_factory=lambda: [128, 64])
    decoder_hidden: list[int] = field(default_factory=lambda: [64, 128])
    dropout: float = 0.1
    use_batch_norm: bool = True

    # VAE option
    variational: bool = False
    kl_weight: float = 0.001  # beta for beta-VAE

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ProbeConfig:
    """Linear probe hyperparameters."""

    learning_rate: float = 1e-2
    epochs: int = 100
    cv_folds: int = 5

    # Statistical significance
    alpha: float = 0.05
    n_permutation_tests: int = 1000

    # Novel direction search
    n_pca_components: int = 20
    min_correlation_threshold: float = 0.3


@dataclass
class EmbeddingConfig:
    """Top-level configuration aggregating all sub-configs."""

    features: EmbeddingFeatureConfig = field(default_factory=EmbeddingFeatureConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # ClickHouse connection (same fields as GNNConfig)
    clickhouse_host: str = "ch.bloomsburytech.com"
    clickhouse_port: int = 443
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "polymarket"

    # Paths
    model_save_dir: str = "network/embedding/checkpoints"
    results_dir: str = "network/embedding/results"
    log_dir: str = "network/embedding/logs"
