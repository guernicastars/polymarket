"""Configuration for the embedding module.

Follows the dataclass pattern from network.gnn.config — one class per concern,
aggregated into a top-level EmbeddingConfig.

Two embedding architectures:
  - Autoencoder: summary features (27-dim) → latent embedding (static snapshot)
  - Transformer: hourly time series (variable-length) → latent embedding (temporal dynamics)
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
    max_iter: int = 5000
    cv_folds: int = 5

    # Statistical significance
    alpha: float = 0.05
    n_permutation_tests: int = 1000

    # Novel direction search
    n_pca_components: int = 20
    min_correlation_threshold: float = 0.3


@dataclass
class TransformerConfig:
    """Patch-based encoder-only transformer for temporal market embeddings."""

    # --- Patch tokenization ---
    # Group consecutive hourly bars into patches; each patch becomes one token.
    # 24h patch → a 30-day market yields ~30 tokens.
    patch_size: int = 24          # hours per patch
    n_input_features: int = 12   # features per hourly bar (reuses GNN's 12 features)

    # --- Transformer architecture ---
    d_model: int = 64            # embedding dimension (matches AE latent_dim for fusion)
    n_heads: int = 4             # attention heads (d_model / n_heads = 16 per head)
    n_layers: int = 3            # encoder layers
    d_ff: int = 256              # feed-forward hidden dim (4x d_model)
    dropout: float = 0.2         # higher than AE (0.1) — transformers need more with small data
    attn_dropout: float = 0.1    # attention weight dropout
    pre_norm: bool = True        # pre-norm (more stable for small models)

    # --- Positional encoding ---
    max_patches: int = 128       # max sequence length in patches (~128 days)
    use_relative_pos: bool = True  # position as fraction of market lifetime [0, 1]

    # --- Pooling ---
    use_cls_token: bool = True   # CLS token for embedding extraction

    # --- Pre-training (Masked Patch Prediction) ---
    mask_ratio: float = 0.30     # fraction of patches to mask during pre-training
    pretrain_lr: float = 5e-4
    pretrain_weight_decay: float = 1e-3
    pretrain_epochs: int = 100
    pretrain_patience: int = 15
    pretrain_batch_size: int = 64
    pretrain_warmup_steps: int = 200  # linear warmup before cosine decay

    # --- Fine-tuning ---
    finetune_lr: float = 1e-4    # lower LR for fine-tuning
    finetune_epochs: int = 50
    finetune_patience: int = 10
    finetune_batch_size: int = 32

    # --- Training common ---
    grad_clip: float = 1.0

    # --- Temporal dataset ---
    bar_interval_hours: int = 1   # aggregate to 1h bars
    min_bars: int = 48            # minimum 48 hours of data to include a market
    max_bars: int = 3072          # cap at 128 days of hourly data (128 * 24)
    include_active: bool = True   # include active (unresolved) markets for pre-training
    min_volume_pretrain: float = 100.0   # lower threshold for pre-training data
    min_volume_finetune: float = 1000.0  # standard threshold for fine-tuning

    # --- Fusion ---
    fusion_method: str = "concat"  # 'concat' or 'cross_attention'


@dataclass
class EmbeddingConfig:
    """Top-level configuration aggregating all sub-configs."""

    features: EmbeddingFeatureConfig = field(default_factory=EmbeddingFeatureConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # ClickHouse connection (same fields as GNNConfig)
    clickhouse_host: str = "ch.bloomsburytech.com"
    clickhouse_port: int = 443
    clickhouse_user: str = "default"
    clickhouse_password: str = "clickhouse_admin_2026"
    clickhouse_database: str = "polymarket"

    # Paths
    model_save_dir: str = "network/embedding/checkpoints"
    results_dir: str = "network/embedding/results"
    log_dir: str = "network/embedding/logs"
