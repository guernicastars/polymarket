"""Market embedding module: autoencoder + transformer + linear probes.

Two complementary embedding architectures:

  Autoencoder (static snapshot):
    ClickHouse (resolved markets)
      -> ResolvedMarketDataset (27 summary features per lifecycle)
      -> MarketAutoencoder / VariationalAutoencoder
      -> LinearProbe + DisentanglementAnalyzer

  Transformer (temporal dynamics):
    ClickHouse (all markets — active + resolved)
      -> TemporalMarketDataset (variable-length hourly bar sequences)
      -> MarketTransformer (patch-based encoder-only, pre-trained via MPP)
      -> LinearProbe (same framework, apples-to-apples comparison)

  Fusion:
    Two methods for combining AE + transformer embeddings:
      - Concatenation: simple concat, run probes on combined vector
      - Cross-attention: learnable gated fusion via bidirectional cross-attention
    Temporal probes (trajectory shape, momentum profile, volume pattern) test
    whether temporal dynamics add information beyond summary statistics.
"""
