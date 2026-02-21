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
    Concatenate AE + transformer embeddings, run probes on combined vector.
    Tests whether temporal patterns add information beyond summary statistics.
"""
