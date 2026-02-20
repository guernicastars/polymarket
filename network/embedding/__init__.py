"""Market embedding PoC: autoencoder + linear probes on resolved Polymarket data.

Research question: Do neural embeddings solve multicollinearity? Are interpretable
concepts linearly separable in the learned embedding space?

Pipeline:
  ClickHouse (resolved markets)
    -> ResolvedMarketDataset (summary features per market lifecycle)
    -> MarketAutoencoder / VariationalAutoencoder (compress to latent space)
    -> LinearProbe (test separability of known concepts)
    -> DisentanglementAnalyzer (PCA, novel directions, stability)
    -> FeatureInterpreter (trace back to input features, plain-language output)
"""
