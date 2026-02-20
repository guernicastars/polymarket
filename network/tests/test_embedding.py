"""Tests for the embedding PoC module.

Tests cover:
  - Autoencoder forward pass shapes (AE and VAE)
  - Gradient flow through encoder/decoder
  - Reconstruction loss decreases over training
  - VAE: KL divergence, reparameterization
  - Linear probe correctness on synthetic data
  - Disentanglement analysis (PCA, orthogonality, novel directions)
  - Feature interpreter (correlation attribution)
  - Config validation
  - Dataset collation
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from network.embedding.config import (
    AutoencoderConfig,
    EmbeddingConfig,
    EmbeddingFeatureConfig,
    ProbeConfig,
)
from network.embedding.model import (
    MarketAutoencoder,
    VariationalAutoencoder,
    create_autoencoder,
)
from network.embedding.probes import LinearProbe, ProbeResult
from network.embedding.analysis import DisentanglementAnalyzer, NovelDirection
from network.embedding.interpret import FeatureInterpreter
from network.embedding.data import collate_embedding_batch


# ============================================================
# Autoencoder forward pass tests
# ============================================================

class TestMarketAutoencoder:
    """Test the standard autoencoder architecture."""

    def setup_method(self):
        self.cfg = AutoencoderConfig(latent_dim=32, encoder_hidden=[64, 48], decoder_hidden=[48, 64])
        self.input_dim = 27
        self.model = MarketAutoencoder(self.input_dim, self.cfg)

    def test_forward_shape(self):
        x = torch.randn(16, self.input_dim)
        x_hat, z = self.model(x)
        assert x_hat.shape == (16, self.input_dim), f"Expected (16, {self.input_dim}), got {x_hat.shape}"
        assert z.shape == (16, 32), f"Expected (16, 32), got {z.shape}"

    def test_encode_shape(self):
        x = torch.randn(8, self.input_dim)
        z = self.model.encode(x)
        assert z.shape == (8, 32)

    def test_decode_shape(self):
        z = torch.randn(8, 32)
        x_hat = self.model.decode(z)
        assert x_hat.shape == (8, self.input_dim)

    def test_gradient_flow(self):
        x = torch.randn(4, self.input_dim, requires_grad=True)
        x_hat, z = self.model(x)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "No gradient flowing to input"

    def test_single_sample(self):
        self.model.eval()
        x = torch.randn(1, self.input_dim)
        x_hat, z = self.model(x)
        assert x_hat.shape == (1, self.input_dim)
        assert z.shape == (1, 32)

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 100, f"Model too small: {n_params}"
        assert n_params < 1_000_000, f"Model too large: {n_params}"

    def test_reconstruction_decreases(self):
        """Verify that MSE loss decreases over a few training steps."""
        x = torch.randn(32, self.input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            x_hat, z = self.model(x)
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ============================================================
# VAE tests
# ============================================================

class TestVariationalAutoencoder:
    """Test the variational autoencoder."""

    def setup_method(self):
        self.cfg = AutoencoderConfig(
            latent_dim=16, encoder_hidden=[32, 24], decoder_hidden=[24, 32], variational=True
        )
        self.input_dim = 27
        self.model = VariationalAutoencoder(self.input_dim, self.cfg)

    def test_forward_shape(self):
        x = torch.randn(8, self.input_dim)
        x_hat, mu, log_var, z = self.model(x)
        assert x_hat.shape == (8, self.input_dim)
        assert mu.shape == (8, 16)
        assert log_var.shape == (8, 16)
        assert z.shape == (8, 16)

    def test_kl_non_negative(self):
        x = torch.randn(16, self.input_dim)
        x_hat, mu, log_var, z = self.model(x)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        assert kl.item() >= 0, f"KL divergence should be non-negative, got {kl.item()}"

    def test_loss_function(self):
        x = torch.randn(16, self.input_dim)
        x_hat, mu, log_var, z = self.model(x)
        total, recon, kl = VariationalAutoencoder.loss(x, x_hat, mu, log_var, kl_weight=0.001)
        assert total.item() > 0
        assert recon.item() > 0
        assert kl.item() >= 0
        assert abs(total.item() - (recon.item() + 0.001 * kl.item())) < 1e-4

    def test_reparameterization_gradient(self):
        x = torch.randn(4, self.input_dim, requires_grad=True)
        x_hat, mu, log_var, z = self.model(x)
        loss = z.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Reparameterization gradient not flowing"

    def test_eval_deterministic(self):
        """In eval mode, reparameterize returns mu (no sampling)."""
        self.model.eval()
        x = torch.randn(4, self.input_dim)
        with torch.no_grad():
            _, mu1, _, z1 = self.model(x)
            _, mu2, _, z2 = self.model(x)
        assert torch.allclose(z1, z2), "Eval mode should be deterministic"
        assert torch.allclose(z1, mu1), "Eval z should equal mu"


# ============================================================
# Factory test
# ============================================================

class TestFactory:

    def test_creates_ae(self):
        cfg = AutoencoderConfig(variational=False)
        model = create_autoencoder(27, cfg)
        assert isinstance(model, MarketAutoencoder)

    def test_creates_vae(self):
        cfg = AutoencoderConfig(variational=True)
        model = create_autoencoder(27, cfg)
        assert isinstance(model, VariationalAutoencoder)


# ============================================================
# Linear probe tests
# ============================================================

class TestLinearProbe:
    """Test linear probes on synthetic data with known structure."""

    def setup_method(self):
        self.cfg = ProbeConfig(cv_folds=3, n_permutation_tests=50, alpha=0.05)
        self.probe = LinearProbe(self.cfg)

    def test_classification_separable(self):
        """Probe should achieve high accuracy on linearly separable data."""
        rng = np.random.RandomState(42)
        n = 200
        d = 16

        # Create two clearly separable clusters
        embeddings = rng.randn(n, d).astype(np.float32)
        labels = np.zeros(n, dtype=np.int32)
        labels[:n // 2] = 0
        labels[n // 2:] = 1
        # Make first dim perfectly predictive
        embeddings[:n // 2, 0] -= 3.0
        embeddings[n // 2:, 0] += 3.0

        result = self.probe.probe_classification(embeddings, labels, "test_concept")
        assert result.accuracy > 0.85, f"Expected high accuracy on separable data, got {result.accuracy}"
        assert result.task_type == "classification"

    def test_classification_random(self):
        """Probe should achieve ~baseline on random labels."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(200, 16).astype(np.float32)
        labels = rng.randint(0, 2, 200).astype(np.int32)

        result = self.probe.probe_classification(embeddings, labels, "random_concept")
        # Should be close to 50% (baseline for balanced binary)
        assert result.accuracy < 0.7, f"Random labels shouldn't give high accuracy: {result.accuracy}"

    def test_regression(self):
        """Probe should find linear relationship."""
        rng = np.random.RandomState(42)
        n = 200
        d = 16
        embeddings = rng.randn(n, d).astype(np.float32)
        # Target is linear combination of first 3 dims
        targets = embeddings[:, 0] * 2 + embeddings[:, 1] * 1.5 + rng.randn(n) * 0.1

        result = self.probe.probe_regression(embeddings, targets.astype(np.float32), "linear_target")
        assert result.accuracy > 0.5, f"Expected good R^2 on linear target, got {result.accuracy}"

    def test_invalid_labels_skipped(self):
        """Probes with too few valid samples should return empty result."""
        embeddings = np.random.randn(10, 16).astype(np.float32)
        labels = np.full(10, -1, dtype=np.int32)
        result = self.probe.probe_classification(embeddings, labels, "invalid")
        assert result.accuracy == 0.0
        assert result.p_value == 1.0

    def test_run_all_probes(self):
        """Smoke test for run_all_probes with synthetic labels."""
        rng = np.random.RandomState(42)
        n = 100
        embeddings = rng.randn(n, 16).astype(np.float32)
        labels = [
            {
                "outcome_binary": rng.randint(0, 2),
                "category": rng.choice(["politics", "sports", "crypto"]),
                "duration_bucket": rng.choice(["short", "medium", "long"]),
                "volume_bucket": rng.choice(["low", "medium", "high"]),
                "volatility_regime": rng.choice(["low", "high"]),
            }
            for _ in range(n)
        ]

        results = self.probe.run_all_probes(embeddings, labels, [f"feat_{i}" for i in range(16)])
        assert len(results) >= 4  # at least 4 probes should run


# ============================================================
# Analysis tests
# ============================================================

class TestDisentanglementAnalyzer:

    def setup_method(self):
        self.cfg = ProbeConfig(n_pca_components=10, min_correlation_threshold=0.3)
        self.analyzer = DisentanglementAnalyzer(self.cfg)

    def test_pca_analysis(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 32).astype(np.float32)
        raw_features = rng.randn(100, 27).astype(np.float32)
        feature_names = [f"f{i}" for i in range(27)]

        result = self.analyzer.pca_analysis(embeddings, feature_names, raw_features)
        assert "components" in result
        assert "explained_variance_ratio" in result
        assert result["cumulative_variance"][-1] <= 1.0 + 1e-6
        assert result["component_feature_correlations"].shape == (10, 27)

    def test_novel_directions(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 32).astype(np.float32)
        raw_features = rng.randn(100, 10).astype(np.float32)
        feature_names = [f"f{i}" for i in range(10)]

        novel = self.analyzer.search_novel_directions(embeddings, feature_names, raw_features)
        assert isinstance(novel, list)
        for nd in novel:
            assert nd.max_correlation < self.cfg.min_correlation_threshold

    def test_orthogonality(self):
        # Create two orthogonal probes
        coef1 = np.array([1, 0, 0, 0], dtype=np.float32)
        coef2 = np.array([0, 1, 0, 0], dtype=np.float32)
        probes = [
            ProbeResult("a", "clf", 0.8, 0.1, 0.5, 0.01, True, coef1, [0.8]),
            ProbeResult("b", "clf", 0.7, 0.1, 0.5, 0.02, True, coef2, [0.7]),
        ]
        sim_matrix = self.analyzer.orthogonality_test(probes)
        assert sim_matrix.shape == (2, 2)
        assert abs(sim_matrix[0, 1]) < 0.01, "Orthogonal probes should have ~0 cosine sim"
        assert abs(sim_matrix[0, 0] - 1.0) < 0.01, "Self-similarity should be ~1"

    def test_cluster_validation(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(60, 16).astype(np.float32)
        labels = [
            {"category": "politics" if i < 30 else "sports", "outcome_binary": i % 2}
            for i in range(60)
        ]
        result = self.analyzer.cluster_validation(embeddings, labels)
        assert "silhouette_category" in result

    def test_tsne_shape(self):
        embeddings = np.random.randn(50, 16).astype(np.float32)
        coords = self.analyzer.tsne_coordinates(embeddings, perplexity=10.0)
        assert coords.shape == (50, 2)


# ============================================================
# Interpreter tests
# ============================================================

class TestFeatureInterpreter:

    def test_correlation_attribution(self):
        """Attribution should identify the feature most correlated with direction."""
        rng = np.random.RandomState(42)
        n, latent_dim, n_feat = 100, 16, 5

        # Embeddings where first dim correlates with first raw feature
        raw_features = rng.randn(n, n_feat).astype(np.float32)
        embeddings = rng.randn(n, latent_dim).astype(np.float32)
        embeddings[:, 0] = raw_features[:, 0] * 2 + rng.randn(n) * 0.1

        direction = np.zeros(latent_dim, dtype=np.float32)
        direction[0] = 1.0  # direction along first embedding dim

        feature_names = [f"feat_{i}" for i in range(n_feat)]
        model = MarketAutoencoder(n_feat, AutoencoderConfig(latent_dim=latent_dim))
        interp = FeatureInterpreter(model, feature_names)

        attr = interp.attribute_direction_correlation(direction, embeddings, raw_features)
        assert abs(attr["feat_0"]) > 0.5, f"Expected high attribution for feat_0, got {attr['feat_0']}"

    def test_interpret_probe(self):
        rng = np.random.RandomState(42)
        n, latent_dim, n_feat = 50, 8, 5

        embeddings = rng.randn(n, latent_dim).astype(np.float32)
        raw_features = rng.randn(n, n_feat).astype(np.float32)
        condition_ids = [f"cid_{i}" for i in range(n)]
        feature_names = [f"feat_{i}" for i in range(n_feat)]

        probe_result = ProbeResult(
            "test", "classification", 0.8, 0.05, 0.5, 0.01, True,
            rng.randn(latent_dim).astype(np.float32), [0.8]
        )

        model = MarketAutoencoder(n_feat, AutoencoderConfig(latent_dim=latent_dim))
        interp = FeatureInterpreter(model, feature_names)
        result = interp.interpret_probe(probe_result, embeddings, raw_features, condition_ids)

        assert result.name == "test"
        assert len(result.input_attributions) == n_feat
        assert len(result.top_positive_markets) <= 5
        assert len(result.top_negative_markets) <= 5

    def test_generate_report(self):
        rng = np.random.RandomState(42)
        feature_names = ["feat_0", "feat_1"]
        model = MarketAutoencoder(2, AutoencoderConfig(latent_dim=4))
        interp = FeatureInterpreter(model, feature_names)

        from network.embedding.interpret import InterpretedDirection
        directions = [
            InterpretedDirection(
                name="test",
                description="Test direction",
                direction_vector=np.array([1, 0, 0, 0]),
                input_attributions={"feat_0": 0.9, "feat_1": 0.1},
                top_positive_markets=["cid_1"],
                top_negative_markets=["cid_2"],
                probe_accuracy=0.85,
            )
        ]
        report = interp.generate_report(directions)
        assert "test" in report
        assert "feat_0" in report


# ============================================================
# Collate function test
# ============================================================

class TestCollate:

    def test_collate_batch(self):
        batch = [
            {
                "features": torch.randn(10),
                "raw_features": torch.randn(10),
                "labels": {"category": "politics", "outcome_binary": 1},
                "condition_id": "cid_0",
                "question": "Q0",
            },
            {
                "features": torch.randn(10),
                "raw_features": torch.randn(10),
                "labels": {"category": "sports", "outcome_binary": 0},
                "condition_id": "cid_1",
                "question": "Q1",
            },
        ]
        collated = collate_embedding_batch(batch)
        assert collated["features"].shape == (2, 10)
        assert collated["raw_features"].shape == (2, 10)
        assert len(collated["labels"]["category"]) == 2
        assert len(collated["condition_ids"]) == 2


# ============================================================
# Config tests
# ============================================================

class TestConfig:

    def test_default_config(self):
        cfg = EmbeddingConfig()
        assert cfg.autoencoder.latent_dim == 64
        assert cfg.features.n_features == 27
        assert cfg.probe.cv_folds == 5

    def test_feature_config_defaults(self):
        fc = EmbeddingFeatureConfig()
        assert fc.min_volume_total == 1000.0
        assert fc.lifetime_cutoff_ratio == 0.8
        assert fc.include_price is True

    def test_ae_config_split_sums_to_one(self):
        ac = AutoencoderConfig()
        total = ac.train_ratio + ac.val_ratio + ac.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, expected 1.0"
