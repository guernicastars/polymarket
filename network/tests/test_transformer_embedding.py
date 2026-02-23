"""Tests for the transformer embedding module.

Tests cover:
  - PatchEmbedding: correct shapes, truncation to patch boundary
  - RelativePositionalEncoding: shape, learnable vs continuous modes
  - PreNormEncoderLayer: residual connection, shape preservation
  - MarketTransformer: full forward pass, CLS extraction, padding mask
  - MarketTransformerForPretraining: masking, loss computation, gradient flow
  - TransformerConfig: default values, override propagation
  - collate_temporal_batch: padding, variable-length batching
  - Training convergence: loss decreases on synthetic data
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from network.embedding.config import TransformerConfig, EmbeddingConfig
from network.embedding.transformer_model import (
    PatchEmbedding,
    RelativePositionalEncoding,
    PreNormEncoderLayer,
    MarketTransformer,
    MaskedPatchPredictionHead,
    MarketTransformerForPretraining,
    CrossAttentionFusion,
)
from network.embedding.probes import LinearProbe


# ============================================================
# Patch Embedding
# ============================================================

class TestPatchEmbedding:
    def setup_method(self):
        self.patch_size = 24
        self.n_features = 12
        self.d_model = 64
        self.embed = PatchEmbedding(self.patch_size, self.n_features, self.d_model)

    def test_output_shape(self):
        """Patch embedding produces correct (B, n_patches, d_model) shape."""
        x = torch.randn(4, 720, 12)  # 30 days of hourly data
        out = self.embed(x)
        assert out.shape == (4, 30, 64), f"Expected (4, 30, 64), got {out.shape}"

    def test_truncation(self):
        """Input not divisible by patch_size is truncated, not padded."""
        x = torch.randn(2, 735, 12)  # 735 / 24 = 30.625 → 30 patches
        out = self.embed(x)
        assert out.shape == (2, 30, 64)

    def test_single_patch(self):
        """Minimum viable sequence: exactly one patch."""
        x = torch.randn(1, 24, 12)
        out = self.embed(x)
        assert out.shape == (1, 1, 64)

    def test_gradient_flow(self):
        """Gradients flow through patch embedding to projection weights."""
        x = torch.randn(2, 48, 12)
        out = self.embed(x)
        # Use pow(2).sum() — plain sum() yields zero grads through LayerNorm
        loss = out.pow(2).sum()
        loss.backward()
        assert self.embed.proj.weight.grad is not None
        assert self.embed.proj.weight.grad.abs().sum() > 0


# ============================================================
# Positional Encoding
# ============================================================

class TestRelativePositionalEncoding:
    def setup_method(self):
        self.pe = RelativePositionalEncoding(d_model=64, max_patches=128)

    def test_fallback_shape(self):
        """Learnable position embedding has correct shape."""
        out = self.pe(30)
        assert out.shape == (1, 30, 64)

    def test_relative_shape(self):
        """Continuous relative position encoding has correct shape."""
        rel_pos = torch.linspace(0, 1, 30).unsqueeze(0).expand(4, -1)
        out = self.pe(30, rel_pos)
        assert out.shape == (4, 30, 64)

    def test_different_positions_different_encoding(self):
        """Different relative positions produce different encodings."""
        pos_a = torch.tensor([[0.0, 0.5, 1.0]])
        pos_b = torch.tensor([[0.1, 0.6, 0.9]])
        enc_a = self.pe(3, pos_a)
        enc_b = self.pe(3, pos_b)
        assert not torch.allclose(enc_a, enc_b)


# ============================================================
# Pre-Norm Encoder Layer
# ============================================================

class TestPreNormEncoderLayer:
    def setup_method(self):
        self.layer = PreNormEncoderLayer(d_model=64, n_heads=4, d_ff=256)

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        x = torch.randn(2, 30, 64)
        out = self.layer(x)
        assert out.shape == x.shape

    def test_with_padding_mask(self):
        """Works correctly with key_padding_mask."""
        x = torch.randn(2, 30, 64)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[0, 25:] = True  # pad last 5 positions for first sample
        out = self.layer(x, key_padding_mask=mask)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output is different from input (not identity) but not completely unrelated."""
        x = torch.randn(2, 10, 64)
        out = self.layer(x)
        # Residual: output should be correlated with input
        cosine_sim = F.cosine_similarity(x.flatten(), out.flatten(), dim=0)
        assert cosine_sim > 0.1, "Residual connection seems broken"


# ============================================================
# Full MarketTransformer
# ============================================================

class TestMarketTransformer:
    def setup_method(self):
        self.cfg = TransformerConfig(
            patch_size=24, n_input_features=12, d_model=64,
            n_heads=4, n_layers=2, d_ff=128, max_patches=64,
        )
        self.model = MarketTransformer(self.cfg)

    def test_forward_shape(self):
        """Full forward pass produces correct embedding and patch shapes."""
        x = torch.randn(4, 720, 12)  # 30 days
        embedding, patch_emb = self.model(x)
        assert embedding.shape == (4, 64), f"Embedding: {embedding.shape}"
        assert patch_emb.shape == (4, 30, 64), f"Patches: {patch_emb.shape}"

    def test_encode_shortcut(self):
        """encode() returns only the CLS embedding."""
        x = torch.randn(2, 240, 12)  # 10 days
        emb = self.model.encode(x)
        assert emb.shape == (2, 64)

    def test_variable_length(self):
        """Different sequence lengths produce same embedding dimension."""
        x_short = torch.randn(1, 48, 12)   # 2 days
        x_long = torch.randn(1, 720, 12)   # 30 days
        emb_short = self.model.encode(x_short)
        emb_long = self.model.encode(x_long)
        assert emb_short.shape == emb_long.shape == (1, 64)

    def test_with_padding_mask(self):
        """Padding mask is correctly converted from bar-level to patch-level."""
        x = torch.randn(2, 720, 12)
        mask = torch.zeros(2, 720, dtype=torch.bool)
        mask[1, 480:] = True  # pad last 10 days of second sample
        embedding, _ = self.model(x, padding_mask=mask)
        assert embedding.shape == (2, 64)

    def test_with_relative_positions(self):
        """Relative positional encoding is applied correctly."""
        x = torch.randn(2, 240, 12)  # 10 patches
        rel_pos = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        embedding, _ = self.model(x, relative_positions=rel_pos)
        assert embedding.shape == (2, 64)

    def test_deterministic_eval(self):
        """Model produces same output in eval mode with same input."""
        self.model.eval()
        x = torch.randn(2, 240, 12)
        emb1 = self.model.encode(x)
        emb2 = self.model.encode(x)
        assert torch.allclose(emb1, emb2)

    def test_gradient_flow_to_parameters(self):
        """Gradients flow through encoder layers to key parameters."""
        x = torch.randn(2, 240, 12)
        emb, patches = self.model(x)
        loss = emb.pow(2).sum() + patches.pow(2).sum()
        loss.backward()
        # mask_token: only used in MPP, not in regular forward
        # pos_encoder.rel_proj: only used when relative_positions are provided
        skip = {"mask_token", "rel_proj"}
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(s in name for s in skip):
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_parameter_count(self):
        """Model has a reasonable number of parameters for its config."""
        n_params = sum(p.numel() for p in self.model.parameters())
        # 2 layers, d_model=64, d_ff=128: should be ~50-200K params
        assert 10_000 < n_params < 500_000, f"Unexpected param count: {n_params}"


# ============================================================
# Masked Patch Prediction
# ============================================================

class TestMaskedPatchPrediction:
    def setup_method(self):
        self.cfg = TransformerConfig(
            patch_size=24, n_input_features=12, d_model=32,
            n_heads=4, n_layers=2, d_ff=64, mask_ratio=0.30,
        )
        self.model = MarketTransformerForPretraining(self.cfg)

    def test_forward_returns_loss(self):
        """Pre-training forward pass returns scalar loss."""
        x = torch.randn(4, 240, 12)  # 10 patches
        loss, embedding, predictions = self.model(x)
        assert loss.shape == ()
        assert loss.item() > 0
        assert embedding.shape == (4, 32)

    def test_mask_ratio_affects_n_masked(self):
        """Higher mask ratio masks more patches."""
        x = torch.randn(4, 240, 12)  # 10 patches
        loss_low, _, pred_low = self.model(x, mask_ratio=0.1)
        loss_high, _, pred_high = self.model(x, mask_ratio=0.5)
        # 10% of 10 patches = 1, 50% = 5
        assert pred_low.shape[1] == 1
        assert pred_high.shape[1] == 5

    def test_prediction_shape(self):
        """Predictions have shape (B, n_masked, patch_size * n_features)."""
        x = torch.randn(2, 240, 12)  # 10 patches, mask 30% → 3 patches
        loss, emb, preds = self.model(x)
        expected_n_mask = max(1, int(10 * 0.30))  # 3
        assert preds.shape == (2, expected_n_mask, 24 * 12)

    def test_gradient_flow(self):
        """Gradients flow through the full pre-training pipeline."""
        x = torch.randn(2, 240, 12, requires_grad=True)
        loss, _, _ = self.model(x)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_loss_decreases(self):
        """Loss decreases with gradient steps (basic training convergence)."""
        torch.manual_seed(42)
        model = MarketTransformerForPretraining(self.cfg)
        x = torch.randn(16, 120, 12)  # 5 patches per sample, larger batch
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            loss, _, _ = model(x, mask_ratio=0.3)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_padding_mask_respected(self):
        """Masked prediction doesn't select padding patches for masking."""
        x = torch.randn(2, 240, 12)
        mask = torch.zeros(2, 240, dtype=torch.bool)
        mask[0, 120:] = True  # first sample: only 5 real patches, 5 padded
        loss, emb, preds = self.model(x, padding_mask=mask, mask_ratio=0.3)
        assert loss.item() > 0  # should still compute a valid loss


# ============================================================
# Config
# ============================================================

class TestTransformerConfig:
    def test_defaults(self):
        """Default config has sane values."""
        cfg = TransformerConfig()
        assert cfg.patch_size == 24
        assert cfg.d_model == 64
        assert cfg.n_heads == 4
        assert cfg.n_layers == 3
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.mask_ratio > 0 and cfg.mask_ratio < 1

    def test_embedding_config_includes_transformer(self):
        """Top-level EmbeddingConfig includes transformer sub-config."""
        cfg = EmbeddingConfig()
        assert hasattr(cfg, "transformer")
        assert isinstance(cfg.transformer, TransformerConfig)

    def test_d_model_divisible_by_heads(self):
        """d_model must be divisible by n_heads for attention."""
        cfg = TransformerConfig(d_model=64, n_heads=4)
        assert cfg.d_model % cfg.n_heads == 0


# ============================================================
# Collation
# ============================================================

class TestCollation:
    def test_variable_length_collation(self):
        """collate_temporal_batch handles variable-length sequences."""
        from network.embedding.temporal_dataset import collate_temporal_batch

        batch = [
            {
                "features": torch.randn(120, 12),  # 5 patches
                "padding_mask": torch.zeros(120, dtype=torch.bool),
                "relative_positions": torch.linspace(0, 1, 5),
                "condition_id": "abc",
                "labels": {"outcome_binary": 1},
            },
            {
                "features": torch.randn(240, 12),  # 10 patches
                "padding_mask": torch.zeros(240, dtype=torch.bool),
                "relative_positions": torch.linspace(0, 1, 10),
                "condition_id": "def",
                "labels": {"outcome_binary": 0},
            },
        ]

        collated = collate_temporal_batch(batch)
        assert collated["features"].shape[0] == 2
        assert collated["features"].shape[1] >= 240  # padded to longest
        assert collated["features"].shape[2] == 12
        assert collated["padding_mask"].shape == collated["features"].shape[:2]
        # First sample should have padding (shorter sequence)
        assert collated["padding_mask"][0, 120:].any()
        # Second sample should have no padding
        assert not collated["padding_mask"][1, :240].any()

    def test_collated_through_model(self):
        """Collated batch runs through the transformer without errors."""
        from network.embedding.temporal_dataset import collate_temporal_batch

        cfg = TransformerConfig(
            patch_size=24, d_model=32, n_heads=4, n_layers=1, d_ff=64,
        )
        model = MarketTransformer(cfg)

        batch = [
            {
                "features": torch.randn(72, 12),   # 3 patches
                "padding_mask": torch.zeros(72, dtype=torch.bool),
                "relative_positions": torch.linspace(0, 1, 3),
                "condition_id": "a",
                "labels": {},
            },
            {
                "features": torch.randn(120, 12),  # 5 patches
                "padding_mask": torch.zeros(120, dtype=torch.bool),
                "relative_positions": torch.linspace(0, 1, 5),
                "condition_id": "b",
                "labels": {},
            },
        ]

        collated = collate_temporal_batch(batch)
        emb, patches = model(
            collated["features"],
            padding_mask=collated["padding_mask"],
            relative_positions=collated["relative_positions"],
        )
        assert emb.shape == (2, 32)


# ============================================================
# Cross-Attention Fusion
# ============================================================

class TestCrossAttentionFusion:
    def setup_method(self):
        self.fusion = CrossAttentionFusion(
            d_ae=64, d_tf=64, d_fused=64, n_heads=4, dropout=0.1,
        )

    def test_output_shape(self):
        """Fusion produces correct (B, d_fused) shape."""
        ae = torch.randn(4, 64)
        tf = torch.randn(4, 64)
        fused = self.fusion(ae, tf)
        assert fused.shape == (4, 64)

    def test_different_input_dims(self):
        """Fusion works with different AE and TF dimensions."""
        fusion = CrossAttentionFusion(d_ae=32, d_tf=64, d_fused=48, n_heads=4)
        ae = torch.randn(4, 32)
        tf = torch.randn(4, 64)
        fused = fusion(ae, tf)
        assert fused.shape == (4, 48)

    def test_gradient_flow(self):
        """Gradients flow through both branches."""
        ae = torch.randn(4, 64, requires_grad=True)
        tf = torch.randn(4, 64, requires_grad=True)
        fused = self.fusion(ae, tf)
        loss = fused.pow(2).sum()
        loss.backward()
        assert ae.grad is not None and ae.grad.abs().sum() > 0
        assert tf.grad is not None and tf.grad.abs().sum() > 0

    def test_gate_bounded(self):
        """Gate values are in [0, 1] (sigmoid output)."""
        ae = torch.randn(8, 64)
        tf = torch.randn(8, 64)
        # Access gate directly
        ae_proj = self.fusion.proj_ae(ae).unsqueeze(1)
        tf_proj = self.fusion.proj_tf(tf).unsqueeze(1)
        gate_input = torch.cat([ae_proj.squeeze(1), tf_proj.squeeze(1)], dim=-1)
        g = self.fusion.gate(gate_input)
        assert g.min() >= 0.0 and g.max() <= 1.0

    def test_deterministic_eval(self):
        """Same input produces same output in eval mode."""
        self.fusion.eval()
        ae = torch.randn(2, 64)
        tf = torch.randn(2, 64)
        out1 = self.fusion(ae, tf)
        out2 = self.fusion(ae, tf)
        assert torch.allclose(out1, out2)

    def test_parameter_count(self):
        """Fusion model has reasonable parameter count."""
        n_params = sum(p.numel() for p in self.fusion.parameters())
        # Projections + attention + gate + norms: should be ~30K-100K
        assert 5_000 < n_params < 200_000, f"Unexpected param count: {n_params}"

    def test_single_sample(self):
        """Works with batch size 1."""
        ae = torch.randn(1, 64)
        tf = torch.randn(1, 64)
        fused = self.fusion(ae, tf)
        assert fused.shape == (1, 64)


# ============================================================
# Temporal Probes
# ============================================================

class TestTemporalProbes:
    def test_trajectory_monotonic_up(self):
        """Positive cumulative returns classified as monotonic_up."""
        # Steady upward trend
        seq = np.zeros((100, 12))
        seq[:, 0] = 0.01  # constant positive log returns
        label = LinearProbe._classify_trajectory(seq)
        assert label == 0, f"Expected 0 (monotonic_up), got {label}"

    def test_trajectory_monotonic_down(self):
        """Negative cumulative returns classified as monotonic_down."""
        seq = np.zeros((100, 12))
        seq[:, 0] = -0.01
        label = LinearProbe._classify_trajectory(seq)
        assert label == 1, f"Expected 1 (monotonic_down), got {label}"

    def test_trajectory_mean_reverting(self):
        """Oscillating returns near zero classified as mean_reverting."""
        seq = np.zeros((100, 12))
        # Oscillate: up then down repeatedly, ending near zero
        for i in range(100):
            seq[i, 0] = 0.05 * (1 if i % 2 == 0 else -1)
        label = LinearProbe._classify_trajectory(seq)
        assert label == 3, f"Expected 3 (mean_reverting), got {label}"

    def test_trajectory_short_sequence(self):
        """Short sequences return -1 (invalid)."""
        seq = np.zeros((5, 12))
        label = LinearProbe._classify_trajectory(seq)
        assert label == -1

    def test_momentum_accelerating(self):
        """Larger returns in second half classified as accelerating."""
        seq = np.zeros((100, 12))
        seq[:50, 0] = 0.01   # small moves first half
        seq[50:, 0] = 0.05   # large moves second half
        label = LinearProbe._classify_momentum(seq)
        assert label == 0, f"Expected 0 (accelerating), got {label}"

    def test_momentum_decelerating(self):
        """Larger returns in first half classified as decelerating."""
        seq = np.zeros((100, 12))
        seq[:50, 0] = 0.05
        seq[50:, 0] = 0.01
        label = LinearProbe._classify_momentum(seq)
        assert label == 1, f"Expected 1 (decelerating), got {label}"

    def test_momentum_steady(self):
        """Equal returns both halves classified as steady."""
        seq = np.zeros((100, 12))
        seq[:, 0] = 0.02
        label = LinearProbe._classify_momentum(seq)
        assert label == 2, f"Expected 2 (steady), got {label}"

    def test_volume_front_loaded(self):
        """Higher volume delta in first third classified as front_loaded."""
        seq = np.zeros((90, 12))
        seq[:30, 6] = 5.0   # high volume delta first third
        seq[60:, 6] = 0.5   # low volume delta last third
        label = LinearProbe._classify_volume_pattern(seq)
        assert label == 0, f"Expected 0 (front_loaded), got {label}"

    def test_volume_back_loaded(self):
        """Higher volume delta in last third classified as back_loaded."""
        seq = np.zeros((90, 12))
        seq[:30, 6] = 0.5
        seq[60:, 6] = 5.0
        label = LinearProbe._classify_volume_pattern(seq)
        assert label == 1, f"Expected 1 (back_loaded), got {label}"

    def test_volume_uniform(self):
        """Equal volume across thirds classified as uniform."""
        seq = np.zeros((90, 12))
        seq[:, 6] = 1.0
        label = LinearProbe._classify_volume_pattern(seq)
        assert label == 2, f"Expected 2 (uniform), got {label}"

    def test_run_temporal_probes_synthetic(self):
        """Temporal probes run on synthetic separable data."""
        from network.embedding.config import ProbeConfig

        n = 200
        d = 32
        rng = np.random.RandomState(42)
        embeddings = rng.randn(n, d).astype(np.float32)

        # Create sequences with clear trajectory patterns embedded in features
        sequences = []
        for i in range(n):
            length = 100
            seq = np.zeros((length, 12))
            if i < n // 2:
                seq[:, 0] = 0.01  # monotonic up
            else:
                seq[:, 0] = -0.01  # monotonic down
            # Also make embeddings separable along the same axis
            embeddings[i, 0] = 1.0 if i < n // 2 else -1.0
            sequences.append(seq)

        labels = [{"dummy": True}] * n

        probe = LinearProbe(ProbeConfig(n_permutation_tests=10))
        results = probe.run_temporal_probes(embeddings, sequences, labels)
        assert len(results) > 0
        # trajectory_shape should be trivially separable
        traj_result = [r for r in results if r.concept_name == "trajectory_shape"]
        assert len(traj_result) == 1
        assert traj_result[0].accuracy > 0.8
