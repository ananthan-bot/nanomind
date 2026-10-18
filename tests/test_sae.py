"""
tests/test_sae.py — Comprehensive unit tests for Sparse Autoencoders and Dictionary Learning.
"""
import torch
from nanomind.sae import (
    SAEConfig,
    SAEType,
    SAEOutput,
    VanillaSAE,
    TopKSAE,
    JumpReLUSAE,
    GatedSAE,
    DeadNeuronTracker,
    SAETrainer,
    FeatureSteerer,
    FeatureAttribution,
    SAEBenchmark,
)


class TestSAEConfig:
    def test_config_creation(self):
        cfg = SAEConfig(d_in=32, d_sae=128, k_sparse=16)
        assert cfg.d_in == 32
        assert cfg.d_sae == 128
        assert cfg.k_sparse == 16
        assert cfg.sae_type == SAEType.TOPK


class TestVanillaSAE:
    def test_forward_and_reconstruction(self):
        cfg = SAEConfig(d_in=16, d_sae=64, l1_coeff=1e-3, normalize_decoder=True)
        sae = VanillaSAE(cfg)
        x = torch.randn(8, 16)
        out = sae(x)

        assert out.reconstruction.shape == (8, 16)
        assert out.latents.shape == (8, 64)
        assert out.loss > 0

        # Check unit norm constraint on decoder
        col_norms = sae.w_dec.weight.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4)


class TestTopKSAE:
    def test_exact_k_sparsity(self):
        cfg = SAEConfig(d_in=24, d_sae=96, k_sparse=6, sae_type=SAEType.TOPK)
        sae = TopKSAE(cfg)
        x = torch.randn(10, 24)
        out = sae(x)

        assert out.reconstruction.shape == (10, 24)
        # Exactly k elements should be > 0 per sample
        non_zero_per_row = (out.latents > 0).sum(dim=-1)
        assert (non_zero_per_row == 6).all()
        assert out.l0 == 6.0


class TestJumpReLUSAE:
    def test_jumprelu_thresholding(self):
        cfg = SAEConfig(d_in=16, d_sae=32, threshold=0.2, sae_type=SAEType.JUMPRELU)
        sae = JumpReLUSAE(cfg)
        x = torch.randn(5, 16)
        out = sae(x)

        assert out.reconstruction.shape == (5, 16)
        # All non-zero latents must be greater than threshold
        active_vals = out.latents[out.latents > 0]
        if len(active_vals) > 0:
            assert (active_vals >= 0.2).all()


class TestGatedSAE:
    def test_gated_dual_path(self):
        cfg = SAEConfig(d_in=16, d_sae=48, sae_type=SAEType.GATED)
        sae = GatedSAE(cfg)
        x = torch.randn(6, 16)
        out = sae(x)

        assert out.reconstruction.shape == (6, 16)
        assert out.latents.shape == (6, 48)
        assert out.loss > 0


class TestFeatureSteererAndBenchmark:
    def test_steer_and_clamp(self):
        cfg = SAEConfig(d_in=16, d_sae=32, k_sparse=4)
        sae = TopKSAE(cfg)
        steerer = FeatureSteerer(sae)
        x = torch.randn(4, 16)

        steered = steerer.steer_activation(x, feature_idx=3, coefficient=2.0)
        assert steered.shape == (4, 16)
        assert not torch.allclose(steered, x)

        clamped = steerer.clamp_feature(x, feature_idx=3, target_value=5.0)
        assert clamped.shape == (4, 16)

    def test_sae_benchmark_metrics(self):
        cfg = SAEConfig(d_in=16, d_sae=32, k_sparse=4)
        sae = TopKSAE(cfg)
        x = torch.randn(20, 16)
        bench = SAEBenchmark.evaluate(sae, x)

        assert "nmse" in bench
        assert "explained_variance" in bench
        assert "l0_sparsity" in bench
        assert "dead_feature_ratio" in bench
        assert bench["l0_sparsity"] == 4.0
