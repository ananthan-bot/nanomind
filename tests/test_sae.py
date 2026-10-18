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
