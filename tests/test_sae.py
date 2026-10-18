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
