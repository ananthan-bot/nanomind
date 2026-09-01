"""
tests/test_moe.py — Tests for Mixture of Experts (MoE).
"""

import pytest
import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.moe import (
    MoEConfig, Expert, TopKRouter, SparseMoELayer,
    MoETransformerBlock, NanoMindMoE,
    load_balance_loss, expert_utilization,
)

B, T, D = 2, 8, 64
N_EXP, TOP_K = 4, 2


def tiny_moe_model(num_experts=N_EXP, top_k=TOP_K):
    torch.manual_seed(0)
    cfg     = ModelConfig(vocab_size=32, block_size=T, d_model=D,
                          n_layers=2, n_heads=4, dropout=0.0)
    moe_cfg = MoEConfig(num_experts=num_experts, top_k=top_k,
                        load_balance_coef=0.01)
    return NanoMindMoE(cfg, moe_cfg)


# ── MoEConfig ─────────────────────────────────────────────────────────────────

class TestMoEConfig:
    def test_defaults(self):
        cfg = MoEConfig()
        assert cfg.num_experts == 8
        assert cfg.top_k == 2

    def test_invalid_num_experts(self):
        with pytest.raises(AssertionError):
            MoEConfig(num_experts=0)

    def test_top_k_exceeds_experts(self):
        with pytest.raises(AssertionError):
            MoEConfig(num_experts=4, top_k=5)

    def test_invalid_activation(self):
        with pytest.raises(AssertionError):
            MoEConfig(activation="tanh")

    def test_negative_load_balance(self):
        with pytest.raises(AssertionError):
            MoEConfig(load_balance_coef=-0.1)


# ── Expert ────────────────────────────────────────────────────────────────────

class TestExpert:
    def test_output_shape_gelu(self):
        exp = Expert(D, D * 4, activation="gelu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_output_shape_relu(self):
        exp = Expert(D, D * 4, activation="relu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_output_shape_swiglu(self):
        exp = Expert(D, D * 4, activation="swiglu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_3d_input(self):
        exp = Expert(D, D * 4)
        x   = torch.randn(B, T, D)
        assert exp(x).shape == (B, T, D)


# ── TopKRouter ────────────────────────────────────────────────────────────────

class TestTopKRouter:
    def test_output_shapes(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        indices, weights, logits = router(x)
        assert indices.shape == (B * T, TOP_K)
        assert weights.shape == (B * T, TOP_K)
        assert logits.shape  == (B * T, N_EXP)

    def test_indices_in_range(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        indices, _, _ = router(x)
        assert (indices >= 0).all()
        assert (indices < N_EXP).all()

    def test_weights_sum_to_one(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        _, weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_top_k_unique_per_token(self):
        router  = TopKRouter(D, N_EXP, TOP_K)
        x       = torch.randn(B, T, D)
        indices, _, _ = router(x)
        for row in indices:
            assert len(set(row.tolist())) == TOP_K
