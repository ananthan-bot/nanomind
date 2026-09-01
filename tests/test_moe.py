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


# ── SparseMoELayer ────────────────────────────────────────────────────────────

class TestSparseMoELayer:
    def test_output_shape(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.0)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        out, aux = moe(x)
        assert out.shape == (B, T, D)

    def test_aux_loss_zero_when_disabled(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.0)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        _, aux = moe(x)
        assert aux.item() == 0.0

    def test_aux_loss_positive_when_enabled(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.01)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        _, aux = moe(x)
        assert aux.item() >= 0.0

    def test_output_finite(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        out, _ = moe(x)
        assert out.isfinite().all()

    def test_n_experts_in_layer(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        moe = SparseMoELayer(D, cfg)
        assert len(moe.experts) == N_EXP


# ── load_balance_loss ─────────────────────────────────────────────────────────

class TestLoadBalanceLoss:
    def test_returns_scalar(self):
        logits  = torch.randn(16, N_EXP)
        indices = torch.randint(0, N_EXP, (16, TOP_K))
        loss    = load_balance_loss(logits, indices, N_EXP)
        assert loss.shape == ()

    def test_non_negative(self):
        logits  = torch.randn(32, N_EXP)
        indices = torch.randint(0, N_EXP, (32, TOP_K))
        loss    = load_balance_loss(logits, indices, N_EXP)
        assert loss.item() >= 0.0

    def test_balanced_routing_gives_low_loss(self):
        # Uniform routing: each expert gets exactly 1/N of tokens
        n_tokens = N_EXP * 4
        logits   = torch.zeros(n_tokens, N_EXP)
        # Assign tokens round-robin to ensure balance
        indices  = torch.tensor([[i % N_EXP, (i+1) % N_EXP]
                                  for i in range(n_tokens)])
        loss     = load_balance_loss(logits, indices, N_EXP)
        # Balanced loss should be close to 1.0 (= N × 1/N × 1/N × N = 1)
        assert loss.item() < 2.0

    def test_expert_utilization_keys(self):
        indices  = torch.randint(0, N_EXP, (32, TOP_K))
        stats    = expert_utilization(indices, N_EXP)
        for key in ("counts", "fractions", "min_frac", "max_frac", "utilization"):
            assert key in stats
