"""
tests/test_gqa.py — Tests for Grouped-Query Attention (GQA) and MQA.
"""

import pytest
import torch

from nanomind.attention.gqa import GroupedQueryAttention, MultiQueryAttention, repeat_kv
from nanomind.attention.gqa_rope import GQARoPEAttention
from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.model import NanoMind, ModelConfig

B, T, D = 2, 16, 64
N_HEADS = 8
N_KV    = 2
HEAD_DIM = D // N_HEADS


# ── repeat_kv ─────────────────────────────────────────────────────────────────

class TestRepeatKV:
    def test_n_rep_1_is_identity(self):
        x = torch.randn(B, N_KV, T, HEAD_DIM)
        assert torch.equal(repeat_kv(x, 1), x)

    def test_output_shape(self):
        x = torch.randn(B, N_KV, T, HEAD_DIM)
        out = repeat_kv(x, N_HEADS // N_KV)
        assert out.shape == (B, N_HEADS, T, HEAD_DIM)

    def test_repeated_values_equal(self):
        x   = torch.randn(B, N_KV, T, HEAD_DIM)
        n_rep = N_HEADS // N_KV
        out = repeat_kv(x, n_rep)
        for i in range(N_KV):
            for r in range(n_rep):
                assert torch.equal(out[:, i * n_rep + r], x[:, i])

    def test_single_kv_head(self):
        x   = torch.randn(B, 1, T, HEAD_DIM)
        out = repeat_kv(x, N_HEADS)
        assert out.shape == (B, N_HEADS, T, HEAD_DIM)
        # All heads should be identical
        for h in range(N_HEADS):
            assert torch.equal(out[:, h], x[:, 0])


# ── GroupedQueryAttention ─────────────────────────────────────────────────────

class TestGroupedQueryAttention:
    def test_output_shape(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x   = torch.randn(B, T, D)
        out, w = gqa(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, N_HEADS, T, T)

    def test_single_token(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x   = torch.randn(B, 1, D)
        out, _ = gqa(x)
        assert out.shape == (B, 1, D)

    def test_causal(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x1  = torch.randn(1, T, D)
        x2  = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = gqa(x1)
        out2, _ = gqa(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(D, N_HEADS, 3, T)   # 8 % 3 != 0

    def test_fewer_params_than_mha(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T)
        from nanomind.attention import CausalSelfAttention
        mha = CausalSelfAttention(D, N_HEADS, T)
        gqa_params = sum(p.numel() for p in gqa.parameters())
        mha_params = sum(p.numel() for p in mha.parameters())
        assert gqa_params < mha_params


# ── MultiQueryAttention ───────────────────────────────────────────────────────

class TestMultiQueryAttention:
    def test_output_shape(self):
        mqa = MultiQueryAttention(D, N_HEADS, T, dropout=0.0)
        x   = torch.randn(B, T, D)
        out, w = mqa(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, N_HEADS, T, T)

    def test_n_kv_heads_is_one(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert mqa.n_kv_heads == 1

    def test_kv_proj_size(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert mqa.k_proj.out_features == HEAD_DIM   # single KV head
        assert mqa.v_proj.out_features == HEAD_DIM

    def test_mqa_is_gqa_subclass(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert isinstance(mqa, GroupedQueryAttention)
