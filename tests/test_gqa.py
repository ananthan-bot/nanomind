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
