"""
tests/test_attention.py — Tests for the NanoMind attention mechanism.
"""

import pytest
import torch

from nanomind.attention import (
    CausalSelfAttention,
    KVCache,
    AttentionConfig,
    make_causal_mask,
    scaled_dot_product_attention,
)

B, T, D, H = 2, 16, 64, 4   # batch, seq_len, d_model, n_heads


@pytest.fixture
def attn() -> CausalSelfAttention:
    return CausalSelfAttention(d_model=D, n_heads=H, block_size=T, dropout=0.0)


# ── Output shapes ─────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_forward_output_shape(self, attn):
        x = torch.randn(B, T, D)
        out, weights = attn(x)
        assert out.shape == (B, T, D)

    def test_attention_weights_shape(self, attn):
        x = torch.randn(B, T, D)
        _, weights = attn(x)
        assert weights.shape == (B, H, T, T)

    def test_single_token(self, attn):
        x = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)

    def test_full_block_size(self, attn):
        x = torch.randn(B, T, D)
        out, _ = attn(x)
        assert out.shape == (B, T, D)
