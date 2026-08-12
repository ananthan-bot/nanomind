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


# ── Causal mask ───────────────────────────────────────────────────────────────

class TestCausalMask:
    def test_shape(self):
        mask = make_causal_mask(8, torch.device("cpu"))
        assert mask.shape == (1, 1, 8, 8)

    def test_lower_triangle_is_false(self):
        mask = make_causal_mask(4, torch.device("cpu")).squeeze()
        # Lower triangle (and diagonal) should be False (allowed to attend)
        for i in range(4):
            for j in range(i + 1):
                assert not mask[i, j].item(), f"Position ({i},{j}) should not be masked"

    def test_upper_triangle_is_true(self):
        mask = make_causal_mask(4, torch.device("cpu")).squeeze()
        # Upper triangle should be True (masked out)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j].item(), f"Position ({i},{j}) should be masked"

    def test_attention_is_causal(self, attn):
        # Token 0 should not be influenced by token 1 (future)
        x = torch.zeros(1, T, D)
        x[0, 0] = 1.0   # only token 0 is non-zero
        x[0, 1] = 2.0   # future token
        out_full, _ = attn(x)
        x2 = x.clone(); x2[0, 1] = 999.0   # change future token drastically
        out_changed, _ = attn(x2)
        # Token 0 output should be identical regardless of future tokens
        assert torch.allclose(out_full[0, 0], out_changed[0, 0], atol=1e-5)


# ── Head splitting / merging ──────────────────────────────────────────────────

class TestHeadSplitMerge:
    def test_split_shape(self, attn):
        x = torch.randn(B, T, D)
        split = attn._split_heads(x)
        assert split.shape == (B, H, T, D // H)

    def test_merge_shape(self, attn):
        x = torch.randn(B, H, T, D // H)
        merged = attn._merge_heads(x)
        assert merged.shape == (B, T, D)

    def test_split_merge_roundtrip(self, attn):
        x = torch.randn(B, T, D)
        roundtrip = attn._merge_heads(attn._split_heads(x))
        assert torch.allclose(x, roundtrip)
