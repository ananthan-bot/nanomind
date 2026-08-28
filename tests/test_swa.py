"""
tests/test_swa.py — Tests for Sliding Window Attention (SWA).
"""

import pytest
import torch

from nanomind.attention.swa import SlidingWindowAttention, build_sliding_window_mask
from nanomind.attention.swa_rope import SWARoPEAttention
from nanomind.attention.complexity import attention_memory_bytes
from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.model import NanoMind, ModelConfig

B, T, D, H = 2, 16, 64, 4
W = 4    # small window for tests


# ── build_sliding_window_mask ─────────────────────────────────────────────────

class TestBuildSlidingWindowMask:
    def test_output_shape(self):
        mask = build_sliding_window_mask(T, W)
        assert mask.shape == (T, T)

    def test_dtype_is_bool(self):
        mask = build_sliding_window_mask(T, W)
        assert mask.dtype == torch.bool

    def test_causal_upper_triangle_is_false(self):
        mask = build_sliding_window_mask(T, W)
        # Upper triangle (future tokens) must all be False
        for i in range(T):
            for j in range(i + 1, T):
                assert not mask[i, j].item(), f"mask[{i},{j}] should be False (future)"

    def test_diagonal_is_true(self):
        mask = build_sliding_window_mask(T, W)
        for i in range(T):
            assert mask[i, i].item(), f"mask[{i},{i}] should be True (self)"

    def test_beyond_window_is_false(self):
        mask = build_sliding_window_mask(T, W)
        for i in range(T):
            for j in range(max(0, i - W), i):
                assert mask[i, j].item(), f"mask[{i},{j}] in window — should be True"
            for j in range(0, max(0, i - W)):
                assert not mask[i, j].item(), f"mask[{i},{j}] beyond window — should be False"

    def test_full_window_equals_causal(self):
        mask_full   = build_sliding_window_mask(T, T)
        mask_causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        assert torch.equal(mask_full, mask_causal)

    def test_window_1_is_diagonal(self):
        mask = build_sliding_window_mask(T, 1)
        expected = torch.eye(T, dtype=torch.bool)
        assert torch.equal(mask, expected)


# ── SlidingWindowAttention ────────────────────────────────────────────────────

class TestSlidingWindowAttention:
    def test_output_shape(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, wts = attn(x)
        assert out.shape == (B, T, D)
        assert wts.shape == (B, H, T, T)

    def test_single_token(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)

    def test_weights_outside_window_are_zero(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        # Attention weights outside window should be exactly 0
        for i in range(T):
            for j in range(0, max(0, i - W)):
                assert wts[0, :, i, j].abs().max().item() < 1e-6,                     f"weights[{i},{j}] should be 0 (beyond window)"

    def test_future_weights_are_zero(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        for i in range(T):
            for j in range(i + 1, T):
                assert wts[0, :, i, j].abs().max().item() < 1e-6,                     f"weights[{i},{j}] should be 0 (future)"

    def test_causal_invariance(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_window_clips_to_block_size(self):
        attn = SlidingWindowAttention(D, H, T, window_size=T * 10)
        assert attn.window_size == T


# ── SWARoPEAttention ──────────────────────────────────────────────────────────

class TestSWARoPEAttention:
    def test_output_shape(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, wts = attn(x)
        assert out.shape == (B, T, D)

    def test_has_rope_module(self):
        from nanomind.pos.rope import RotaryEmbedding
        attn = SWARoPEAttention(D, H, T, window_size=W)
        assert isinstance(attn.rope, RotaryEmbedding)

    def test_causal(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_window_respected(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        for i in range(T):
            for j in range(0, max(0, i - W)):
                assert wts[0, :, i, j].abs().max().item() < 1e-6
