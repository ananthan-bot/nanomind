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
