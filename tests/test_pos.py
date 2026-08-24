"""
tests/test_pos.py — Tests for positional embedding strategies.
"""

import math
import pytest
import torch

from nanomind.pos import (
    rotate_half,
    precompute_rope_freqs,
    apply_rotary_emb,
    RotaryEmbedding,
    build_alibi_bias,
    get_alibi_slopes,
    RoPECausalSelfAttention,
    ALiBiCausalSelfAttention,
    get_attention,
    list_pos_types,
)

B, T, D, H = 2, 16, 64, 4
HEAD_DIM = D // H


# ── rotate_half ───────────────────────────────────────────────────────────────

class TestRotateHalf:
    def test_output_shape(self):
        x = torch.randn(B, H, T, HEAD_DIM)
        assert rotate_half(x).shape == x.shape

    def test_double_rotation_is_neg_identity(self):
        x = torch.randn(4, 8)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)

    def test_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = rotate_half(x)
        # [-x2, x1] = [-3, -4, 1, 2]
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(out, expected)


# ── precompute_rope_freqs ─────────────────────────────────────────────────────

class TestPrecomputeRopeFreqs:
    def test_output_shapes(self):
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        assert cos.shape == (T, HEAD_DIM)
        assert sin.shape == (T, HEAD_DIM)

    def test_first_position_cos_is_one(self):
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        # At position 0, freqs = 0, so cos(0) = 1, sin(0) = 0
        assert torch.allclose(cos[0], torch.ones(HEAD_DIM), atol=1e-5)
        assert torch.allclose(sin[0], torch.zeros(HEAD_DIM), atol=1e-5)


# ── apply_rotary_emb ──────────────────────────────────────────────────────────

class TestApplyRotaryEmb:
    def test_output_shapes(self):
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_magnitude_preserved(self):
        """RoPE is a rotation so it preserves vector magnitude."""
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)
