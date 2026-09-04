"""
tests/test_flash.py — Tests for Flash Attention.
"""

import pytest
import torch
import torch.nn.functional as F

from nanomind.flash import (
    FlashConfig, FlashAttention, FlashTransformerBlock, NanoMindFlash,
    tiled_flash_attention, OnlineSoftmaxState,
    standard_attention_memory, flash_attention_memory, memory_comparison_report,
)
from nanomind.model.config import ModelConfig

B, H, N, Dh = 2, 4, 32, 16
D = H * Dh


def make_qkv():
    torch.manual_seed(0)
    q = torch.randn(B, H, N, Dh)
    k = torch.randn(B, H, N, Dh)
    v = torch.randn(B, H, N, Dh)
    return q, k, v


# ── FlashConfig ───────────────────────────────────────────────────────────────

class TestFlashConfig:
    def test_defaults(self):
        cfg = FlashConfig()
        assert cfg.block_q  == 64
        assert cfg.causal   is True
        assert cfg.use_torch_sdpa is True

    def test_invalid_block_q(self):
        with pytest.raises(AssertionError):
            FlashConfig(block_q=0)

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            FlashConfig(dropout=1.5)


# ── OnlineSoftmaxState ────────────────────────────────────────────────────────

class TestOnlineSoftmax:
    def test_init_shapes(self):
        q   = torch.randn(B, H, N, Dh)
        st  = OnlineSoftmaxState(q)
        assert st.m.shape == (B, H, N, 1)
        assert st.l.shape == (B, H, N, 1)
        assert st.O.shape == (B, H, N, Dh)

    def test_single_tile_matches_softmax(self):
        """With one full tile, online softmax == standard softmax."""
        q, k, v = make_qkv()
        scale   = Dh ** -0.5
        s       = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B,H,N,N)

        st = OnlineSoftmaxState(q)
        st.update(s, v)
        online_out = st.finalize()

        # Standard attention output
        std_out = F.softmax(s, dim=-1) @ v
        assert torch.allclose(online_out, std_out, atol=1e-5)

    def test_two_tiles_matches_softmax(self):
        """Splitting K/V into two tiles should give the same result."""
        q, k, v = make_qkv()
        scale   = Dh ** -0.5

        # Standard
        s       = torch.matmul(q, k.transpose(-2, -1)) * scale
        std_out = F.softmax(s, dim=-1) @ v

        # Two-tile online
        half = N // 2
        st   = OnlineSoftmaxState(q)
        s1   = torch.matmul(q, k[:, :, :half].transpose(-2, -1)) * scale
        st.update(s1, v[:, :, :half])
        s2   = torch.matmul(q, k[:, :, half:].transpose(-2, -1)) * scale
        st.update(s2, v[:, :, half:])
        online_out = st.finalize()

        assert torch.allclose(online_out, std_out, atol=1e-4)
