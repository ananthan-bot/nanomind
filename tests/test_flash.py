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


# ── tiled_flash_attention ─────────────────────────────────────────────────────

class TestTiledFlashAttention:
    def test_output_shape(self):
        q, k, v = make_qkv()
        out = tiled_flash_attention(q, k, v, block_q=8, block_kv=8)
        assert out.shape == (B, H, N, Dh)

    def test_causal_matches_sdpa(self):
        """Tiled implementation should match torch.sdpa with is_causal=True."""
        q, k, v  = make_qkv()
        scale    = Dh ** -0.5
        ref      = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        tiled    = tiled_flash_attention(q, k, v, block_q=8, block_kv=8,
                                          causal=True, scale=scale)
        assert torch.allclose(ref, tiled, atol=1e-4),             f"Max diff: {(ref-tiled).abs().max():.2e}"

    def test_non_causal_matches_sdpa(self):
        q, k, v = make_qkv()
        scale   = Dh ** -0.5
        ref     = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
        tiled   = tiled_flash_attention(q, k, v, block_q=8, block_kv=8,
                                         causal=False, scale=scale)
        assert torch.allclose(ref, tiled, atol=1e-4)

    def test_output_finite(self):
        q, k, v = make_qkv()
        out = tiled_flash_attention(q, k, v, block_q=8, block_kv=8, causal=True)
        assert out.isfinite().all()


# ── FlashAttention module ─────────────────────────────────────────────────────

class TestFlashAttentionModule:
    def test_output_shape_sdpa(self):
        attn = FlashAttention(D, H, FlashConfig(use_torch_sdpa=True))
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.shape == (B, N, D)

    def test_output_shape_tiled(self):
        attn = FlashAttention(D, H, FlashConfig(use_torch_sdpa=False))
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.shape == (B, N, D)

    def test_sdpa_and_tiled_close(self):
        """Both backends should produce numerically close outputs."""
        torch.manual_seed(1)
        x     = torch.randn(1, N, D)
        attn1 = FlashAttention(D, H, FlashConfig(use_torch_sdpa=True))
        attn2 = FlashAttention(D, H, FlashConfig(use_torch_sdpa=False))
        # Copy weights
        attn2.load_state_dict(attn1.state_dict())
        with torch.no_grad():
            out1, _ = attn1(x)
            out2, _ = attn2(x)
        assert torch.allclose(out1, out2, atol=1e-4),             f"Max diff: {(out1-out2).abs().max():.2e}"

    def test_output_finite(self):
        attn = FlashAttention(D, H)
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.isfinite().all()


# ── FlashTransformerBlock ─────────────────────────────────────────────────────

class TestFlashTransformerBlock:
    def test_output_shape(self):
        block = FlashTransformerBlock(D, H, FlashConfig(use_torch_sdpa=True))
        x     = torch.randn(B, N, D)
        out   = block(x)
        assert out.shape == (B, N, D)

    def test_residual_unchanged_dim(self):
        block = FlashTransformerBlock(D, H)
        x     = torch.randn(1, N, D)
        assert block(x).shape == x.shape

    def test_gradient_flows(self):
        block = FlashTransformerBlock(D, H)
        x     = torch.randn(B, N, D, requires_grad=True)
        out   = block(x)
        out.sum().backward()
        assert x.grad is not None


# ── Memory analysis ───────────────────────────────────────────────────────────

class TestMemoryAnalysis:
    def test_standard_larger_than_flash(self):
        std   = standard_attention_memory(1, 4, 512, 64)
        flash = flash_attention_memory(1, 4, 512, 64)
        assert std["total_bytes"] > flash["total_bytes"]

    def test_standard_scales_quadratically(self):
        s256 = standard_attention_memory(1, 1, 256, 64)
        s512 = standard_attention_memory(1, 1, 512, 64)
        # Attention matrix: (2N)² = 4× more memory
        ratio = s512["attn_matrix_bytes"] / s256["attn_matrix_bytes"]
        assert abs(ratio - 4.0) < 0.01

    def test_flash_scales_linearly(self):
        f256  = flash_attention_memory(1, 1, 256, 64)
        f512  = flash_attention_memory(1, 1, 512, 64)
        # QKV buffers scale linearly; tile is constant
        # output buffer 2× bigger for 2× seq_len
        assert f512["output_bytes"] == f256["output_bytes"] * 2

    def test_report_is_string(self):
        report = memory_comparison_report(1024)
        assert isinstance(report, str)
        assert "Flash" in report


# ── NanoMindFlash ─────────────────────────────────────────────────────────────

class TestNanoMindFlash:
    def _make(self, sdpa=True):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=32, block_size=N, d_model=D,
                          n_layers=2, n_heads=H, dropout=0.0)
        return NanoMindFlash(cfg, FlashConfig(use_torch_sdpa=sdpa))

    def test_forward_shape(self):
        model  = self._make()
        idx    = torch.randint(0, 32, (B, N))
        logits, loss = model(idx)
        assert logits.shape == (B, N, 32)
        assert loss is None

    def test_training_loss(self):
        model   = self._make()
        idx     = torch.randint(0, 32, (B, N))
        targets = torch.randint(0, 32, (B, N))
        _, loss = model(idx, targets)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        model   = self._make()
        idx     = torch.randint(0, 32, (B, N))
        targets = torch.randint(0, 32, (B, N))
        _, loss = model(idx, targets)
        loss.backward()
        for n, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {n}"

    def test_tiled_backend(self):
        model  = self._make(sdpa=False)
        idx    = torch.randint(0, 32, (B, N))
        logits, _ = model(idx)
        assert logits.shape == (B, N, 32)
        assert logits.isfinite().all()
