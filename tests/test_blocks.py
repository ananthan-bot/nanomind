"""
tests/test_blocks.py — Tests for NanoMind transformer blocks.
"""

import pytest
import torch

from nanomind.blocks import (
    TransformerBlock,
    FeedForward,
    LayerNorm,
    RMSNorm,
    BlockConfig,
    get_norm,
    get_ffn,
    block_from_config,
)

B, T, D, H = 2, 16, 64, 4


@pytest.fixture
def block() -> TransformerBlock:
    return TransformerBlock(d_model=D, n_heads=H, block_size=T, dropout=0.0)


# ── TransformerBlock output shape ─────────────────────────────────────────────

class TestTransformerBlockShape:
    def test_output_shape(self, block):
        x = torch.randn(B, T, D)
        out, weights = block(x)
        assert out.shape == (B, T, D)

    def test_weights_shape(self, block):
        x = torch.randn(B, T, D)
        _, weights = block(x)
        assert weights.shape == (B, H, T, T)

    def test_single_token(self, block):
        x = torch.randn(B, 1, D)
        out, _ = block(x)
        assert out.shape == (B, 1, D)

    def test_batch_independence(self, block):
        # Each sample in a batch should be independent
        x = torch.randn(B, T, D)
        out_full, _ = block(x)
        out_single, _ = block(x[:1])
        assert torch.allclose(out_full[:1], out_single, atol=1e-5)


# ── Residual connections ──────────────────────────────────────────────────────

class TestResidualConnections:
    def test_residual_preserves_scale(self, block):
        # With very small weights (near zero init), output ~ input
        # We check the block doesnt explode or collapse
        x = torch.randn(B, T, D)
        out, _ = block(x)
        # Output should be in a reasonable range
        assert out.abs().max().item() < 1000.0

    def test_residual_scale_default_one(self, block):
        assert block.residual_scale == 1.0

    def test_residual_scale_can_be_changed(self, block):
        block.residual_scale = 0.5
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)


# ── Pre-LN vs Post-LN ────────────────────────────────────────────────────────

class TestNormPlacement:
    def test_pre_norm_output_shape(self):
        block = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                 dropout=0.0, norm_placement="pre")
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)

    def test_post_norm_output_shape(self):
        block = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                 dropout=0.0, norm_placement="post")
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)

    def test_pre_post_outputs_differ(self):
        torch.manual_seed(0)
        pre  = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                dropout=0.0, norm_placement="pre")
        torch.manual_seed(0)
        post = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                dropout=0.0, norm_placement="post")
        x = torch.randn(B, T, D)
        out_pre,  _ = pre(x)
        out_post, _ = post(x)
        # Different placement -> different outputs
        assert not torch.allclose(out_pre, out_post)


# ── FeedForward ───────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_gelu_output_shape(self):
        ffn = FeedForward(d_model=D, dropout=0.0, activation="gelu")
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_swiglu_output_shape(self):
        ffn = FeedForward(d_model=D, dropout=0.0, activation="swiglu")
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_custom_d_ff(self):
        ffn = FeedForward(d_model=D, d_ff=D * 8, dropout=0.0)
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_get_ffn_factory(self):
        ffn = get_ffn(D, activation="gelu")
        assert isinstance(ffn, FeedForward)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            get_ffn(D, activation="relu")
