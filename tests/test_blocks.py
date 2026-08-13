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
