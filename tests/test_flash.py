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
