"""
tests/test_model.py — Tests for the NanoMind model.
"""

import pytest
import torch

from nanomind.model import NanoMind, ModelConfig

# Tiny config for fast tests
CFG = ModelConfig(
    vocab_size=64,
    block_size=16,
    d_model=32,
    n_layers=2,
    n_heads=2,
    dropout=0.0,
)
B, T = 2, 8


@pytest.fixture
def model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(CFG)


# ── Forward pass shapes ───────────────────────────────────────────────────────

class TestForwardShape:
    def test_logits_shape(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, T))
        logits, loss = model(idx)
        assert logits.shape == (B, T, CFG.vocab_size)

    def test_no_targets_loss_is_none(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx)
        assert loss is None

    def test_with_targets_loss_is_scalar(self, model):
        idx     = torch.randint(0, CFG.vocab_size, (B, T))
        targets = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx, targets)
        assert loss is not None
        assert loss.shape == ()   # scalar

    def test_single_token(self, model):
        idx = torch.randint(0, CFG.vocab_size, (1, 1))
        logits, _ = model(idx)
        assert logits.shape == (1, 1, CFG.vocab_size)

    def test_full_block_size(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, CFG.block_size))
        logits, _ = model(idx)
        assert logits.shape == (B, CFG.block_size, CFG.vocab_size)
