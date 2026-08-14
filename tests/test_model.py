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


# ── Loss computation ──────────────────────────────────────────────────────────

class TestLossComputation:
    def test_loss_is_positive(self, model):
        idx     = torch.randint(0, CFG.vocab_size, (B, T))
        targets = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx, targets)
        assert loss.item() > 0

    def test_loss_decreases_with_correct_targets(self, model):
        # Force a simple case: one token, targets = input (should be easy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        idx     = torch.tensor([[5, 10, 15]])
        targets = torch.tensor([[10, 15, 5]])
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # Loss should go down over 30 steps
        assert losses[-1] < losses[0]

    def test_random_loss_near_log_vocab(self, model):
        # Untrained model: loss should be ~log(vocab_size) = log(64) ≈ 4.15
        import math
        idx     = torch.randint(0, CFG.vocab_size, (4, T))
        targets = torch.randint(0, CFG.vocab_size, (4, T))
        _, loss = model(idx, targets)
        expected = math.log(CFG.vocab_size)
        assert abs(loss.item() - expected) < 1.0


# ── Weight tying ──────────────────────────────────────────────────────────────

class TestWeightTying:
    def test_weights_are_same_object(self, model):
        # With weight_tying=True, lm_head.weight IS token_emb.weight
        assert model.lm_head.weight is model.token_emb.weight

    def test_no_weight_tying(self):
        cfg = ModelConfig(
            vocab_size=64, block_size=16, d_model=32,
            n_layers=2, n_heads=2, dropout=0.0, weight_tying=False
        )
        m = NanoMind(cfg)
        assert m.lm_head.weight is not m.token_emb.weight

    def test_tying_reduces_params(self):
        tied   = NanoMind(ModelConfig(vocab_size=64, block_size=16, d_model=32,
                                      n_layers=2, n_heads=2, dropout=0.0, weight_tying=True))
        untied = NanoMind(ModelConfig(vocab_size=64, block_size=16, d_model=32,
                                      n_layers=2, n_heads=2, dropout=0.0, weight_tying=False))
        assert tied.num_parameters() < untied.num_parameters()
