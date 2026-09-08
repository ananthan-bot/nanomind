"""
tests/test_dpo.py — Tests for DPO and Knowledge Distillation.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.dpo import DPOConfig, DPODataset, DPOTrainer, dpo_loss, compute_log_probs
from nanomind.distill import (
    DistillConfig, DistillTrainer,
    distillation_loss, soft_cross_entropy, feature_distillation_loss,
)

CORPUS = "abcdefghijklmnop " * 4
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model(d=D, layers=2):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=d,
                      n_layers=layers, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── DPOConfig ─────────────────────────────────────────────────────────────────

class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"

    def test_invalid_beta(self):
        with pytest.raises(AssertionError):
            DPOConfig(beta=0.0)

    def test_invalid_loss_type(self):
        with pytest.raises(AssertionError):
            DPOConfig(loss_type="ppo")

    def test_invalid_smoothing(self):
        with pytest.raises(AssertionError):
            DPOConfig(label_smoothing=0.6)


# ── dpo_loss ──────────────────────────────────────────────────────────────────

class TestDPOLoss:
    def test_loss_positive(self):
        lrc = torch.randn(B)
        lrr = torch.randn(B)
        loss, _ = dpo_loss(lrc, lrr)
        assert loss.item() > 0.0

    def test_returns_info_dict(self):
        lrc = torch.randn(4)
        lrr = torch.randn(4)
        _, info = dpo_loss(lrc, lrr)
        for k in ("loss","chosen_rewards","rejected_rewards","reward_margin","reward_accuracy"):
            assert k in info

    def test_perfect_separation_low_loss(self):
        """If chosen log-ratio >> rejected, loss should be near zero."""
        lrc = torch.tensor([5.0] * 8)
        lrr = torch.tensor([-5.0] * 8)
        loss, info = dpo_loss(lrc, lrr, beta=0.1)
        assert loss.item() < 0.01
        assert info["reward_accuracy"] == 1.0

    def test_ipo_loss_type(self):
        lrc = torch.randn(4)
        lrr = torch.randn(4)
        loss, _ = dpo_loss(lrc, lrr, loss_type="ipo")
        assert loss.item() >= 0.0

    def test_compute_log_probs_shape(self):
        logits = torch.randn(B, T, VOCAB)
        ids    = torch.randint(0, VOCAB, (B, T))
        lp     = compute_log_probs(logits, ids)
        assert lp.shape == (B,)
