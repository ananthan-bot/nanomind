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
