"""
tests/test_rlhf.py — Tests for RLHF components.
"""

import pytest
import torch
import torch.nn as nn

from nanomind import NanoMind, ModelConfig
from nanomind.rlhf import (
    RewardModelConfig, PPOConfig,
    RewardModel, PreferenceDataset,
    preference_loss, preference_accuracy, reward_stats,
    ValueHead, compute_gae,
    token_kl_divergence, approx_token_kl, AdaptiveKLController,
    PPORolloutBuffer, ppo_total_loss,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = "abcdefghij " * 5
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg), cfg

def tiny_rm():
    m, cfg = tiny_model()
    return RewardModel(m, cfg, RewardModelConfig()), cfg


# ── RewardModelConfig ─────────────────────────────────────────────────────────

class TestRewardModelConfig:
    def test_defaults(self):
        cfg = RewardModelConfig()
        assert cfg.pooling == "last"

    def test_invalid_pooling(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(pooling="max")

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(dropout=1.5)


# ── RewardModel ───────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_output_shape(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        r     = rm(ids)
        assert r.shape == (B,)

    def test_output_scalar_per_sequence(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (1, T))
        r     = rm(ids)
        assert r.ndim == 1

    def test_mean_pooling(self):
        m, cfg = tiny_model()
        rm     = RewardModel(m, cfg, RewardModelConfig(pooling="mean"))
        ids    = torch.randint(0, VOCAB, (B, T))
        r      = rm(ids)
        assert r.shape == (B,)

    def test_output_finite(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        assert rm(ids).isfinite().all()
