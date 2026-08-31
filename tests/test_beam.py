"""
tests/test_beam.py — Tests for Beam Search and Diverse Beam Search.
"""

import pytest
import torch

from nanomind import NanoMind, ModelConfig
from nanomind.generate.beam import (
    BeamConfig, BeamHypothesis, BeamHypotheses,
    beam_search, diverse_beam_search,
    _block_repeat_ngrams,
)
from nanomind.generate.beam_generator import BeamSearchGenerator
from nanomind.tokenizer.char import CharTokenizer

VOCAB, BLOCK, D, H = 32, 16, 64, 4
B = 1

def tiny_model():
    torch.manual_seed(42)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D, n_layers=2, n_heads=H, dropout=0.0))

TOKENIZER = CharTokenizer().build("abcdefghijklmnopqrstuvwxyz " * 5)

def make_idx(t=4):
    return torch.randint(0, VOCAB, (1, t))


# ── BeamConfig ────────────────────────────────────────────────────────────────

class TestBeamConfig:
    def test_defaults(self):
        cfg = BeamConfig()
        assert cfg.num_beams == 4
        assert cfg.length_penalty == 1.0
        assert cfg.num_beam_groups == 1

    def test_invalid_num_beams(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=0)

    def test_invalid_group_divisor(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=4, num_beam_groups=3)

    def test_invalid_return_n_best(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=2, return_n_best=5)

    def test_temperature_positive(self):
        with pytest.raises(AssertionError):
            BeamConfig(temperature=0.0)
