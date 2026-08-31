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


# ── BeamHypothesis ────────────────────────────────────────────────────────────

class TestBeamHypothesis:
    def test_extend_appends_token(self):
        hyp = BeamHypothesis([1, 2, 3], log_prob=-1.0)
        new = hyp.extend(4, -0.5)
        assert new.tokens == [1, 2, 3, 4]
        assert abs(new.log_prob - (-1.5)) < 1e-6

    def test_len(self):
        hyp = BeamHypothesis([1, 2, 3])
        assert len(hyp) == 3

    def test_score_with_no_penalty(self):
        hyp  = BeamHypothesis([1, 2, 3, 4], log_prob=-4.0)
        s1   = hyp.score(length_penalty=1.0)
        s0   = hyp.score(length_penalty=0.0)
        assert isinstance(s1, float)
        assert isinstance(s0, float)

    def test_longer_favoured_by_high_penalty(self):
        short = BeamHypothesis([1, 2],             log_prob=-2.0)
        long_ = BeamHypothesis([1, 2, 3, 4, 5, 6], log_prob=-6.0)
        # With high length penalty, longer sequence should score better
        assert long_.score(2.0) > short.score(2.0)


class TestBeamHypotheses:
    def test_add_and_best(self):
        bh = BeamHypotheses(num_beams=2, length_penalty=1.0)
        h1 = BeamHypothesis([1, 2], log_prob=-1.0)
        h2 = BeamHypothesis([1, 3], log_prob=-0.5)
        bh.add(h1)
        bh.add(h2)
        best = bh.best(2)
        assert len(best) == 2
        assert best[0].log_prob == -0.5   # higher score first

    def test_capacity_capped_at_num_beams(self):
        bh = BeamHypotheses(num_beams=2, length_penalty=1.0)
        for i in range(5):
            bh.add(BeamHypothesis([i], log_prob=float(-i)))
        assert len(bh.hyps) <= 2

    def test_is_done_when_full_and_early_stop(self):
        bh = BeamHypotheses(num_beams=2, early_stopping=True)
        bh.add(BeamHypothesis([1], log_prob=0.0))
        bh.add(BeamHypothesis([2], log_prob=-0.1))
        assert bh.is_done

    def test_not_done_without_early_stop(self):
        bh = BeamHypotheses(num_beams=2, early_stopping=False)
        bh.add(BeamHypothesis([1], log_prob=0.0))
        bh.add(BeamHypothesis([2], log_prob=-0.1))
        assert not bh.is_done
