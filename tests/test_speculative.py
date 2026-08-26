"""
tests/test_speculative.py — Tests for speculative decoding.
"""

import pytest
import torch

from nanomind import NanoMind, ModelConfig
from nanomind.speculative import (
    SpeculativeConfig,
    SpeculativeGenerator,
    speculative_decode,
    generate_draft,
    verify_draft,
    rejection_sample,
    SpeculativeStats,
)
from nanomind.tokenizer.char import CharTokenizer

VOCAB  = 32
BLOCK  = 16
D_BIG  = 64
D_SML  = 32
B      = 1

def big_model():
    torch.manual_seed(0)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D_BIG, n_layers=2, n_heads=4, dropout=0.0))

def small_model():
    torch.manual_seed(1)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D_SML, n_layers=1, n_heads=2, dropout=0.0))

TOKENIZER = CharTokenizer().build("abcdefghijklmnopqrstuvwxyz " * 5)

def make_idx(t=4):
    return torch.randint(0, VOCAB, (1, t))


# ── SpeculativeConfig ─────────────────────────────────────────────────────────

class TestSpeculativeConfig:
    def test_defaults(self):
        cfg = SpeculativeConfig()
        assert cfg.n_draft == 5
        assert cfg.max_new_tokens == 100

    def test_invalid_n_draft(self):
        with pytest.raises(AssertionError):
            SpeculativeConfig(n_draft=0)

    def test_scaling_property(self):
        cfg = SpeculativeConfig(temperature=0.5)
        assert cfg.temperature == 0.5


# ── generate_draft ────────────────────────────────────────────────────────────

class TestGenerateDraft:
    def test_output_shapes(self):
        model = small_model()
        idx   = make_idx()
        ids, probs = generate_draft(model, idx, n_draft=5)
        assert ids.shape   == (5,)
        assert probs.shape == (5,)

    def test_probs_in_range(self):
        model = small_model()
        idx   = make_idx()
        _, probs = generate_draft(model, idx, n_draft=5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_ids_in_vocab(self):
        model = small_model()
        idx   = make_idx()
        ids, _ = generate_draft(model, idx, n_draft=5)
        assert (ids >= 0).all() and (ids < VOCAB).all()

    def test_n_draft_respected(self):
        model = small_model()
        idx   = make_idx()
        for n in [1, 3, 8]:
            ids, probs = generate_draft(model, idx, n_draft=n)
            assert ids.shape[0] == n
