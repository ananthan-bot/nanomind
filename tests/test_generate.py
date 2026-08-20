"""
tests/test_generate.py — Tests for NanoMind text generation.
"""

import pytest
import torch
import torch.nn.functional as F

from nanomind.model import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.generate import (
    GenerationConfig,
    Generator,
    greedy_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    sample_next_token,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
)

VOCAB = 32
CFG   = ModelConfig(
    vocab_size=VOCAB, block_size=16,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)
CORPUS = "abcdefghijklmnopqrstuvwxyz " * 10


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NanoMind(CFG)


@pytest.fixture
def tokenizer():
    return CharTokenizer().build(CORPUS)


@pytest.fixture
def generator(model, tokenizer):
    return Generator(model, tokenizer, device=torch.device("cpu"))


# ── greedy_decode ─────────────────────────────────────────────────────────────

class TestGreedyDecode:
    def test_returns_argmax(self):
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        assert greedy_decode(logits).item() == 1   # index of max = 1

    def test_deterministic(self):
        logits = torch.randn(VOCAB)
        assert greedy_decode(logits).item() == greedy_decode(logits).item()

    def test_returns_scalar(self):
        logits = torch.randn(VOCAB)
        result = greedy_decode(logits)
        assert result.shape == ()
