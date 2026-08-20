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


# ── Logit processors ──────────────────────────────────────────────────────────

class TestLogitProcessors:
    def test_temperature_sharpens(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        low_t  = F.softmax(apply_temperature(logits, 0.1), dim=-1)
        high_t = F.softmax(apply_temperature(logits, 5.0), dim=-1)
        # Low temperature -> higher max prob
        assert low_t.max() > high_t.max()

    def test_top_k_keeps_k_tokens(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_k(logits, top_k=5)
        n_finite = (filtered != float("-inf")).sum().item()
        assert n_finite == 5

    def test_top_k_zero_no_filter(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_k(logits, top_k=0)
        assert torch.equal(filtered, logits)

    def test_top_p_filters_low_prob(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_p(logits, top_p=0.5)
        n_inf    = (filtered == float("-inf")).sum().item()
        assert n_inf > 0   # some tokens should be removed

    def test_repetition_penalty_reduces_seen_token(self):
        logits  = torch.zeros(VOCAB)
        logits[5] = 2.0
        past    = torch.tensor([5])
        penalized = apply_repetition_penalty(logits, past, penalty=2.0)
        assert penalized[5] < logits[5]   # penalized = original / 2

    def test_rep_penalty_one_is_no_op(self):
        logits  = torch.randn(VOCAB)
        past    = torch.arange(VOCAB)
        result  = apply_repetition_penalty(logits, past, penalty=1.0)
        assert torch.equal(result, logits)


# ── Sampling strategies ───────────────────────────────────────────────────────

class TestSamplingStrategies:
    def test_temperature_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = temperature_sample(logits)
        assert 0 <= tok.item() < VOCAB

    def test_top_k_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = top_k_sample(logits, top_k=5)
        assert 0 <= tok.item() < VOCAB

    def test_top_p_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = top_p_sample(logits, top_p=0.9)
        assert 0 <= tok.item() < VOCAB

    def test_greedy_vs_temperature_zero(self):
        logits = torch.randn(VOCAB)
        greedy = greedy_decode(logits).item()
        sampled = temperature_sample(logits, temperature=1e-8).item()
        assert greedy == sampled   # Very low temp -> greedy-like

    def test_sample_next_token_dispatcher_greedy(self):
        logits = torch.randn(VOCAB)
        g      = greedy_decode(logits).item()
        s      = sample_next_token(logits, strategy="greedy").item()
        assert g == s


# ── Generator ─────────────────────────────────────────────────────────────────

class TestGenerator:
    def test_generate_returns_string(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="greedy")
        out = generator.generate("abc", cfg)
        assert isinstance(out, str)

    def test_generate_length_bounded(self, generator):
        cfg = GenerationConfig(max_new_tokens=10, strategy="greedy")
        out = generator.generate("abc", cfg)
        # Decoded output length depends on tokenizer but tokens <= max_new_tokens
        assert len(generator.tokenizer.encode(out)) <= 10

    def test_greedy_is_deterministic(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="greedy")
        out1 = generator.generate("abc", cfg)
        out2 = generator.generate("abc", cfg)
        assert out1 == out2

    def test_seeded_generation_is_deterministic(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="temperature", seed=42)
        out1 = generator.generate("abc", cfg)
        out2 = generator.generate("abc", cfg)
        assert out1 == out2


# ── Generator.stream() ────────────────────────────────────────────────────────

class TestGeneratorStream:
    def test_stream_yields_strings(self, generator):
        cfg    = GenerationConfig(max_new_tokens=5, strategy="greedy")
        tokens = list(generator.stream("abc", cfg))
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_n_tokens(self, generator):
        cfg    = GenerationConfig(max_new_tokens=5, strategy="greedy")
        tokens = list(generator.stream("abc", cfg))
        assert len(tokens) == 5

    def test_stream_concat_matches_generate(self, generator):
        cfg  = GenerationConfig(max_new_tokens=5, strategy="greedy")
        gen  = generator.generate("abc", cfg)
        strm = "".join(generator.stream("abc", cfg))
        assert gen == strm


# ── GenerationConfig ──────────────────────────────────────────────────────────

class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.strategy == "temperature"
        assert cfg.max_new_tokens == 100

    def test_invalid_strategy(self):
        with pytest.raises(AssertionError):
            GenerationConfig(strategy="random_walk")

    def test_invalid_top_p(self):
        with pytest.raises(AssertionError):
            GenerationConfig(top_p=1.5)

    def test_invalid_rep_penalty(self):
        with pytest.raises(AssertionError):
            GenerationConfig(repetition_penalty=0.5)   # must be >= 1.0
