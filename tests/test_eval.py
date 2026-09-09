"""
tests/test_eval.py — Tests for benchmarking and evaluation suite.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.eval import (
    BenchmarkConfig, EvalRunner,
    compute_perplexity, perplexity_from_logits,
    benchmark_prefill, benchmark_memory, full_benchmark_report,
    top_k_accuracy, multi_k_accuracy, evaluate_accuracy,
)

CORPUS = "abcdefghij " * 8
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)

def tiny_loader():
    xs = torch.randint(0, VOCAB, (16, T))
    ys = torch.randint(0, VOCAB, (16, T))
    return DataLoader(TensorDataset(xs, ys), batch_size=B)


# ── BenchmarkConfig ───────────────────────────────────────────────────────────

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.batch_size == 4
        assert 1 in cfg.top_k_values

    def test_invalid_batch_size(self):
        with pytest.raises(AssertionError):
            BenchmarkConfig(batch_size=0)

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            BenchmarkConfig(dtype="int8")


# ── Perplexity ────────────────────────────────────────────────────────────────

class TestPerplexity:
    def test_compute_perplexity_keys(self):
        model  = tiny_model()
        loader = tiny_loader()
        result = compute_perplexity(model, loader, max_batches=3)
        for k in ("perplexity", "nll", "n_tokens", "bits_per_char"):
            assert k in result

    def test_perplexity_positive(self):
        model  = tiny_model()
        loader = tiny_loader()
        r      = compute_perplexity(model, loader, max_batches=3)
        assert r["perplexity"] > 1.0

    def test_perplexity_from_logits(self):
        logits  = torch.randn(8, VOCAB)
        targets = torch.randint(0, VOCAB, (8,))
        ppl     = perplexity_from_logits(logits, targets)
        assert ppl > 1.0
        assert math.isfinite(ppl)

    def test_perfect_prediction_low_ppl(self):
        """Near-perfect predictions → PPL close to 1."""
        V      = 10
        logits = torch.zeros(4, V)
        tgts   = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 100.0   # model assigns all prob to token 0
        ppl    = perplexity_from_logits(logits, tgts)
        assert ppl < 1.01
