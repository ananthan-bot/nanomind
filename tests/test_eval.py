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


# ── Throughput + Memory ───────────────────────────────────────────────────────

class TestThroughput:
    def test_benchmark_memory_keys(self):
        model = tiny_model()
        mem   = benchmark_memory(model)
        for k in ("params_mb", "total_mb", "n_params"):
            assert k in mem

    def test_params_positive(self):
        model = tiny_model()
        mem   = benchmark_memory(model)
        assert mem["n_params"] > 0
        assert mem["params_mb"] > 0.0

    def test_benchmark_prefill_keys(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        speed = benchmark_prefill(model, cfg)
        for k in ("tokens_per_sec", "ms_per_batch"):
            assert k in speed

    def test_throughput_positive(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        speed = benchmark_prefill(model, cfg)
        assert speed["tokens_per_sec"] > 0.0

    def test_full_report_is_string(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        report = full_benchmark_report(model, cfg, name="Test")
        assert isinstance(report, str)
        assert "Test" in report


# ── Accuracy ──────────────────────────────────────────────────────────────────

class TestAccuracy:
    def test_top_1_perfect(self):
        logits  = torch.zeros(4, VOCAB)
        targets = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 100.0
        assert top_k_accuracy(logits, targets, k=1) == 1.0

    def test_top_1_all_wrong(self):
        logits  = torch.zeros(4, VOCAB)
        targets = torch.ones(4, dtype=torch.long)
        logits[:, 0] = 100.0   # always predicts 0, targets are 1
        assert top_k_accuracy(logits, targets, k=1) == 0.0

    def test_top_k_geq_top_1(self):
        logits  = torch.randn(16, VOCAB)
        targets = torch.randint(0, VOCAB, (16,))
        acc1    = top_k_accuracy(logits, targets, k=1)
        acc5    = top_k_accuracy(logits, targets, k=5)
        assert acc5 >= acc1

    def test_multi_k_accuracy_keys(self):
        logits  = torch.randn(8, VOCAB)
        targets = torch.randint(0, VOCAB, (8,))
        result  = multi_k_accuracy(logits, targets, k_values=[1, 5])
        assert "top_1" in result
        assert "top_5" in result

    def test_evaluate_accuracy_dataset(self):
        model  = tiny_model()
        loader = tiny_loader()
        result = evaluate_accuracy(model, loader, k_values=[1, 5], max_batches=3)
        assert "top_1" in result
        assert 0.0 <= result["top_1"] <= 1.0
