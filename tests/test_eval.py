"""
tests/test_eval.py — Tests for NanoMind evaluation metrics.
"""

import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model import NanoMind, ModelConfig
from nanomind.eval import (
    EvalConfig,
    EvalResult,
    Evaluator,
    perplexity,
    bits_per_character,
    token_accuracy,
    top_k_accuracy,
    type_token_ratio,
    distinct_n,
    repetition_fraction,
    generation_report,
)

VOCAB = 32
CFG   = ModelConfig(
    vocab_size=VOCAB, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NanoMind(CFG)


def make_loader(n: int = 32, b: int = 4, t: int = 8):
    tokens = torch.randint(0, VOCAB, (n + t,))
    xs = torch.stack([tokens[i:i+t]   for i in range(n)])
    ys = torch.stack([tokens[i+1:i+t+1] for i in range(n)])
    return DataLoader(TensorDataset(xs, ys), batch_size=b, drop_last=True)


# ── perplexity / BPC ──────────────────────────────────────────────────────────

class TestPerplexity:
    def test_ppl_of_zero_loss(self):
        assert perplexity(0.0) == 1.0

    def test_ppl_of_log_vocab(self):
        # Uniform model: loss = log(vocab_size), PPL = vocab_size
        loss = math.log(VOCAB)
        assert abs(perplexity(loss) - VOCAB) < 0.01

    def test_ppl_monotone_in_loss(self):
        assert perplexity(1.0) < perplexity(2.0) < perplexity(3.0)

    def test_bpc_zero_loss(self):
        assert bits_per_character(0.0) == 0.0

    def test_bpc_log2(self):
        # loss = log(2) -> BPC = 1.0
        assert abs(bits_per_character(math.log(2)) - 1.0) < 1e-9

    def test_bpc_larger_than_loss_in_nats(self):
        # BPC = loss / log(2) > loss for loss > 0
        loss = 2.5
        assert bits_per_character(loss) > loss


# ── Accuracy metrics ──────────────────────────────────────────────────────────

class TestAccuracyMetrics:
    def test_perfect_accuracy(self):
        vocab = 10
        targets = torch.tensor([0, 1, 2, 3])
        logits  = torch.zeros(4, vocab)
        for i, t in enumerate(targets):
            logits[i, t] = 100.0   # argmax = target
        assert token_accuracy(logits, targets) == 1.0

    def test_zero_accuracy(self):
        vocab = 10
        targets = torch.tensor([0, 0, 0, 0])
        logits  = torch.zeros(4, vocab)
        logits[:, 1] = 100.0   # argmax = 1, targets = 0
        assert token_accuracy(logits, targets) == 0.0

    def test_accuracy_in_01(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc = token_accuracy(logits, y)
        assert 0.0 <= acc <= 1.0

    def test_top_k_acc_geq_top1(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc1 = token_accuracy(logits, y)
        acc5 = top_k_accuracy(logits, y, k=5)
        assert acc5 >= acc1

    def test_top_k_perfect_if_k_equals_vocab(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc = top_k_accuracy(logits, y, k=VOCAB)
        assert acc == 1.0   # every token is in top-VOCAB


# ── EvalResult ────────────────────────────────────────────────────────────────

class TestEvalResult:
    def test_from_loss_computes_ppl(self):
        r = EvalResult.from_loss(math.log(VOCAB))
        assert abs(r.ppl - VOCAB) < 0.01

    def test_from_loss_computes_bpc(self):
        loss = math.log(2)
        r    = EvalResult.from_loss(loss)
        assert abs(r.bpc - 1.0) < 1e-6

    def test_str_contains_ppl(self):
        r = EvalResult.from_loss(1.0)
        assert "ppl" in str(r)

    def test_str_contains_loss(self):
        r = EvalResult.from_loss(1.0)
        assert "loss" in str(r)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_perplexity_returns_result(self, model):
        loader = make_loader()
        ev     = Evaluator(model, EvalConfig(), torch.device("cpu"))
        result = ev.evaluate_perplexity(loader)
        assert isinstance(result, EvalResult)
        assert result.ppl > 1.0

    def test_full_eval_all_metrics(self, model):
        loader = make_loader()
        ev     = Evaluator(model, EvalConfig(compute_acc=True, compute_top_k=True))
        result = ev.full_eval(loader)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.top_k_acc <= 1.0
        assert result.n_batches > 0

    def test_max_batches_limits_evaluation(self, model):
        loader = make_loader(n=64)
        ev1 = Evaluator(model, EvalConfig(max_batches=1))
        ev2 = Evaluator(model, EvalConfig(max_batches=0))
        r1  = ev1.full_eval(loader)
        r2  = ev2.full_eval(loader)
        assert r1.n_batches == 1
        assert r2.n_batches > 1
