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
