"""
tests/conftest.py — Shared pytest fixtures for NanoMind test suite.
"""

import pytest
import torch

from nanomind.model import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer

TINY_CFG = ModelConfig(
    vocab_size=32, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)
CORPUS = "abcdefghijklmnopqrstuvwxyz " * 5


@pytest.fixture(scope="session")
def tiny_model():
    """A tiny NanoMind model shared across all tests in a session."""
    torch.manual_seed(0)
    return NanoMind(TINY_CFG)


@pytest.fixture(scope="session")
def char_tokenizer():
    """A fitted CharTokenizer shared across all tests."""
    return CharTokenizer().build(CORPUS)
