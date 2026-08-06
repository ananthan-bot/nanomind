"""
tests/conftest.py — Shared pytest fixtures for NanoMind tests.
"""

import pytest
import torch

from nanomind.utils.seed import set_seed


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    """Set a fixed random seed before every test for reproducibility."""
    set_seed(42)


@pytest.fixture
def device() -> torch.device:
    """Return the best available device for tensor operations in tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_text() -> str:
    """A tiny corpus for fast tokenizer and data tests."""
    return (
        "Hello, world! This is NanoMind.\n"
        "A small language model built from scratch.\n"
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
        "0123456789 !@#$%^&*()\n"
    ) * 10
