"""
tests/test_data.py — Tests for the NanoMind data pipeline.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from nanomind.data import (
    DataConfig,
    TextDataset,
    IterableTextDataset,
    split_dataset,
    get_dataloaders,
    dataset_stats,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = (
    "abcdefghijklmnopqrstuvwxyz " * 40
)
BLOCK_SIZE = 16
BATCH_SIZE = 4


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer().build(CORPUS)


@pytest.fixture
def dataset(tokenizer) -> TextDataset:
    return TextDataset.from_string(CORPUS, tokenizer, BLOCK_SIZE)


# ── TextDataset ───────────────────────────────────────────────────────────────

class TestTextDataset:
    def test_len(self, dataset):
        expected = dataset.num_tokens - BLOCK_SIZE
        assert len(dataset) == expected

    def test_item_shapes(self, dataset):
        x, y = dataset[0]
        assert x.shape == (BLOCK_SIZE,)
        assert y.shape == (BLOCK_SIZE,)

    def test_x_y_shifted_by_one(self, dataset):
        x, y = dataset[0]
        # y should be x shifted right by 1
        assert torch.equal(x[1:], y[:-1])

    def test_consecutive_windows_overlap(self, dataset):
        x0, _ = dataset[0]
        x1, _ = dataset[1]
        # Consecutive windows overlap by (block_size - 1) tokens
        assert torch.equal(x0[1:], x1[:-1])

    def test_dtype_is_long(self, dataset):
        x, y = dataset[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long
