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


# ── Split ─────────────────────────────────────────────────────────────────────

class TestSplitDataset:
    def test_sizes_sum_to_total(self, dataset):
        train_ds, val_ds = split_dataset(dataset, val_fraction=0.1)
        assert len(train_ds) + len(val_ds) == len(dataset)

    def test_val_fraction_respected(self, dataset):
        frac = 0.2
        _, val_ds = split_dataset(dataset, val_fraction=frac)
        ratio = len(val_ds) / len(dataset)
        assert abs(ratio - frac) < 0.01

    def test_reproducible_with_same_seed(self, dataset):
        train1, _ = split_dataset(dataset, seed=99)
        train2, _ = split_dataset(dataset, seed=99)
        # Same seed => same split => same first indices
        assert train1.indices[:5] == train2.indices[:5]

    def test_different_seeds_differ(self, dataset):
        train1, _ = split_dataset(dataset, seed=0)
        train2, _ = split_dataset(dataset, seed=1)
        assert train1.indices[:5] != train2.indices[:5]


# ── DataLoader ────────────────────────────────────────────────────────────────

class TestGetDataloaders:
    def test_returns_two_loaders(self, tokenizer):
        train, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        assert train is not None
        assert val is not None

    def test_train_batch_shape(self, tokenizer):
        train, _ = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        x, y = next(iter(train))
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_val_batch_shape(self, tokenizer):
        _, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        x, y = next(iter(val))
        assert x.shape[1] == BLOCK_SIZE

    def test_train_larger_than_val(self, tokenizer):
        train, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        assert len(train) >= len(val)
