"""
tests/test_data.py — Tests for the NanoMind data pipeline.
"""

import pytest
import tempfile
import torch
from pathlib import Path

from nanomind.tokenizer.char import CharTokenizer
from nanomind.data import (
    DataConfig, InMemoryTokenDataset, MixedDataset,
    pack_documents, make_input_target_pairs, dataset_stats,
)
from nanomind.data.text_dataset import TextFileDataset

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 20
TOKENIZER = CharTokenizer().build(CORPUS)
TOKENS    = torch.tensor(TOKENIZER.encode(CORPUS))
BLOCK     = 16


# ── DataConfig ────────────────────────────────────────────────────────────────

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.block_size == 512
        assert cfg.pack_documents is True
        assert cfg.stride == 512   # defaults to block_size

    def test_split_ratio_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            DataConfig(split_ratio=(0.7, 0.4))

    def test_invalid_block_size(self):
        with pytest.raises(AssertionError):
            DataConfig(block_size=0)

    def test_stride_defaults_to_block_size(self):
        cfg = DataConfig(block_size=64)
        assert cfg.stride == 64

    def test_custom_stride(self):
        cfg = DataConfig(block_size=64, stride=32)
        assert cfg.stride == 32


# ── pack_documents ────────────────────────────────────────────────────────────

class TestPackDocuments:
    def test_output_chunk_length(self):
        docs   = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        chunks = pack_documents(docs, block_size=4)
        assert all(len(c) == 4 for c in chunks)

    def test_empty_docs(self):
        chunks = pack_documents([], block_size=8)
        assert chunks == []

    def test_eos_inserted(self):
        docs   = [[1, 2], [3, 4]]
        flat   = []
        for d in docs:
            flat.extend(d)
            flat.append(0)
        chunks = pack_documents(docs, block_size=len(flat), drop_last=False)
        assert 0 in chunks[0]

    def test_stride_overlap(self):
        docs       = [list(range(20))]
        no_overlap = pack_documents(docs, block_size=5, stride=5)
        overlap    = pack_documents(docs, block_size=5, stride=2)
        assert len(overlap) > len(no_overlap)

    def test_make_input_target_pairs_shape(self):
        chunks      = [[i for i in range(9)] for _ in range(4)]
        inputs, tgts = make_input_target_pairs(chunks)
        assert inputs.shape == (4, 8)
        assert tgts.shape   == (4, 8)
        assert torch.equal(inputs[:, 1:], tgts[:, :-1])

    def test_make_pairs_empty(self):
        x, y = make_input_target_pairs([])
        assert x.shape[0] == 0


# ── InMemoryTokenDataset ──────────────────────────────────────────────────────

class TestInMemoryTokenDataset:
    def test_len(self):
        ds = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        assert len(ds) > 0

    def test_item_shapes(self):
        ds   = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        x, y = ds[0]
        assert x.shape == (BLOCK,)
        assert y.shape == (BLOCK,)

    def test_target_is_shifted(self):
        ds   = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        x, y = ds[0]
        assert torch.equal(x[1:], y[:-1])

    def test_split(self):
        ds          = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        train, val  = ds.split(0.8)
        assert len(train) > len(val)
        assert len(train) + len(val) <= len(ds) + 2   # stride rounding

    def test_stride_fewer_samples(self):
        ds_full   = InMemoryTokenDataset(TOKENS, block_size=BLOCK, stride=BLOCK)
        ds_stride = InMemoryTokenDataset(TOKENS, block_size=BLOCK, stride=BLOCK // 2)
        assert len(ds_stride) > len(ds_full)
