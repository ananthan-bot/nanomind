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
