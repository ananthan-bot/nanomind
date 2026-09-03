"""
tests/test_cache.py — Tests for KV Cache.
"""

import pytest
import torch

from nanomind.model.config import ModelConfig
from nanomind.cache import (
    KVCacheConfig, LayerKVCache, KVCacheManager,
    NanoMindCached, CachedGenerator, estimate_cache_memory,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = "abcdefghijklmnopqrstuvwxyz " * 4
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
BLOCK  = 32
D, H   = 64, 4
B      = 1


def tiny_model():
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                       d_model=D, n_layers=2, n_heads=H, dropout=0.0)
    ccfg = KVCacheConfig(max_batch_size=B, max_seq_len=BLOCK,
                         n_layers=2, n_heads=H, head_dim=D // H)
    return NanoMindCached(mcfg, ccfg)


# ── KVCacheConfig ─────────────────────────────────────────────────────────────

class TestKVCacheConfig:
    def test_defaults(self):
        cfg = KVCacheConfig()
        assert cfg.max_batch_size == 1
        assert cfg.dtype == "float32"

    def test_cache_size_bytes_positive(self):
        cfg = KVCacheConfig(max_seq_len=128, n_layers=4, n_heads=4, head_dim=16)
        assert cfg.cache_size_bytes > 0

    def test_cache_size_mb(self):
        cfg = KVCacheConfig(max_seq_len=128, n_layers=4, n_heads=4, head_dim=16)
        assert cfg.cache_size_mb > 0

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            KVCacheConfig(dtype="int8")

    def test_invalid_batch_size(self):
        with pytest.raises(AssertionError):
            KVCacheConfig(max_batch_size=0)
