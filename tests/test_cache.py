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


# ── LayerKVCache ──────────────────────────────────────────────────────────────

class TestLayerKVCache:
    def _make(self, max_seq=BLOCK):
        return LayerKVCache(max_batch_size=B, max_seq_len=max_seq,
                            n_heads=H, head_dim=D // H)

    def test_initial_state(self):
        c = self._make()
        assert c.is_empty
        assert c.current_len == 0

    def test_update_shape(self):
        c    = self._make()
        k    = torch.randn(B, 4, H, D // H)
        v    = torch.randn(B, 4, H, D // H)
        k_o, v_o = c.update(k, v)
        assert k_o.shape == (B, 4, H, D // H)
        assert v_o.shape == (B, 4, H, D // H)

    def test_current_len_increments(self):
        c = self._make()
        k = torch.randn(B, 3, H, D // H)
        v = torch.randn(B, 3, H, D // H)
        c.update(k, v)
        assert c.current_len == 3

    def test_accumulates_history(self):
        c = self._make()
        for _ in range(4):
            k = torch.randn(B, 2, H, D // H)
            v = torch.randn(B, 2, H, D // H)
            k_o, _ = c.update(k, v)
        assert k_o.shape[1] == 8   # 4 steps × 2 tokens

    def test_reset_clears(self):
        c = self._make()
        k = torch.randn(B, 4, H, D // H)
        c.update(k, k)
        c.reset()
        assert c.is_empty

    def test_overflow_raises(self):
        c = self._make(max_seq=4)
        k = torch.randn(B, 5, H, D // H)
        with pytest.raises(AssertionError):
            c.update(k, k)
