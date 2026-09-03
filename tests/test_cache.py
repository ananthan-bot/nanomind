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


# ── KVCacheManager ────────────────────────────────────────────────────────────

class TestKVCacheManager:
    def _make(self, n_layers=2):
        cfg = KVCacheConfig(max_batch_size=B, max_seq_len=BLOCK,
                            n_layers=n_layers, n_heads=H, head_dim=D // H)
        return KVCacheManager(cfg)

    def test_correct_number_of_caches(self):
        mgr = self._make(n_layers=4)
        assert len(mgr._caches) == 4

    def test_get_returns_layer_cache(self):
        mgr = self._make()
        assert isinstance(mgr.get(0), LayerKVCache)

    def test_current_len_after_update(self):
        mgr = self._make()
        k   = torch.randn(B, 3, H, D // H)
        mgr.get(0).update(k, k)
        assert mgr.current_len == 3

    def test_reset_all_layers(self):
        mgr = self._make()
        k   = torch.randn(B, 3, H, D // H)
        mgr.get(0).update(k, k)
        mgr.reset()
        assert mgr.current_len == 0

    def test_stats_keys(self):
        mgr   = self._make()
        stats = mgr.stats()
        for key in ("n_layers", "current_len", "max_seq_len", "fill_ratio", "memory_mb"):
            assert key in stats

    def test_memory_positive(self):
        mgr = self._make()
        assert mgr.total_memory_bytes() > 0


# ── NanoMindCached ────────────────────────────────────────────────────────────

class TestNanoMindCached:
    def test_forward_no_cache(self):
        model  = tiny_model()
        idx    = torch.randint(0, VOCAB, (B, 8))
        logits, loss = model(idx)
        assert logits.shape == (B, 8, VOCAB)
        assert loss is None

    def test_prefill_shape(self):
        model  = tiny_model()
        cache  = model.new_cache()
        idx    = torch.randint(0, VOCAB, (B, 8))
        logits = model.prefill(idx, cache)
        assert logits.shape == (B, 8, VOCAB)
        assert cache.current_len == 8

    def test_decode_step_shape(self):
        model  = tiny_model()
        cache  = model.new_cache()
        prompt = torch.randint(0, VOCAB, (B, 5))
        model.prefill(prompt, cache)
        tok    = torch.randint(0, VOCAB, (B, 1))
        logits = model.decode_step(tok, cache)
        assert logits.shape == (B, 1, VOCAB)

    def test_cache_grows_with_decode_steps(self):
        model  = tiny_model()
        cache  = model.new_cache()
        prompt = torch.randint(0, VOCAB, (B, 5))
        model.prefill(prompt, cache)
        for _ in range(3):
            tok    = torch.randint(0, VOCAB, (B, 1))
            model.decode_step(tok, cache)
        assert cache.current_len == 5 + 3

    def test_training_loss(self):
        model   = tiny_model()
        idx     = torch.randint(0, VOCAB, (B, 8))
        targets = torch.randint(0, VOCAB, (B, 8))
        _, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0.0


# ── CachedGenerator ───────────────────────────────────────────────────────────

class TestCachedGenerator:
    def test_generate_returns_string(self):
        gen  = CachedGenerator(tiny_model(), TOK)
        text = gen.generate("abc", max_new_tokens=5)
        assert isinstance(text, str)

    def test_generate_correct_length(self):
        gen   = CachedGenerator(tiny_model(), TOK)
        text  = gen.generate("abc", max_new_tokens=10)
        # Generated text should have up to 10 chars (may be fewer if EOS hit)
        assert len(text) <= 10

    def test_greedy_deterministic(self):
        """With temperature=0.01 and top_k=1, output should be deterministic."""
        model = tiny_model()
        gen   = CachedGenerator(model, TOK)
        t1    = gen.generate("abc", max_new_tokens=5, temperature=0.01, top_k=1)
        t2    = gen.generate("abc", max_new_tokens=5, temperature=0.01, top_k=1)
        assert t1 == t2

    def test_repr(self):
        gen = CachedGenerator(tiny_model(), TOK)
        assert "Cached" in repr(gen)
