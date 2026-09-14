"""tests/test_cache.py — Tests for NanoMind KV-cache and fast inference."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.cache import (
    CacheConfig, KVCache, LayerCache,
    CachedAttention, CacheManager,
    PrefixCache, CachedInferenceEngine,
    SpeculativeDecoder,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
class CharTok:
    def __init__(self, text="abcde "):
        chars = sorted(set(text * 5))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V=8, D=16, T=16):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  t.view(-1)) if t is not None else None
        return logits, loss

TOK   = CharTok()
MODEL = TinyLM(V=TOK.vocab_size)

def tiny_cfg(**kw):
    return CacheConfig(n_layers=2, n_heads=2, d_head=4, max_seq_len=16, **kw)


# ── CacheConfig ───────────────────────────────────────────────────────────────

class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.max_seq_len == 2048
        assert cfg.eviction == "sliding"

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            CacheConfig(dtype="int8")

    def test_invalid_eviction(self):
        with pytest.raises(AssertionError):
            CacheConfig(eviction="fifo")

    def test_memory_mb_positive(self):
        cfg = tiny_cfg()
        assert cfg.memory_mb > 0.0

    def test_memory_mb_formula(self):
        cfg = CacheConfig(n_layers=1, n_heads=1, d_head=4, max_seq_len=8,
                          max_batch_size=1, dtype="float32")
        # 2 * 1 * 1 * 1 * 8 * 4 * 4 bytes = 256 bytes = 0.000244 MB
        expected = 2 * 1 * 1 * 1 * 8 * 4 * 4 / (1024**2)
        assert abs(cfg.memory_mb - expected) < 1e-9


# ── KVCache ───────────────────────────────────────────────────────────────────

class TestKVCache:
    def _cache(self): return KVCache(tiny_cfg())
    def _kv(self, s=4): return (torch.randn(1, 2, s, 4), torch.randn(1, 2, s, 4))

    def test_initial_seq_len_zero(self):
        assert self._cache().seq_len == 0

    def test_append_updates_seq_len(self):
        cache = self._cache()
        k, v  = self._kv(4)
        cache.append(0, k, v)
        assert cache.seq_len == 4

    def test_get_returns_tensors(self):
        cache = self._cache()
        k, v  = self._kv(3)
        cache.append(0, k, v)
        ck, cv = cache.get(0)
        assert ck.shape == (1, 2, 3, 4)
        assert cv.shape == (1, 2, 3, 4)

    def test_get_content_matches(self):
        cache = self._cache()
        k, v  = self._kv(2)
        cache.append(0, k, v)
        ck, cv = cache.get(0)
        assert torch.allclose(ck, k)
        assert torch.allclose(cv, v)

    def test_reset_clears_seq_len(self):
        cache = self._cache()
        k, v  = self._kv(4)
        cache.append(0, k, v)
        cache.reset()
        assert cache.seq_len == 0

    def test_sliding_eviction(self):
        cfg   = CacheConfig(n_layers=1, n_heads=1, d_head=4, max_seq_len=8)
        cache = KVCache(cfg)
        k, v  = torch.randn(1, 1, 6, 4), torch.randn(1, 1, 6, 4)
        cache.append(0, k, v)
        k2, v2 = torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4)
        cache.append(0, k2, v2)
        assert cache.seq_len == 8  # capped at max_seq_len

    def test_memory_used_mb(self):
        cache = self._cache()
        assert cache.memory_used_mb() > 0.0

    def test_stats_keys(self):
        cache = self._cache()
        stats = cache.stats()
        for k in ("n_layers", "seq_len", "max_seq_len", "fill_pct", "memory_mb"):
            assert k in stats

    def test_n_layers(self):
        assert self._cache().n_layers == 2


# ── CachedAttention ───────────────────────────────────────────────────────────

class TestCachedAttention:
    def _attn(self):
        return CachedAttention(d_model=16, n_heads=2, layer_id=0)

    def test_forward_no_cache(self):
        attn   = self._attn()
        x      = torch.randn(1, 4, 16)
        out    = attn(x)
        assert out.shape == (1, 4, 16)

    def test_forward_with_cache(self):
        attn   = self._attn()
        cache  = KVCache(CacheConfig(n_layers=1, n_heads=2, d_head=8, max_seq_len=16))
        x      = torch.randn(1, 4, 16)
        out    = attn(x, cache=cache)
        assert out.shape == (1, 4, 16)

    def test_cache_seq_len_grows(self):
        attn  = self._attn()
        cache = KVCache(CacheConfig(n_layers=1, n_heads=2, d_head=8, max_seq_len=16))
        x1    = torch.randn(1, 4, 16)
        x2    = torch.randn(1, 2, 16)
        attn(x1, cache=cache)
        attn(x2, cache=cache)
        assert cache.seq_len == 6

    def test_output_dtype_preserved(self):
        attn = self._attn()
        x    = torch.randn(1, 3, 16)
        out  = attn(x)
        assert out.dtype == torch.float32


# ── CacheManager ─────────────────────────────────────────────────────────────

class TestCacheManager:
    def _mgr(self, max_r=3):
        return CacheManager(tiny_cfg(), max_requests=max_r)

    def test_get_or_create(self):
        mgr   = self._mgr()
        cache = mgr.get_or_create("r1")
        assert isinstance(cache, KVCache)

    def test_active_count_increments(self):
        mgr = self._mgr()
        mgr.get_or_create("r1")
        mgr.get_or_create("r2")
        assert mgr.active_count == 2

    def test_lru_eviction(self):
        mgr = self._mgr(max_r=2)
        mgr.get_or_create("r1")
        mgr.get_or_create("r2")
        mgr.get_or_create("r3")   # should evict r1
        assert mgr.active_count == 2
        assert "r1" not in mgr._caches

    def test_release(self):
        mgr = self._mgr()
        mgr.get_or_create("r1")
        mgr.release("r1")
        assert mgr.active_count == 0

    def test_stats_keys(self):
        mgr   = self._mgr()
        mgr.get_or_create("r1")
        stats = mgr.stats()
        assert "active" in stats and "request_ids" in stats

    def test_same_id_returns_same_cache(self):
        mgr = self._mgr()
        c1  = mgr.get_or_create("r1")
        c2  = mgr.get_or_create("r1")
        assert c1 is c2


# ── PrefixCache ───────────────────────────────────────────────────────────────

class TestPrefixCache:
    def _pc(self): return PrefixCache(tiny_cfg(), max_prefixes=4)

    def test_store_and_lookup(self):
        pc = self._pc()
        kv = KVCache(tiny_cfg())
        pc.store("hello", kv)
        assert pc.lookup("hello") is kv

    def test_miss_returns_none(self):
        pc = self._pc()
        assert pc.lookup("not stored") is None

    def test_hit_rate(self):
        pc = self._pc()
        pc.store("hi", KVCache(tiny_cfg()))
        pc.lookup("hi")        # hit
        pc.lookup("miss")      # miss
        assert abs(pc.hit_rate - 0.5) < 1e-6

    def test_max_prefixes_eviction(self):
        pc = PrefixCache(tiny_cfg(), max_prefixes=2)
        pc.store("a", KVCache(tiny_cfg()))
        pc.store("b", KVCache(tiny_cfg()))
        pc.store("c", KVCache(tiny_cfg()))   # evicts "a"
        assert pc.lookup("a") is None

    def test_invalidate(self):
        pc = self._pc()
        pc.store("x", KVCache(tiny_cfg()))
        pc.invalidate("x")
        assert pc.lookup("x") is None

    def test_clear(self):
        pc = self._pc()
        pc.store("a", KVCache(tiny_cfg()))
        pc.store("b", KVCache(tiny_cfg()))
        pc.clear()
        assert pc.lookup("a") is None


# ── CachedInferenceEngine ─────────────────────────────────────────────────────

class TestCachedInferenceEngine:
    def _engine(self):
        return CachedInferenceEngine(MODEL, TOK)

    def test_generate_returns_string(self):
        eng = self._engine()
        out = eng.generate("abc", max_new_tokens=4)
        assert isinstance(out, str)

    def test_benchmark_keys(self):
        eng   = self._engine()
        bench = eng.benchmark("ab", max_new_tokens=4)
        for k in ("n_tokens", "total_s", "ms_per_token", "tokens_per_sec"):
            assert k in bench

    def test_benchmark_n_tokens(self):
        eng   = self._engine()
        bench = eng.benchmark("a", max_new_tokens=5)
        assert bench["n_tokens"] == 5

    def test_benchmark_speed_positive(self):
        eng   = self._engine()
        bench = eng.benchmark("a", max_new_tokens=3)
        assert bench["tokens_per_sec"] > 0.0


# ── SpeculativeDecoder ────────────────────────────────────────────────────────

class TestSpeculativeDecoder:
    def _decoder(self):
        draft  = TinyLM(V=TOK.vocab_size, D=8)
        target = TinyLM(V=TOK.vocab_size, D=16)
        return SpeculativeDecoder(draft, target, TOK, k=2)

    def test_generate_returns_string(self):
        d   = self._decoder()
        out = d.generate("abcd", max_new_tokens=4)
        assert isinstance(out, str)

    def test_acceptance_rate_range(self):
        d = self._decoder()
        d.generate("abc", max_new_tokens=4)
        assert 0.0 <= d.acceptance_rate <= 1.0

    def test_reset_stats(self):
        d = self._decoder()
        d.generate("abc", max_new_tokens=4)
        d.reset_stats()
        assert d.acceptance_rate == 0.0

    def test_output_length_bounded(self):
        d   = self._decoder()
        out = d.generate("ab", max_new_tokens=5)
        assert len(out) <= 5 * 3  # generous bound for multi-char tokens


# ── LayerCache ────────────────────────────────────────────────────────────────

class TestLayerCache:
    def test_append_updates_seq_len(self):
        lc = LayerCache(k=torch.zeros(1, 2, 8, 4), v=torch.zeros(1, 2, 8, 4))
        k  = torch.randn(1, 2, 3, 4)
        v  = torch.randn(1, 2, 3, 4)
        lc.append(k, v)
        assert lc.seq_len == 3

    def test_get_after_append(self):
        lc = LayerCache(k=torch.zeros(1, 2, 8, 4), v=torch.zeros(1, 2, 8, 4))
        k  = torch.ones(1, 2, 2, 4)
        v  = torch.ones(1, 2, 2, 4) * 2
        lc.append(k, v)
        ck, cv = lc.get()
        assert torch.allclose(ck, k)
        assert torch.allclose(cv, v)

    def test_reset_seq_len_zero(self):
        lc = LayerCache(k=torch.zeros(1, 2, 8, 4), v=torch.zeros(1, 2, 8, 4))
        lc.append(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        lc.reset()
        assert lc.seq_len == 0


# ── Multi-layer KVCache ───────────────────────────────────────────────────────

class TestMultiLayerKVCache:
    def test_all_layers_updated(self):
        cfg = CacheConfig(n_layers=3, n_heads=2, d_head=4, max_seq_len=16)
        cache = KVCache(cfg)
        for layer in range(3):
            cache.append(layer, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        for layer in range(3):
            k, v = cache.get(layer)
            assert k.shape[2] == 4

    def test_reset_specific_layer(self):
        cfg = CacheConfig(n_layers=2, n_heads=2, d_head=4, max_seq_len=16)
        cache = KVCache(cfg)
        cache.append(0, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        cache.append(1, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        cache.reset(layer=0)
        k0, _ = cache.get(0)
        k1, _ = cache.get(1)
        assert k0.shape[2] == 0    # layer 0 reset
        assert k1.shape[2] == 4   # layer 1 intact


# ── CacheManager touch / LRU order ───────────────────────────────────────────

class TestCacheManagerTouch:
    def test_touch_prevents_eviction(self):
        mgr = CacheManager(tiny_cfg(), max_requests=2)
        mgr.get_or_create("r1")
        mgr.get_or_create("r2")
        mgr.touch("r1")            # r1 is now MRU
        mgr.get_or_create("r3")   # should evict r2 (LRU), not r1
        assert "r1" in mgr._caches
        assert "r2" not in mgr._caches

    def test_reset_all(self):
        mgr = CacheManager(tiny_cfg(), max_requests=3)
        c   = mgr.get_or_create("r1")
        c.append(0, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        mgr.reset_all()
        assert c.seq_len == 0

    def test_total_memory_in_stats(self):
        mgr = CacheManager(tiny_cfg(), max_requests=3)
        mgr.get_or_create("r1")
        stats = mgr.stats()
        assert "total_mem_mb" in stats
        assert stats["total_mem_mb"] > 0.0


# ── PrefixCache stats ─────────────────────────────────────────────────────────

class TestPrefixCacheStats:
    def test_stats_keys(self):
        pc    = PrefixCache(tiny_cfg())
        stats = pc.stats()
        for k in ("n_cached", "max", "hits", "misses", "hit_rate"):
            assert k in stats

    def test_zero_hit_rate_initially(self):
        pc = PrefixCache(tiny_cfg())
        assert pc.hit_rate == 0.0

    def test_n_cached_increments(self):
        pc = PrefixCache(tiny_cfg(), max_prefixes=10)
        pc.store("a", KVCache(tiny_cfg()))
        pc.store("b", KVCache(tiny_cfg()))
        assert pc.stats()["n_cached"] == 2

    def test_same_prefix_not_duplicated(self):
        pc = PrefixCache(tiny_cfg(), max_prefixes=10)
        kv = KVCache(tiny_cfg())
        pc.store("same", kv)
        pc.store("same", KVCache(tiny_cfg()))   # overwrite
        assert pc.stats()["n_cached"] == 1
