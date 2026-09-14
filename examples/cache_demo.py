"""
examples/cache_demo.py — NanoMind KV-Cache & Fast Inference demo.

Demonstrates:
  1. KVCache creation, append, get, stats
  2. CacheManager multi-request LRU eviction
  3. PrefixCache prompt caching + hit/miss tracking
  4. CachedInferenceEngine benchmark
  5. SpeculativeDecoder draft+verify

Usage:
    python examples/cache_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.cache import (
    CacheConfig, KVCache, LayerCache,
    CacheManager, PrefixCache,
    CachedInferenceEngine,
    SpeculativeDecoder,
)

# ── Minimal model + tokenizer ─────────────────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V, D=32, T=32):
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

CORPUS = "nanomind cache kv fast inference speculative"
tok    = CharTok(CORPUS)
model  = TinyLM(tok.vocab_size)

print("=" * 55)
print("NanoMind KV-Cache & Fast Inference Demo")
print("=" * 55)

# ── KVCache ───────────────────────────────────────────────────────────────────
print("
── KVCache ──")
cfg   = CacheConfig(n_layers=2, n_heads=2, d_head=8, max_seq_len=32)
cache = KVCache(cfg)
print(f"  Estimated memory: {cfg.memory_mb:.4f} MB")
k = torch.randn(1, 2, 4, 8)
v = torch.randn(1, 2, 4, 8)
cache.append(0, k, v)
cached_k, cached_v = cache.get(0)
print(f"  Appended 4 tokens → cached shape: {tuple(cached_k.shape)}")
print(f"  Stats: {cache.stats()}")

# ── CacheManager ──────────────────────────────────────────────────────────────
print("
── CacheManager (LRU, max=3) ──")
mgr = CacheManager(cfg, max_requests=3)
for i in range(4):
    c = mgr.get_or_create(f"req-{i:03d}")
print(f"  Active caches: {mgr.active_count}")  # oldest evicted
print(f"  Stats: {mgr.stats()}")
mgr.release("req-001")
print(f"  After release: {mgr.active_count}")

# ── PrefixCache ───────────────────────────────────────────────────────────────
print("
── PrefixCache ──")
pc     = PrefixCache(cfg, max_prefixes=4)
prefix = "You are a helpful NanoMind assistant."
pc.store(prefix, KVCache(cfg))
hit  = pc.lookup(prefix)
miss = pc.lookup("completely different prefix text")
print(f"  Hit:  {hit is not None}")
print(f"  Miss: {miss is None}")
print(f"  Stats: {pc.stats()}")

# ── CachedInferenceEngine benchmark ──────────────────────────────────────────
print("
── Inference Benchmark ──")
engine = CachedInferenceEngine(model, tok)
bench  = engine.benchmark("nanomind", max_new_tokens=10)
print(f"  {bench}")
text = engine.generate("cache", max_new_tokens=8, temperature=0.8)
print(f"  Generated: {text!r}")

# ── SpeculativeDecoder ────────────────────────────────────────────────────────
print("
── Speculative Decoding ──")
small  = TinyLM(tok.vocab_size, D=16)   # draft (tiny)
large  = TinyLM(tok.vocab_size, D=32)   # target (bigger)
decoder = SpeculativeDecoder(small, large, tok, k=3)
result  = decoder.generate("nanomind", max_new_tokens=10)
print(f"  Output: {result!r}")
print(f"  Acceptance rate: {decoder.acceptance_rate:.1%}")
print("
Cache demo complete!")
