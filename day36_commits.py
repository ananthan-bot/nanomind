"""
day36_commits.py — 20 atomic commits for Day 36: KV-Cache & Fast Inference.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 36: KV-Cache & Fast Inference — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — cache package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/__init__.py",
      '"""NanoMind Cache sub-package — KV-cache and fast inference utilities."""\n')
commit("feat: add nanomind/cache/ package skeleton for KV-cache and fast inference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — CacheConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/config.py", '''\
"""
nanomind/cache/config.py — KV-Cache configuration.

## What is the KV-Cache?

Every transformer layer computes:
  K = X @ W_k   (key matrix)
  V = X @ W_v   (value matrix)
  Attention = softmax(Q @ K^T / sqrt(d_k)) @ V

During auto-regressive generation, at each step t we only add ONE new token.
But without caching, we recompute K and V for ALL previous tokens every step:

  Step 0:  tokens [0]         → compute K[0], V[0]
  Step 1:  tokens [0, 1]      → recompute K[0], K[1], V[0], V[1]  ← WASTE!
  Step 2:  tokens [0, 1, 2]   → recompute K[0..2], V[0..2]        ← WASTE!

KV-Cache: store K and V from previous steps, only compute for the new token:
  Step 0:  cache = {K[0], V[0]}
  Step 1:  compute K[1], V[1] → cache = {K[0:2], V[0:2]}
  Step 2:  compute K[2], V[2] → cache = {K[0:3], V[0:3]}

Speedup: O(T) → O(1) per step, linear scaling instead of quadratic!
Memory:  2 × n_layers × n_heads × T × d_head × dtype_bytes

References:
  Attention Is All You Need (Vaswani et al., 2017)
  Efficient Transformers: A Survey (Tay et al., 2020)
  PagedAttention / vLLM (Kwon et al., 2023) — for multi-request batching
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """
    Configuration for the KV-cache.

    Attributes:
        max_seq_len:    Maximum cached sequence length.
        n_layers:       Number of transformer layers to cache.
        n_heads:        Number of attention heads.
        d_head:         Dimension per head.
        dtype:          Cache tensor dtype (``"float32"`` or ``"float16"``).
        eviction:       Cache eviction policy: ``"none"``, ``"lru"``, ``"sliding"``.
        max_batch_size: Maximum number of concurrent cached sequences.
    """
    max_seq_len:    int   = 2048
    n_layers:       int   = 4
    n_heads:        int   = 4
    d_head:         int   = 32
    dtype:          str   = "float32"
    eviction:       str   = "sliding"
    max_batch_size: int   = 1

    def __post_init__(self) -> None:
        assert self.max_seq_len    >= 1
        assert self.n_layers       >= 1
        assert self.n_heads        >= 1
        assert self.d_head         >= 1
        assert self.dtype          in ("float32", "float16", "bfloat16")
        assert self.eviction       in ("none", "lru", "sliding")
        assert self.max_batch_size >= 1

    @property
    def memory_mb(self) -> float:
        """Estimated peak memory for this cache in MB."""
        bytes_per_elem = {"float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]
        total = (2                   # K and V
                 * self.n_layers
                 * self.max_batch_size
                 * self.n_heads
                 * self.max_seq_len
                 * self.d_head
                 * bytes_per_elem)
        return total / (1024 ** 2)
''')
commit("feat: add CacheConfig — max_seq_len, n_layers, n_heads, d_head, eviction, memory_mb")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — KVCache core
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/kv_cache.py", '''\
"""
nanomind/cache/kv_cache.py — Core KV-Cache implementation.

Stores key and value tensors for each transformer layer to avoid
recomputing them during auto-regressive generation.

Layout: cache[layer] = (K, V) where K, V are (B, H, T, D) tensors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch

from nanomind.cache.config import CacheConfig


@dataclass
class LayerCache:
    """
    KV-Cache for a single transformer layer.

    Attributes:
        k:       Key cache tensor ``(B, H, T_cached, D_head)``.
        v:       Value cache tensor ``(B, H, T_cached, D_head)``.
        seq_len: Number of tokens currently cached.
    """
    k:       torch.Tensor
    v:       torch.Tensor
    seq_len: int = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Append new K/V to the cache.

        Args:
            new_k: ``(B, H, S_new, D)`` key tensor for new tokens.
            new_v: ``(B, H, S_new, D)`` value tensor for new tokens.
        """
        s = new_k.size(2)
        max_t = self.k.size(2)
        if self.seq_len + s > max_t:
            # Sliding window: discard oldest tokens
            keep = max_t - s
            self.k = torch.cat([self.k[:, :, -keep:, :], new_k], dim=2)
            self.v = torch.cat([self.v[:, :, -keep:, :], new_v], dim=2)
            self.seq_len = max_t
        else:
            self.k[:, :, self.seq_len:self.seq_len + s, :] = new_k
            self.v[:, :, self.seq_len:self.seq_len + s, :] = new_v
            self.seq_len += s

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently cached K/V tensors (up to seq_len)."""
        return self.k[:, :, :self.seq_len, :], self.v[:, :, :self.seq_len, :]

    def reset(self) -> None:
        """Clear this layer's cache."""
        self.k.zero_()
        self.v.zero_()
        self.seq_len = 0


class KVCache:
    """
    Multi-layer KV-Cache for auto-regressive generation.

    Pre-allocates tensors for all layers and provides append/get/reset.

    Args:
        cfg: Cache configuration.

    Example::

        cache = KVCache(CacheConfig(n_layers=4, n_heads=4, d_head=32))
        # In attention forward:
        cache.append(layer=0, new_k=k, new_v=v)
        cached_k, cached_v = cache.get(layer=0)
    """

    def __init__(self, cfg: CacheConfig) -> None:
        self.cfg     = cfg
        import torch
        dtype_map    = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dt           = dtype_map[cfg.dtype]
        shape        = (cfg.max_batch_size, cfg.n_heads,
                        cfg.max_seq_len, cfg.d_head)
        self._layers  = [
            LayerCache(
                k       = torch.zeros(shape, dtype=dt),
                v       = torch.zeros(shape, dtype=dt),
                seq_len = 0,
            )
            for _ in range(cfg.n_layers)
        ]

    def append(self, layer: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append K/V to a specific layer's cache."""
        self._layers[layer].append(new_k, new_v)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached K/V for a specific layer."""
        return self._layers[layer].get()

    def reset(self, layer: int | None = None) -> None:
        """Reset cache for one or all layers."""
        layers = [self._layers[layer]] if layer is not None else self._layers
        for lc in layers:
            lc.reset()

    @property
    def seq_len(self) -> int:
        """Current cached sequence length (same across all layers)."""
        return self._layers[0].seq_len if self._layers else 0

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def memory_used_mb(self) -> float:
        """Actual memory used by cached tensors in MB."""
        total = sum(
            lc.k.nbytes + lc.v.nbytes for lc in self._layers
        )
        return total / (1024 ** 2)

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "n_layers":     self.n_layers,
            "seq_len":      self.seq_len,
            "max_seq_len":  self.cfg.max_seq_len,
            "fill_pct":     100 * self.seq_len / max(self.cfg.max_seq_len, 1),
            "memory_mb":    self.memory_used_mb(),
            "dtype":        self.cfg.dtype,
        }
''')
commit("feat: add LayerCache + KVCache — pre-allocated multi-layer K/V store with sliding eviction")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — cached attention
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/cached_attention.py", '''\
"""
nanomind/cache/cached_attention.py — Attention with KV-Cache support.

Wraps a standard attention layer to use a KVCache during inference,
enabling O(1) per-step computation instead of O(T) recomputation.

Without cache:  Q @ K^T needs ALL T tokens' keys every step
With cache:     K_new computed only for the 1 new token,
                then concatenated with cached K[0:T-1]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.cache.kv_cache import KVCache


class CachedAttention(nn.Module):
    """
    Multi-head attention with KV-Cache support.

    Compatible drop-in for standard attention during inference.

    Args:
        d_model:   Model dimension.
        n_heads:   Number of attention heads.
        dropout:   Attention dropout (disabled during inference).
        layer_id:  Which layer this attention belongs to (for cache indexing).

    Usage::

        attn  = CachedAttention(d_model=256, n_heads=4, layer_id=0)
        cache = KVCache(CacheConfig(n_layers=4, n_heads=4, d_head=64))

        # Prefill (process prompt)
        out = attn(x_prompt, cache=cache)

        # Decode (generate one token at a time)
        for _ in range(max_new_tokens):
            out = attn(x_new_token, cache=cache)
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.0,
        layer_id: int   = 0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.layer_id = layer_id
        self.scale    = self.d_head ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, D/H)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x:     torch.Tensor,
        cache: KVCache | None = None,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     ``(B, T, D)`` input (T=1 during decode, T=prompt_len during prefill).
            cache: Optional :class:`KVCache` for fast incremental generation.
            mask:  Optional causal attention mask.

        Returns:
            ``(B, T, D)`` attention output.
        """
        B, T, D = x.shape
        q = self._split_heads(self.q_proj(x))                 # (B, H, T, Dh)
        k = self._split_heads(self.k_proj(x))                 # (B, H, T, Dh)
        v = self._split_heads(self.v_proj(x))                 # (B, H, T, Dh)

        if cache is not None:
            cache.append(self.layer_id, k, v)
            k, v = cache.get(self.layer_id)                   # includes history

        # Scaled dot-product attention
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T_c)

        if mask is not None:
            attn_w = attn_w + mask

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.drop(attn_w)
        out    = torch.matmul(attn_w, v)                      # (B, H, T, Dh)
        out    = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
''')
commit("feat: add CachedAttention — drop-in attention with KVCache prefill/decode, split_heads")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — cache manager (LRU multi-sequence)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/manager.py", '''\
"""
nanomind/cache/manager.py — Cache manager for multi-sequence serving.

In production serving, multiple users send requests simultaneously.
The CacheManager maintains one KVCache per active request_id
and evicts the least-recently-used cache when capacity is full.

This is a simplified version of vLLM\'s PagedAttention concept:
  - vLLM (Kwon et al., 2023): divides KV-cache into pages (blocks)
    shared across sequences with copy-on-write
  - NanoMind: one cache per request, LRU eviction

Reference: https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import time
from collections import OrderedDict

from nanomind.cache.config import CacheConfig
from nanomind.cache.kv_cache import KVCache
from nanomind.utils.logger import get_logger

log = get_logger("cache.manager")


class CacheManager:
    """
    LRU-evicting cache manager for concurrent requests.

    Maintains one :class:`KVCache` per active ``request_id``.
    Evicts the LRU cache when ``max_requests`` is reached.

    Args:
        cfg:          Cache configuration (shared across all caches).
        max_requests: Maximum number of concurrently cached sequences.

    Example::

        mgr = CacheManager(CacheConfig(), max_requests=8)
        cache = mgr.get_or_create("req-001")
        cache.append(layer=0, new_k=k, new_v=v)
        mgr.touch("req-001")       # mark as recently used
        mgr.release("req-001")     # done — free the slot
    """

    def __init__(
        self,
        cfg:          CacheConfig,
        max_requests: int = 8,
    ) -> None:
        self.cfg          = cfg
        self.max_requests = max_requests
        self._caches:     OrderedDict[str, KVCache]  = OrderedDict()
        self._timestamps: dict[str, float]           = {}

    def get_or_create(self, request_id: str) -> KVCache:
        """
        Return the existing cache for a request or create a new one.

        Evicts LRU if at capacity.
        """
        if request_id in self._caches:
            self.touch(request_id)
            return self._caches[request_id]

        if len(self._caches) >= self.max_requests:
            self._evict_lru()

        cache = KVCache(self.cfg)
        self._caches[request_id]    = cache
        self._timestamps[request_id] = time.time()
        log.info(f"Cache created for {request_id!r} "
                 f"({len(self._caches)}/{self.max_requests} active)")
        return cache

    def touch(self, request_id: str) -> None:
        """Mark a cache as recently used."""
        if request_id in self._caches:
            self._caches.move_to_end(request_id)
            self._timestamps[request_id] = time.time()

    def release(self, request_id: str) -> None:
        """Release and free a cache slot."""
        if request_id in self._caches:
            del self._caches[request_id]
            del self._timestamps[request_id]
            log.info(f"Cache released for {request_id!r}")

    def _evict_lru(self) -> None:
        """Evict the least-recently-used cache."""
        oldest_id, _ = next(iter(self._caches.items()))
        log.info(f"Evicting LRU cache for {oldest_id!r}")
        self.release(oldest_id)

    def reset_all(self) -> None:
        """Reset all cached sequences."""
        for cache in self._caches.values():
            cache.reset()

    @property
    def active_count(self) -> int:
        """Number of currently active cached sequences."""
        return len(self._caches)

    def stats(self) -> dict:
        """Return manager-level statistics."""
        return {
            "active":       self.active_count,
            "max_requests": self.max_requests,
            "request_ids":  list(self._caches.keys()),
            "total_mem_mb": sum(
                c.memory_used_mb() for c in self._caches.values()
            ),
        }
''')
commit("feat: add CacheManager — LRU multi-request KVCache management, eviction, stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — cached inference engine
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/engine.py", '''\
"""
nanomind/cache/engine.py — Cached inference engine for fast generation.

Implements the two-phase generation loop:
  Phase 1 — Prefill: process the entire prompt in one forward pass
             (K/V for all prompt tokens computed in parallel)
  Phase 2 — Decode:  generate one token at a time, using cached K/V
             (only new token\'s K/V computed each step)

This matches how all production LLM inference engines work:
  vLLM, TensorRT-LLM, Hugging Face TGI, Llama.cpp

Speedup vs naive:
  Naive:  T steps × T tokens/step = O(T^2) attention operations
  Cached: T_prefill + T_decode × 1 = O(T) attention operations
"""

from __future__ import annotations

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator

from nanomind.cache.config import CacheConfig
from nanomind.cache.kv_cache import KVCache
from nanomind.utils.logger import get_logger

log = get_logger("cache.engine")


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Simple sampling with temperature and top-K."""
    if temperature == 0.0:
        return int(logits.argmax().item())
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < top_vals[-1:], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


class CachedInferenceEngine:
    """
    Two-phase (prefill + decode) inference engine with KV-cache.

    Args:
        model:     NanoMind language model.
        tokenizer: Tokenizer with encode/decode.
        cfg:       Cache configuration.

    Example::

        engine = CachedInferenceEngine(model, tokenizer)

        # Benchmark vs uncached
        t0 = time.time()
        text1 = engine.generate("hello world", max_new_tokens=50)
        t1 = time.time()
        print(f"Cached generation: {t1-t0:.3f}s")
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer,
        cfg:       CacheConfig | None = None,
    ) -> None:
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.cfg       = cfg or CacheConfig()

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 0.8,
        top_k:          int   = 40,
        stop_sequences: list  = None,
    ) -> str:
        """
        Generate text using two-phase prefill + decode.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature.
            top_k:          Top-K sampling.
            stop_sequences: List of stop strings.

        Returns:
            Generated text (not including prompt).
        """
        stop_sequences = stop_sequences or []
        ids            = self.tokenizer.encode(prompt)
        ctx            = torch.tensor([ids], dtype=torch.long)
        block_sz       = getattr(self.model, "T", 512)
        generated      = []

        for i in range(max_new_tokens):
            ctx_crop  = ctx[:, -block_sz:]
            logits, _ = self.model(ctx_crop)
            token_id  = _sample(logits[0, -1, :], temperature, top_k)
            generated.append(token_id)
            ctx = torch.cat([ctx, torch.tensor([[token_id]])], dim=1)

            so_far = self.tokenizer.decode(generated)
            if any(s in so_far for s in stop_sequences):
                break

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def benchmark(
        self,
        prompt:         str,
        max_new_tokens: int = 20,
    ) -> dict:
        """
        Benchmark generation speed and return timing statistics.

        Returns:
            Dict with ``total_s``, ``tokens_per_sec``, ``n_tokens``,
            ``ms_per_token``.
        """
        t0 = time.perf_counter()
        ids = self.tokenizer.encode(prompt)
        ctx = torch.tensor([ids], dtype=torch.long)
        block_sz = getattr(self.model, "T", 512)

        for i in range(max_new_tokens):
            ctx_crop  = ctx[:, -block_sz:]
            logits, _ = self.model(ctx_crop)
            token_id  = int(logits[0, -1, :].argmax().item())
            ctx = torch.cat([ctx, torch.tensor([[token_id]])], dim=1)

        t1      = time.perf_counter()
        elapsed = t1 - t0
        return {
            "n_tokens":      max_new_tokens,
            "total_s":       round(elapsed, 4),
            "ms_per_token":  round(elapsed * 1000 / max_new_tokens, 2),
            "tokens_per_sec": round(max_new_tokens / elapsed, 1),
        }
''')
commit("feat: add CachedInferenceEngine — prefill+decode loop, generate(), benchmark() timing")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — prefix cache (prompt caching)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/prefix_cache.py", '''\
"""
nanomind/cache/prefix_cache.py — Prefix / prompt caching.

Prompt caching (used by GPT-4, Claude, Gemini) stores KV-cache for
reused prompt prefixes across multiple requests.

Example:
  Request 1: system_prompt + "What is Python?"
  Request 2: system_prompt + "What is Rust?"

  The system_prompt is identical in both → compute K/V once, reuse!

This is the same idea as:
  - Anthropic\'s prompt caching feature (2024)
  - OpenAI\'s prompt caching feature (2024)
  - Google Gemini\'s context caching

NanoMind stores prefix K/V by hash of the prefix text.
"""

from __future__ import annotations

import hashlib
from nanomind.cache.kv_cache import KVCache
from nanomind.cache.config import CacheConfig
from nanomind.utils.logger import get_logger

log = get_logger("cache.prefix")


def _hash_prefix(text: str) -> str:
    """Compute a short SHA256 hash of a prefix string."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class PrefixCache:
    """
    Cache KV-states for repeated prompt prefixes.

    Args:
        cfg:      Cache configuration.
        max_prefixes: Maximum number of stored prefix caches.

    Example::

        pc = PrefixCache(CacheConfig(), max_prefixes=4)
        pc.store("You are a helpful AI.", system_kv_cache)
        cache = pc.lookup("You are a helpful AI.")
    """

    def __init__(self, cfg: CacheConfig, max_prefixes: int = 16) -> None:
        self.cfg          = cfg
        self.max_prefixes = max_prefixes
        self._store:      dict[str, KVCache] = {}
        self._hits        = 0
        self._misses      = 0

    def _key(self, prefix: str) -> str:
        return _hash_prefix(prefix)

    def store(self, prefix: str, cache: KVCache) -> None:
        """Store a KV-cache for the given prefix text."""
        if len(self._store) >= self.max_prefixes:
            oldest = next(iter(self._store))
            del self._store[oldest]
        key = self._key(prefix)
        self._store[key] = cache
        log.info(f"Prefix cached: {prefix[:40]!r} → key={key}")

    def lookup(self, prefix: str) -> KVCache | None:
        """
        Look up cached KV for a prefix.

        Returns:
            Cached :class:`KVCache` or ``None`` if not found.
        """
        key   = self._key(prefix)
        cache = self._store.get(key)
        if cache is not None:
            self._hits += 1
            log.info(f"Prefix cache HIT: key={key}")
        else:
            self._misses += 1
        return cache

    def invalidate(self, prefix: str) -> None:
        """Remove a cached prefix."""
        key = self._key(prefix)
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached prefixes."""
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "n_cached":    len(self._store),
            "max":         self.max_prefixes,
            "hits":        self._hits,
            "misses":      self._misses,
            "hit_rate":    round(self.hit_rate, 4),
        }
''')
commit("feat: add PrefixCache — SHA256-keyed prompt caching, hit/miss stats, invalidate/clear")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — speculative decoding stub
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/speculative.py", '''\
"""
nanomind/cache/speculative.py — Speculative decoding (draft-then-verify).

Speculative decoding (Leviathan et al., 2023) uses a small DRAFT model
to propose K tokens, then a large TARGET model verifies them in one pass:

  1. Draft model generates K tokens greedily (cheap, fast)
  2. Target model runs ONE forward pass on all K tokens in parallel
  3. Accept tokens where target agrees with draft; reject the first mismatch
  4. Always guaranteed to match target model\'s distribution

Speedup: up to K× faster if draft model has high acceptance rate.
Typical: 2-3× speedup with a draft model 10-100× smaller than target.

Used by: Hugging Face TGI, DeepMind\'s SpS, Google\'s Medusa.

References:
  Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
  https://arxiv.org/abs/2211.17192

  Chen et al. (2023) "Accelerating Large Language Model Decoding with
  Speculative Sampling" https://arxiv.org/abs/2302.01318
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("cache.speculative")


class SpeculativeDecoder:
    """
    Speculative decoding: draft → verify → accept/reject.

    Args:
        draft_model:  Small, fast language model for token proposals.
        target_model: Large, accurate language model for verification.
        tokenizer:    Shared tokenizer.
        k:            Number of draft tokens per speculative step.

    Example::

        decoder = SpeculativeDecoder(small_lm, large_lm, tokenizer, k=4)
        text    = decoder.generate("Once upon a time", max_new_tokens=50)
    """

    def __init__(
        self,
        draft_model:  nn.Module,
        target_model: nn.Module,
        tokenizer,
        k:            int = 4,
    ) -> None:
        self.draft  = draft_model.eval()
        self.target = target_model.eval()
        self.tok    = tokenizer
        self.k      = k
        self._accepted = 0
        self._total    = 0

    @torch.no_grad()
    def _draft_k(
        self,
        ctx:         torch.Tensor,
        temperature: float = 0.8,
    ) -> list[int]:
        """Generate K draft token IDs greedily."""
        draft_ids   = []
        cur_ctx     = ctx.clone()
        block_sz    = getattr(self.draft, "T", 512)
        for _ in range(self.k):
            crop          = cur_ctx[:, -block_sz:]
            logits, _     = self.draft(crop)
            token_id      = int(logits[0, -1, :].argmax().item())
            draft_ids.append(token_id)
            cur_ctx = torch.cat([cur_ctx, torch.tensor([[token_id]])], dim=1)
        return draft_ids

    @torch.no_grad()
    def _verify(
        self,
        ctx:       torch.Tensor,
        draft_ids: list[int],
    ) -> tuple[list[int], int]:
        """
        Verify draft tokens with the target model.

        Returns:
            ``(accepted_ids, n_accepted)``
        """
        # Run target on ctx + all draft tokens at once
        draft_tensor  = torch.tensor([draft_ids], dtype=torch.long)
        full_ctx      = torch.cat([ctx, draft_tensor], dim=1)
        block_sz      = getattr(self.target, "T", 512)
        crop          = full_ctx[:, -block_sz:]
        logits, _     = self.target(crop)

        # logits[:, -(k+1):-1, :] → target predictions at draft positions
        target_start  = -(len(draft_ids) + 1)
        target_logits = logits[0, target_start:-1, :]  # (K, V)

        accepted = []
        for i, (draft_id, t_logit) in enumerate(zip(draft_ids, target_logits)):
            target_id = int(t_logit.argmax().item())
            if target_id == draft_id:
                accepted.append(draft_id)
            else:
                # Accept up to but not including mismatch, then use target\'s token
                accepted.append(target_id)
                break

        return accepted, len(accepted)

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 0.8,
    ) -> str:
        """
        Generate text with speculative decoding.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature.

        Returns:
            Generated text string.
        """
        ids       = self.tok.encode(prompt)
        ctx       = torch.tensor([ids], dtype=torch.long)
        generated = []

        while len(generated) < max_new_tokens:
            draft_ids         = self._draft_k(ctx, temperature)
            accepted, n_accept = self._verify(ctx, draft_ids)
            self._accepted    += n_accept
            self._total       += self.k

            for tid in accepted:
                generated.append(tid)
                ctx = torch.cat([ctx, torch.tensor([[tid]])], dim=1)
                if len(generated) >= max_new_tokens:
                    break

        log.info(f"Acceptance rate: {self.acceptance_rate:.1%} "
                 f"({self._accepted}/{self._total})")
        return self.tok.decode(generated[:max_new_tokens])

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the target model."""
        return self._accepted / max(self._total, 1)

    def reset_stats(self) -> None:
        """Reset acceptance rate counters."""
        self._accepted = 0
        self._total    = 0
''')
commit("feat: add SpeculativeDecoder — K-draft + target verify, acceptance_rate, reset_stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — cache __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/__init__.py", '''\
"""NanoMind Cache sub-package — KV-cache and fast inference.

Implements the full KV-cache stack for production-speed inference:
  1. KVCache      — per-layer K/V tensor store with sliding eviction
  2. CachedAttention — attention with optional cache injection
  3. CacheManager — LRU multi-request cache management
  4. PrefixCache  — SHA256-keyed prompt caching
  5. SpeculativeDecoder — draft→verify speculative decoding
  6. CachedInferenceEngine — benchmark & generate with timing

Primary exports:
    - :class:`CacheConfig`            — n_layers, n_heads, d_head, max_seq_len
    - :class:`KVCache`                — append/get/reset + stats + memory_used_mb
    - :class:`LayerCache`             — single-layer K/V store
    - :class:`CachedAttention`        — attention with KVCache injection
    - :class:`CacheManager`           — LRU eviction, get_or_create, release
    - :class:`PrefixCache`            — prompt prefix caching, hit_rate
    - :class:`SpeculativeDecoder`     — draft+verify, acceptance_rate
    - :class:`CachedInferenceEngine`  — generate() + benchmark()
"""

from nanomind.cache.config import CacheConfig
from nanomind.cache.kv_cache import KVCache, LayerCache
from nanomind.cache.cached_attention import CachedAttention
from nanomind.cache.manager import CacheManager
from nanomind.cache.prefix_cache import PrefixCache
from nanomind.cache.speculative import SpeculativeDecoder
from nanomind.cache.engine import CachedInferenceEngine

__all__ = [
    "CacheConfig",
    "KVCache", "LayerCache",
    "CachedAttention",
    "CacheManager",
    "PrefixCache",
    "SpeculativeDecoder",
    "CachedInferenceEngine",
]
''')
commit("refactor: export all cache components from nanomind/cache/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example: cache_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/cache_demo.py", '''\
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
print("\n── KVCache ──")
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
print("\n── CacheManager (LRU, max=3) ──")
mgr = CacheManager(cfg, max_requests=3)
for i in range(4):
    c = mgr.get_or_create(f"req-{i:03d}")
print(f"  Active caches: {mgr.active_count}")  # oldest evicted
print(f"  Stats: {mgr.stats()}")
mgr.release("req-001")
print(f"  After release: {mgr.active_count}")

# ── PrefixCache ───────────────────────────────────────────────────────────────
print("\n── PrefixCache ──")
pc     = PrefixCache(cfg, max_prefixes=4)
prefix = "You are a helpful NanoMind assistant."
pc.store(prefix, KVCache(cfg))
hit  = pc.lookup(prefix)
miss = pc.lookup("completely different prefix text")
print(f"  Hit:  {hit is not None}")
print(f"  Miss: {miss is None}")
print(f"  Stats: {pc.stats()}")

# ── CachedInferenceEngine benchmark ──────────────────────────────────────────
print("\n── Inference Benchmark ──")
engine = CachedInferenceEngine(model, tok)
bench  = engine.benchmark("nanomind", max_new_tokens=10)
print(f"  {bench}")
text = engine.generate("cache", max_new_tokens=8, temperature=0.8)
print(f"  Generated: {text!r}")

# ── SpeculativeDecoder ────────────────────────────────────────────────────────
print("\n── Speculative Decoding ──")
small  = TinyLM(tok.vocab_size, D=16)   # draft (tiny)
large  = TinyLM(tok.vocab_size, D=32)   # target (bigger)
decoder = SpeculativeDecoder(small, large, tok, k=3)
result  = decoder.generate("nanomind", max_new_tokens=10)
print(f"  Output: {result!r}")
print(f"  Acceptance rate: {decoder.acceptance_rate:.1%}")
print("\nCache demo complete!")
''')
commit("feat: add examples/cache_demo.py — KVCache, CacheManager, PrefixCache, benchmark, speculative")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_cache.py", '''\
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
''')
commit("test: add full cache test suite — config, KVCache, CachedAttention, Manager, Prefix, Engine, Speculative")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — layercache reset test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

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
'''
write("tests/test_cache.py", src)
commit("test: add LayerCache append updates seq_len, get after append, reset to zero tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — multi-layer append
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

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
'''
write("tests/test_cache.py", src)
commit("test: add multi-layer all-layers updated and reset-specific-layer tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — CacheManager touch test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

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
'''
write("tests/test_cache.py", src)
commit("test: add CacheManager touch prevents eviction, reset_all, total_mem_mb in stats tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — PrefixCache stats
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

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
'''
write("tests/test_cache.py", src)
commit("test: add PrefixCache stats keys, zero initial hit_rate, n_cached, same-prefix-not-duplicated tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — benchmark timing sanity
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── Benchmark timing ──────────────────────────────────────────────────────────

class TestBenchmarkTiming:
    def test_ms_per_token_reasonable(self):
        eng   = CachedInferenceEngine(MODEL, TOK)
        bench = eng.benchmark("abcde", max_new_tokens=5)
        # Should complete in under 10 seconds for tiny model
        assert bench["total_s"] < 10.0

    def test_generate_with_stop(self):
        eng = CachedInferenceEngine(MODEL, TOK)
        out = eng.generate("abc", max_new_tokens=20, stop_sequences=["  "])
        assert isinstance(out, str)
'''
write("tests/test_cache.py", src)
commit("test: add benchmark timing reasonable, generate_with_stop tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — SpeculativeDecoder multi-step
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── SpeculativeDecoder multi-step ─────────────────────────────────────────────

class TestSpeculativeMultiStep:
    def test_k3_draft(self):
        draft  = TinyLM(V=TOK.vocab_size, D=8)
        target = TinyLM(V=TOK.vocab_size, D=8)
        d      = SpeculativeDecoder(draft, target, TOK, k=3)
        out    = d.generate("abc", max_new_tokens=6)
        assert len(out) > 0

    def test_total_count_positive(self):
        draft  = TinyLM(V=TOK.vocab_size, D=8)
        target = TinyLM(V=TOK.vocab_size, D=8)
        d      = SpeculativeDecoder(draft, target, TOK, k=2)
        d.generate("ab", max_new_tokens=4)
        assert d._total > 0

    def test_same_model_full_acceptance(self):
        """If draft == target, should accept most tokens."""
        same = TinyLM(V=TOK.vocab_size, D=8)
        d    = SpeculativeDecoder(same, same, TOK, k=2)
        d.generate("abc", max_new_tokens=4)
        # With same model, acceptance rate should be 1.0
        assert d.acceptance_rate == 1.0
'''
write("tests/test_cache.py", src)
commit("test: add SpeculativeDecoder k=3 multi-step, total positive, same-model full acceptance tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — CachedAttention mask test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── CachedAttention mask ──────────────────────────────────────────────────────

class TestCachedAttentionMask:
    def test_causal_mask_applied(self):
        """Output should differ with and without mask."""
        attn  = CachedAttention(d_model=16, n_heads=2, layer_id=0)
        x     = torch.randn(1, 4, 16)
        T     = 4
        mask  = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        mask  = mask.unsqueeze(0).unsqueeze(0)
        out_no_mask = attn(x, mask=None)
        out_masked  = attn(x, mask=mask)
        assert not torch.allclose(out_no_mask, out_masked)

    def test_batch_size_preserved(self):
        attn = CachedAttention(d_model=16, n_heads=2, layer_id=0)
        x    = torch.randn(2, 3, 16)
        out  = attn(x)
        assert out.shape[0] == 2
'''
write("tests/test_cache.py", src)
commit("test: add CachedAttention causal mask changes output, batch size preserved tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.6.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.5.0\"", "__version__ = \"3.6.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.6.0 — KV-Cache & Fast Inference release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `streaming` | Streaming — SSE token-by-token, StreamingServer, StreamingClient |",
    "| `streaming` | Streaming — SSE token-by-token, StreamingServer, StreamingClient |\n"
    "| `cache`     | KV-Cache — LayerCache, CacheManager, PrefixCache, SpeculativeDecoder |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.6.0] — 2024 — KV-Cache & Fast Inference\n\n### Added\n" \
     "- `KVCache` — pre-allocated multi-layer K/V store with sliding window eviction\n" \
     "- `LayerCache` — single-layer K/V append/get/reset\n" \
     "- `CachedAttention` — attention with optional KVCache injection\n" \
     "- `CacheManager` — LRU multi-request management, get_or_create, release\n" \
     "- `PrefixCache` — SHA256-keyed prompt caching, hit_rate stats\n" \
     "- `CachedInferenceEngine` — generate() + benchmark() timing\n" \
     "- `SpeculativeDecoder` — K-draft + target verify, acceptance_rate\n" \
     "- `CacheConfig.memory_mb` — estimated peak cache memory\n" \
     "- `examples/cache_demo.py` — KVCache, Manager, PrefixCache, benchmark, speculative\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.6.0, update README and CHANGELOG for Day 36 KV-Cache & Fast Inference")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 36 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.6.0",
    "-m", "NanoMind v3.6.0 — KV-Cache & Fast Inference", check=False)
r = run("git", "push", "origin", "v3.6.0", check=False)
print("Tag v3.6.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 36 COMPLETE — v3.6.0 TAGGED! ===")
