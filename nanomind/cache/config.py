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
