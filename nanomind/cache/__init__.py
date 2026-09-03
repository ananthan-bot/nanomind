"""NanoMind KV Cache sub-package for fast autoregressive inference.

KV cache is the single most impactful inference optimization for LLMs:
instead of recomputing all past K/V tensors at every decode step (O(T²)),
we store them and only compute the NEW token's K/V (O(1) per step).

Primary exports:
    - :class:`CachedGenerator`      — high-level generate() with KV cache
    - :class:`NanoMindCached`       — transformer with prefill() + decode_step()
    - :class:`KVCacheManager`       — multi-layer cache manager
    - :class:`LayerKVCache`         — per-layer K/V storage with update()
    - :class:`KVCacheConfig`        — cache configuration and size estimation
    - :class:`CachedSelfAttention`  — cache-aware attention module
    - :class:`CachedTransformerBlock` — transformer block with cache passthrough
    - :func:`print_cache_report`    — pretty-print utilisation stats
    - :func:`estimate_cache_memory` — pre-allocation memory estimate
"""

from nanomind.cache.config import KVCacheConfig
from nanomind.cache.layer_cache import LayerKVCache
from nanomind.cache.manager import KVCacheManager
from nanomind.cache.attention import CachedSelfAttention
from nanomind.cache.block import CachedTransformerBlock
from nanomind.cache.model import NanoMindCached
from nanomind.cache.generator import CachedGenerator
from nanomind.cache.stats import print_cache_report, estimate_cache_memory

__all__ = [
    "KVCacheConfig",
    "LayerKVCache",
    "KVCacheManager",
    "CachedSelfAttention",
    "CachedTransformerBlock",
    "NanoMindCached",
    "CachedGenerator",
    "print_cache_report",
    "estimate_cache_memory",
]
