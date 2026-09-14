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
