"""
nanomind/cache/manager.py — Cache manager for multi-sequence serving.

In production serving, multiple users send requests simultaneously.
The CacheManager maintains one KVCache per active request_id
and evicts the least-recently-used cache when capacity is full.

This is a simplified version of vLLM's PagedAttention concept:
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
