"""
nanomind/cache/prefix_cache.py — Prefix / prompt caching.

Prompt caching (used by GPT-4, Claude, Gemini) stores KV-cache for
reused prompt prefixes across multiple requests.

Example:
  Request 1: system_prompt + "What is Python?"
  Request 2: system_prompt + "What is Rust?"

  The system_prompt is identical in both → compute K/V once, reuse!

This is the same idea as:
  - Anthropic's prompt caching feature (2024)
  - OpenAI's prompt caching feature (2024)
  - Google Gemini's context caching

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
