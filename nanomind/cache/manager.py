"""
nanomind/cache/manager.py — Multi-layer KV cache manager.

KVCacheManager owns one LayerKVCache per transformer layer and provides
a unified interface for the model to use during inference.
"""

from __future__ import annotations

import torch
from nanomind.cache.config import KVCacheConfig
from nanomind.cache.layer_cache import LayerKVCache


_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


class KVCacheManager:
    """
    Manages KV caches for all layers of a transformer model.

    Creates and owns one :class:`LayerKVCache` per transformer layer.
    Provides a simple ``get(layer_idx)`` interface and global reset.

    Args:
        cfg: KV cache configuration.

    Example::

        cache = KVCacheManager(KVCacheConfig(
            n_layers=6, n_heads=8, head_dim=64, max_seq_len=512
        ))

        # Inside attention forward (layer 3):
        k_full, v_full = cache.get(3).update(k_new, v_new)
    """

    def __init__(self, cfg: KVCacheConfig) -> None:
        self.cfg    = cfg
        dtype       = _DTYPE_MAP[cfg.dtype]
        device      = torch.device(cfg.device)

        self._caches: list[LayerKVCache] = [
            LayerKVCache(
                max_batch_size=cfg.max_batch_size,
                max_seq_len=cfg.max_seq_len,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(cfg.n_layers)
        ]

    def get(self, layer_idx: int) -> LayerKVCache:
        """Return the KV cache for a specific layer."""
        return self._caches[layer_idx]

    def reset(self) -> None:
        """Reset all layer caches (start of a new sequence)."""
        for c in self._caches:
            c.reset()

    @property
    def current_len(self) -> int:
        """Current sequence length (same for all layers)."""
        return self._caches[0].current_len if self._caches else 0

    def total_memory_bytes(self) -> int:
        """Total memory used by all K and V cache tensors."""
        return sum(c.memory_bytes() for c in self._caches)

    def total_memory_mb(self) -> float:
        return self.total_memory_bytes() / (1024 ** 2)

    def stats(self) -> dict:
        """Return cache utilisation statistics."""
        return {
            "n_layers":       len(self._caches),
            "current_len":    self.current_len,
            "max_seq_len":    self.cfg.max_seq_len,
            "fill_ratio":     self.current_len / max(self.cfg.max_seq_len, 1),
            "memory_mb":      self.total_memory_mb(),
            "config_mb":      self.cfg.cache_size_mb,
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheManager("
            f"layers={len(self._caches)}, "
            f"len={self.current_len}/{self.cfg.max_seq_len}, "
            f"mem={self.total_memory_mb():.1f}MB)"
        )
