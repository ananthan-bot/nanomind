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
