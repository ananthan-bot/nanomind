"""
nanomind/attention/kv_cache.py — Key-Value cache for fast autoregressive inference.

During generation, the model runs one token at a time. Without a KV-cache,
the full past key/value tensors would be recomputed every step (O(n^2) time).
With a KV-cache we store past K and V and only compute attention for the
new token against the full history.
"""

from __future__ import annotations

import torch


class KVCache:
    """
    Fixed-capacity key-value cache for one attention layer.

    Grows token-by-token up to ``max_seq_len``. After that, older
    entries are evicted (sliding window).

    Args:
        max_seq_len: Maximum number of past tokens to cache.
        n_heads:     Number of attention heads.
        head_dim:    Dimension of each head.
        device:      Device to allocate tensors on.
        dtype:       Data type for cache tensors.
    """

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.device      = device
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Current number of cached tokens."""
        return 0 if self._k is None else self._k.size(2)

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value vectors and return the full cached tensors.

        Args:
            new_k: New keys   ``(B, n_heads, T_new, head_dim)``
            new_v: New values ``(B, n_heads, T_new, head_dim)``

        Returns:
            ``(cached_k, cached_v)`` — full past + new tensors.
        """
        if self._k is None:
            self._k = new_k
            self._v = new_v
        else:
            self._k = torch.cat([self._k, new_k], dim=2)
            self._v = torch.cat([self._v, new_v], dim=2)

        # Evict oldest if over capacity
        if self._k.size(2) > self.max_seq_len:
            self._k = self._k[:, :, -self.max_seq_len:, :]
            self._v = self._v[:, :, -self.max_seq_len:, :]

        return self._k, self._v

    def reset(self) -> None:
        """Clear the cache (call between independent generation runs)."""
        self._k = None
        self._v = None

    def __repr__(self) -> str:
        return (
            f"KVCache(length={self.length}/{self.max_seq_len}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim})"
        )
