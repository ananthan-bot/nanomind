"""
nanomind/cache/layer_cache.py — Per-layer KV cache storage.

Each transformer layer gets one LayerKVCache that holds pre-allocated
key and value tensors. Tokens are appended to the cache one step at a time.

Memory layout:
  k_cache : (max_batch, max_seq_len, n_heads, head_dim)
  v_cache : (max_batch, max_seq_len, n_heads, head_dim)

The ``seq_len`` pointer tracks how many tokens have been written.
"""

from __future__ import annotations

import torch


class LayerKVCache:
    """
    Key-Value cache for a single transformer attention layer.

    Pre-allocates fixed-size K and V tensors and fills them incrementally
    during autoregressive decoding.

    Args:
        max_batch_size: Maximum batch size.
        max_seq_len:    Maximum sequence length (prompt + max new tokens).
        n_heads:        Number of KV heads.
        head_dim:       Dimension per attention head.
        dtype:          Storage dtype.
        device:         Storage device.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len:    int,
        n_heads:        int,
        head_dim:       int,
        dtype:          torch.dtype = torch.float32,
        device:         torch.device | str = "cpu",
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self._len        = 0   # tokens written so far

        shape = (max_batch_size, max_seq_len, n_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V vectors and return the full accumulated cache.

        Args:
            k_new: New key tensor ``(B, T_new, n_heads, head_dim)``.
            v_new: New value tensor ``(B, T_new, n_heads, head_dim)``.

        Returns:
            Tuple of ``(k_full, v_full)`` — the complete K/V cache up to
            the current step ``(B, current_len, n_heads, head_dim)``.
        """
        B, T_new, H, D = k_new.shape
        assert self._len + T_new <= self.max_seq_len, (
            f"KV cache overflow: {self._len} + {T_new} > {self.max_seq_len}"
        )
        self.k_cache[:B, self._len:self._len + T_new] = k_new
        self.v_cache[:B, self._len:self._len + T_new] = v_new
        self._len += T_new

        k_out = self.k_cache[:B, :self._len]
        v_out = self.v_cache[:B, :self._len]
        return k_out, v_out

    def reset(self) -> None:
        """Clear the cache (zero fill + reset pointer)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._len = 0

    @property
    def current_len(self) -> int:
        return self._len

    @property
    def is_empty(self) -> bool:
        return self._len == 0

    @property
    def is_full(self) -> bool:
        return self._len >= self.max_seq_len

    def memory_bytes(self) -> int:
        return self.k_cache.nbytes + self.v_cache.nbytes

    def __repr__(self) -> str:
        return (
            f"LayerKVCache("
            f"len={self._len}/{self.max_seq_len}, "
            f"heads={self.n_heads}, dim={self.head_dim})"
        )
