"""
nanomind/cache/config.py — KV Cache configuration.

Without KV cache, autoregressive generation recomputes the full attention
over ALL previous tokens at every new step:

  Step t: attend over tokens [0 … t]    → O(t²) total work for T steps
  Step t: attend over tokens [0 … t-1]  → repeating all prior work!

With KV cache, we store the computed K and V tensors from past steps and
only compute attention for the NEW token against the cached K/V:

  Step t: compute K_t, V_t (new), attend against cached [K_0…K_{t-1}]
  Total: O(T) new computations, O(T·d) memory for the cache

This is the single most impactful optimization for LLM inference speed,
used in every production LLM serving system (vLLM, TGI, TensorRT-LLM).

References:
  Original attention: Vaswani et al. (2017)
  KV cache analysis:  Pope et al. (2022) — https://arxiv.org/abs/2211.05100
  PagedAttention:     Kwon et al. (2023) — https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """
    Configuration for KV Cache management.

    Attributes:
        max_batch_size:    Maximum batch size the cache is pre-allocated for.
        max_seq_len:       Maximum sequence length (prompt + generated tokens).
        n_layers:          Number of transformer layers.
        n_heads:           Number of KV heads (use n_kv_heads for GQA/MQA).
        head_dim:          Dimension per head (d_model // n_heads).
        dtype:             Tensor dtype for cached K/V (float32 or float16).
        device:            Device for cache tensors.
    """

    max_batch_size: int   = 1
    max_seq_len:    int   = 512
    n_layers:       int   = 6
    n_heads:        int   = 8
    head_dim:       int   = 64
    dtype:          str   = "float32"
    device:         str   = "cpu"

    def __post_init__(self) -> None:
        assert self.max_batch_size >= 1
        assert self.max_seq_len >= 1
        assert self.n_layers >= 1
        assert self.n_heads >= 1
        assert self.head_dim >= 1
        assert self.dtype in ("float32", "float16", "bfloat16")

    @property
    def cache_size_bytes(self) -> int:
        """Total bytes needed for all K and V cache tensors."""
        element_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]
        per_tensor    = self.max_batch_size * self.max_seq_len * self.n_heads * self.head_dim
        return 2 * self.n_layers * per_tensor * element_bytes   # 2 = K + V

    @property
    def cache_size_mb(self) -> float:
        return self.cache_size_bytes / (1024 ** 2)
