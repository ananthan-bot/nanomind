"""
nanomind/flash/config.py — Flash Attention configuration.

Standard attention materialises a full (N × N) attention weight matrix:

  Memory: O(N²)  — explodes for long sequences (N=32k → 1B floats per head)
  Speed:  O(N²)  — dominated by slow HBM reads/writes of the attention matrix

Flash Attention (Dao et al. 2022 / 2023) computes the exact same output but
avoids ever writing the full N×N matrix to HBM (GPU global memory).
Instead it tiles Q, K, V into SRAM-resident blocks and uses online softmax
to accumulate the output block by block:

  Memory: O(N)   — only blocks in SRAM; output O(N)
  Speed:  ~2-4× faster than standard attention in practice
  Math:   Exactly equivalent output to standard attention

Reference: Dao et al. (2022) "FlashAttention" — https://arxiv.org/abs/2205.14135
           Dao et al. (2023) "FlashAttention-2" — https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FlashConfig:
    """
    Configuration for Flash Attention.

    Attributes:
        block_q:    Query tile size (number of query rows per SRAM block).
        block_kv:   Key/Value tile size (number of K/V columns per SRAM block).
        causal:     Apply causal (lower-triangular) mask.
        dropout:    Attention dropout probability (training only).
        use_torch_sdpa: Use PyTorch's built-in scaled_dot_product_attention
                        (flash-efficient on CUDA) when available. Falls back
                        to the pure-Python tiled implementation otherwise.
    """

    block_q:         int   = 64
    block_kv:        int   = 64
    causal:          bool  = True
    dropout:         float = 0.0
    use_torch_sdpa:  bool  = True   # Use torch.nn.functional.scaled_dot_product_attention

    def __post_init__(self) -> None:
        assert self.block_q  >= 1
        assert self.block_kv >= 1
        assert 0.0 <= self.dropout <= 1.0
