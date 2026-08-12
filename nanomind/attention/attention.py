"""
nanomind/attention/attention.py — Causal multi-head self-attention module.

Implements the decoder-style attention block used in GPT:
- All positions attend only to current and past positions (causal mask)
- Fused QKV projection for efficiency
- Optional KV-cache for fast autoregressive inference
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.functional import make_causal_mask, scaled_dot_product_attention


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length (used to pre-allocate the mask).
        dropout:    Attention and residual dropout probability.
        bias:       Whether to use bias in projections (default: False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.dropout    = dropout

        # Fused QKV projection: one Linear produces Q, K, V concatenated
        self.qkv_proj   = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
