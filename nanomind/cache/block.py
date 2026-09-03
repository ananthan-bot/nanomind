"""
nanomind/cache/block.py — Transformer block with KV cache support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.cache.attention import CachedSelfAttention
from nanomind.cache.layer_cache import LayerKVCache
from nanomind.norm.factory import get_norm


class CachedTransformerBlock(nn.Module):
    """
    Pre-norm transformer block with CachedSelfAttention + dense FFN.

    Args:
        d_model:   Model embedding dimension.
        n_heads:   Number of attention heads.
        d_ff:      FFN hidden dimension (default: 4 × d_model).
        dropout:   Dropout probability.
        norm_type: Normalisation type (``"layernorm"`` or ``"rmsnorm"``).
        bias:      Use bias in projections and FFN.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        d_ff:      int | None = None,
        dropout:   float = 0.0,
        norm_type: str   = "layernorm",
        bias:      bool  = False,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = get_norm(norm_type, d_model)
        self.attn  = CachedSelfAttention(d_model, n_heads, dropout, bias)
        self.norm2 = get_norm(norm_type, d_model)
        self.ff1   = nn.Linear(d_model, d_ff, bias=bias)
        self.ff2   = nn.Linear(d_ff, d_model, bias=bias)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(
        self,
        x:        torch.Tensor,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        # Attention with pre-norm
        x = x + self.drop(self.attn(self.norm1(x), kv_cache))
        # FFN with pre-norm
        h = self.norm2(x)
        x = x + self.drop(self.ff2(self.act(self.ff1(h))))
        return x
