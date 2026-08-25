"""
nanomind/attention/gqa_rope.py — Grouped-Query Attention with RoPE.

Combines GQA's memory-efficient KV sharing with RoPE's relative position
encoding. This is the exact attention mechanism used in Llama 2 and Mistral.

Architecture:
  - Q: n_heads projections with RoPE applied
  - K: n_kv_heads projections with RoPE applied
  - V: n_kv_heads projections (no positional encoding needed)
  - K, V expanded via repeat_kv before attention
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.gqa import repeat_kv
from nanomind.pos.rope import RotaryEmbedding


class GQARoPEAttention(nn.Module):
    """
    Grouped-Query Attention with Rotary Position Embeddings.

    This is the standard attention in Llama 2 / Mistral:
      - ``n_kv_heads`` KV heads (far fewer than query heads)
      - RoPE applied to both Q and K *after* projection

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        n_kv_heads: Number of KV heads (must divide n_heads).
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        rope_base:  RoPE frequency base (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_size: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert d_model % n_heads == 0

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj   = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope      = RotaryEmbedding(self.head_dim, block_size, rope_base)
        self.attn_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (must be done before repeat_kv)
        q, k = self.rope(q, k)

        # Expand K and V to match query head count
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scale   = 1.0 / math.sqrt(self.head_dim)
        scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores  = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads} (GQA+RoPE)"
        )
