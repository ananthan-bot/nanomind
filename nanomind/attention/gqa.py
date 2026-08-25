"""
nanomind/attention/gqa.py — Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

Standard multi-head attention (MHA) uses n_heads query, key, and value heads.
GQA and MQA reduce memory pressure by sharing fewer KV heads across query heads:

  ┌─────────────────────────────────────────────────────┐
  │  MHA  (Multi-Head Attention)  → n_kv = n_heads      │
  │  GQA  (Grouped-Query)        → 1 < n_kv < n_heads   │
  │  MQA  (Multi-Query)          → n_kv = 1             │
  └─────────────────────────────────────────────────────┘

Benefits:
  - GQA/MQA dramatically reduce KV-cache size at inference time
  - KV-cache grows as: n_kv_heads × head_dim × seq_len × 2 (K+V)
  - Mistral 7B uses 8 KV heads for 32 query heads (4× cache reduction)
  - Llama 2 70B uses 8 KV heads for 64 query heads (8× cache reduction)

References:
  - GQA: Ainslie et al. (2023) — https://arxiv.org/abs/2305.13245
  - MQA: Shazeer (2019) — https://arxiv.org/abs/1911.02150
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of query heads for GQA.

    Each KV head is repeated ``n_rep`` times along the head dimension.
    This is equivalent to broadcasting each KV group across its query heads.

    Args:
        x:     KV tensor ``(B, n_kv_heads, T, head_dim)``
        n_rep: Number of times to repeat each KV head (= n_heads // n_kv_heads).

    Returns:
        Expanded tensor ``(B, n_heads, T, head_dim)``
    """
    if n_rep == 1:
        return x
    B, n_kv, T, head_dim = x.shape
    return (
        x.unsqueeze(2)                           # (B, n_kv, 1, T, head_dim)
         .expand(B, n_kv, n_rep, T, head_dim)   # (B, n_kv, n_rep, T, head_dim)
         .reshape(B, n_kv * n_rep, T, head_dim) # (B, n_heads, T, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Uses ``n_heads`` query heads but only ``n_kv_heads`` key/value heads.
    Each group of ``n_heads // n_kv_heads`` query heads shares one KV head.

    Special cases:
      - ``n_kv_heads == n_heads``  → standard Multi-Head Attention (MHA)
      - ``n_kv_heads == 1``        → Multi-Query Attention (MQA)

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        n_kv_heads: Number of key/value heads (must divide n_heads evenly).
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        bias:       Whether to add bias to projection layers.

    Raises:
        AssertionError: If ``n_heads % n_kv_heads != 0``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repetitions per KV head
        self.head_dim   = d_model // n_heads
        self.dropout    = dropout

        # Query projection: full n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        # Key / Value projections: only n_kv_heads
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GQA forward pass.

        Steps:
        1. Project input to Q (n_heads), K (n_kv_heads), V (n_kv_heads)
        2. Expand K and V by repeating each n_rep times → (n_heads, T, head_dim)
        3. Compute scaled dot-product attention with causal mask
        4. Project output

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV cache (unused during training).

        Returns:
            Tuple of ``(output, attention_weights)``.
        """
        B, T, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match query heads
        k = repeat_kv(k, self.n_rep)   # (B, n_heads, T, head_dim)
        v = repeat_kv(v, self.n_rep)   # (B, n_heads, T, head_dim)

        # Scaled dot-product attention
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"n_rep={self.n_rep}, "
            f"head_dim={self.head_dim}"
        )


class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-Query Attention (MQA) — special case of GQA where n_kv_heads = 1.

    All query heads share a single key and value head.
    This provides the maximum KV-cache memory reduction.

    Used in: Falcon 7B, early PaLM models.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        bias:       Whether to add bias to projection layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,              # single shared KV head
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads=1 (MQA)"
        )
