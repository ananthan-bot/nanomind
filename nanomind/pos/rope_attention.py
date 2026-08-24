"""
nanomind/pos/rope_attention.py — Causal self-attention with Rotary Position Embeddings.

Extends the base CausalSelfAttention by replacing the learned positional
embedding with RoPE applied inside the attention head computation.

Key difference from base attention:
  - No positional embedding at the input level
  - RoPE rotates Q and K *after* projection, *before* dot-product
  - Relative position is encoded implicitly in the QK scores
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.pos.rope import RotaryEmbedding


class RoPECausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length (used to build causal mask).
        dropout:     Attention dropout probability.
        rope_base:   RoPE frequency base (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.dropout   = dropout

        # Linear projections (no bias — matches LLaMA style)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE module
        self.rope = RotaryEmbedding(self.head_dim, block_size, rope_base)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with RoPE-rotated Q and K.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV cache (not used in training).

        Returns:
            Tuple of ``(output, attention_weights)``
        """
        B, T, _ = x.shape

        # Project and reshape to heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention with causal mask
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)                                 # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}"
        )
