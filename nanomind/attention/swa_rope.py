"""
nanomind/attention/swa_rope.py — Sliding Window Attention with RoPE.

Combines Sliding Window Attention with Rotary Position Embeddings.
This is the exact attention mechanism used in Mistral 7B:
  - SWA limits context to window_size tokens (reduces memory)
  - RoPE encodes relative position (enables length generalisation)
  - GQA reduces KV cache (via n_kv_heads < n_heads — see Day 16)

Reference: Jiang et al. (2023) — https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.swa import build_sliding_window_mask
from nanomind.pos.rope import RotaryEmbedding


class SWARoPEAttention(nn.Module):
    """
    Sliding Window Attention with Rotary Position Embeddings (Mistral-style).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length.
        window_size: Local attention window W.
        dropout:     Attention dropout.
        rope_base:   RoPE frequency base.
        bias:        Whether projections have bias.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        window_size: int   = 256,
        dropout:     float = 0.0,
        rope_base:   float = 10000.0,
        bias:        bool  = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = min(window_size, block_size)

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope      = RotaryEmbedding(self.head_dim, block_size, rope_base)
        self.attn_drop = nn.Dropout(dropout)

        mask = build_sliding_window_mask(block_size, self.window_size)
        self.register_buffer("swa_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        mask   = self.swa_mask[:T, :T]
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size} (SWA+RoPE)"
        )
