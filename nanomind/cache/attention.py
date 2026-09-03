"""
nanomind/cache/attention.py — Cache-aware scaled dot-product attention.

Wraps standard SDPA to work with a LayerKVCache. On each forward call:
  1. Project Q, K, V from input (only the NEW tokens)
  2. Update the KV cache with K_new, V_new
  3. Compute attention of Q_new over full K/V from cache

This avoids recomputing K/V for past tokens on every step.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.cache.layer_cache import LayerKVCache


class CachedSelfAttention(nn.Module):
    """
    Self-attention with KV cache support for fast autoregressive decoding.

    Supports both prefill (process full prompt at once) and decode
    (one new token at a time with cached K/V).

    Args:
        d_model:  Model embedding dimension.
        n_heads:  Number of attention heads.
        dropout:  Attention dropout (only during training).
        bias:     Use bias in Q/K/V/out projections.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.0,
        bias:     bool  = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,
        kv_cache:   LayerKVCache | None = None,
        mask:       torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV cache.

        Args:
            x:        Input ``(B, T, d_model)``. T=1 for decode step.
            kv_cache: LayerKVCache for this layer (None = no cache).
            mask:     Causal mask ``(T, T_full)`` or None.

        Returns:
            Output ``(B, T, d_model)``.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        # Project Q, K, V for new tokens only
        q = self.q_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)
        v = self.v_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)

        # Update cache → get full K/V
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)         # (B, T_full, H, Dh)

        T_full = k.shape[1]

        # Reshape for batched SDPA: (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T_full)

        if mask is not None:
            attn = attn + mask

        if kv_cache is None:
            # Causal mask for full-sequence (training / prefill)
            causal = torch.triu(
                torch.full((T, T_full), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out  = torch.matmul(attn, v)            # (B, H, T, Dh)
        out  = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
