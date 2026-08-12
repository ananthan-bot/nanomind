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

        # Pre-computed causal mask registered as a buffer (moves with the model)
        mask = make_causal_mask(block_size, device=torch.device("cpu"))
        self.register_buffer("causal_mask", mask)   # (1, 1, T, T)

    # ── Head utilities ────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape ``(B, T, d_model)`` -> ``(B, n_heads, T, head_dim)``.
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)   # (B, n_heads, T, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape ``(B, n_heads, T, head_dim)`` -> ``(B, T, d_model)``.
        """
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal self-attention.

        Args:
            x:        Input tensor ``(B, T, d_model)``
            kv_cache: Optional dict for incremental decoding (see Commit 11).

        Returns:
            Tuple of:
            - ``out``     : attended output ``(B, T, d_model)``
            - ``weights`` : attention weights ``(B, n_heads, T, T)``
        """
        B, T, C = x.shape

        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # KV-cache: append new k/v and retrieve full history
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        T_full = k.size(2)   # full cached sequence length

        # Causal mask — only needed when attending to multiple positions
        if T_full > 1:
            mask = self.causal_mask[:, :, :T_full, :T_full]
            # When using cache, query only covers the new token(s)
            if kv_cache is not None and T < T_full:
                mask = mask[:, :, T_full - T:, :]
        else:
            mask = None

        attn_out, weights = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        out = self._merge_heads(attn_out)
        out = self.resid_drop(self.out_proj(out))
        return out, weights
