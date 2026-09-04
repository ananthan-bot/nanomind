"""
nanomind/flash/module.py — FlashAttention as an nn.Module.

Provides a drop-in replacement for standard multi-head self-attention
that uses either:
  1. torch.nn.functional.scaled_dot_product_attention (CUDA flash-efficient)
  2. The pure-PyTorch tiled reference implementation (CPU / educational)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.flash.config import FlashConfig
from nanomind.flash.tiled import tiled_flash_attention


class FlashAttention(nn.Module):
    """
    Multi-head self-attention with Flash Attention backend.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` when
    ``cfg.use_torch_sdpa=True`` (CUDA flash-efficient on compatible hardware)
    and falls back to the tiled reference implementation otherwise.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        cfg:     Flash Attention configuration.
        bias:    Use bias in Q/K/V/out projections.

    Example::

        attn = FlashAttention(256, 8, FlashConfig(causal=True))
        x    = torch.randn(2, 128, 256)
        out, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cfg:     FlashConfig | None = None,
        bias:    bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.cfg      = cfg or FlashConfig()
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop     = nn.Dropout(self.cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """
        Flash attention forward pass.

        Args:
            x:    Input ``(B, T, d_model)``.
            mask: Optional additive mask ``(B, 1, T, T)`` or ``(T, T)``.

        Returns:
            Tuple of ``(output, None)`` — None for API compatibility.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if self.cfg.use_torch_sdpa:
            # PyTorch built-in flash-efficient SDPA
            attn_mask = mask
            if attn_mask is None and self.cfg.causal:
                attn_mask = None   # is_causal handles it
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=self.cfg.causal if attn_mask is None else False,
                scale=self.scale,
            )
        else:
            # Pure-Python tiled reference
            out = tiled_flash_attention(
                q, k, v,
                block_q=self.cfg.block_q,
                block_kv=self.cfg.block_kv,
                causal=self.cfg.causal,
                scale=self.scale,
            )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), None

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"causal={self.cfg.causal}, "
            f"backend={'torch_sdpa' if self.cfg.use_torch_sdpa else 'tiled'}"
        )
