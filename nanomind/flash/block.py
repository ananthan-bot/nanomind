"""
nanomind/flash/block.py — Transformer block with FlashAttention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.flash.module import FlashAttention
from nanomind.flash.config import FlashConfig
from nanomind.norm.factory import get_norm


class FlashTransformerBlock(nn.Module):
    """
    Pre-norm transformer block using FlashAttention instead of standard SDPA.

    Architecture:
        x → RMSNorm → FlashAttention → x + residual
        x → RMSNorm → SwiGLU FFN    → x + residual

    Args:
        d_model:   Embedding dimension.
        n_heads:   Number of attention heads.
        flash_cfg: Flash Attention configuration.
        dropout:   Dropout for attention and residuals.
        norm_type: Normalisation (``"rmsnorm"`` or ``"layernorm"``).
        bias:      Use bias in projections.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        flash_cfg: FlashConfig | None = None,
        dropout:   float = 0.0,
        norm_type: str   = "rmsnorm",
        bias:      bool  = False,
    ) -> None:
        super().__init__()
        d_ff = int(d_model * 8 / 3)   # SwiGLU typical d_ff

        self.norm1  = get_norm(norm_type, d_model)
        self.attn   = FlashAttention(d_model, n_heads, flash_cfg, bias)
        self.norm2  = get_norm(norm_type, d_model)
        self.gate   = nn.Linear(d_model, d_ff, bias=bias)
        self.up     = nn.Linear(d_model, d_ff, bias=bias)
        self.down   = nn.Linear(d_ff, d_model, bias=bias)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flash attention
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.drop(self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h)))
        return x
