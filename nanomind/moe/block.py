"""
nanomind/moe/block.py — Transformer block with MoE FFN replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.moe.config import MoEConfig
from nanomind.moe.layer import SparseMoELayer
from nanomind.pos.factory import get_attention
from nanomind.norm.factory import get_norm


class MoETransformerBlock(nn.Module):
    """
    Transformer block where the dense FFN is replaced by a SparseMoELayer.

    Architecture (pre-norm):
        x → Norm → Attention → x + residual
        x → Norm → SparseMoE → x + residual (+ aux_loss)

    Args:
        d_model:    Embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        moe_cfg:    MoE configuration.
        dropout:    Dropout for attention and residual.
        norm_type:  Layer norm type (``"layernorm"`` or ``"rmsnorm"``).
        pos_type:   Positional embedding type for attention.
        n_kv_heads: For GQA/MQA (None = standard MHA).
        window_size: For Sliding Window Attention.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        moe_cfg:     MoEConfig,
        dropout:     float = 0.0,
        norm_type:   str   = "layernorm",
        pos_type:    str   = "learned",
        n_kv_heads:  int | None = None,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.attn = get_attention(
            pos_type=pos_type,
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
        )
        self.moe      = SparseMoELayer(d_model, moe_cfg)
        self.norm1    = get_norm(norm_type, d_model)
        self.norm2    = get_norm(norm_type, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE transformer block.

        Returns:
            Tuple of ``(output, aux_loss)``.
        """
        # Self-attention with pre-norm
        attn_out, _ = self.attn(self.norm1(x), kv_cache)
        x = x + self.drop(attn_out)

        # MoE FFN with pre-norm
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.drop(moe_out)

        return x, aux_loss
