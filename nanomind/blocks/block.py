"""
nanomind/blocks/block.py — Transformer block (Pre-Norm and Post-Norm variants).

Each block contains:
1. A multi-head causal self-attention sub-layer
2. A position-wise feed-forward sub-layer
Both with residual connections and normalization.

Pre-Norm (used here by default):
    out = x + Attention(Norm(x))
    out = out + FFN(Norm(out))

Post-Norm (original Transformer, Vaswani 2017):
    out = Norm(x + Attention(x))
    out = Norm(out + FFN(out))

Pre-Norm trains more stably at scale without learning rate warmup tricks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.blocks.feedforward import FeedForward
from nanomind.blocks.norms import get_norm


class TransformerBlock(nn.Module):
    """
    A single Pre-Norm transformer block.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        d_ff:       FFN hidden dimension (default: 4 * d_model).
        dropout:    Dropout probability.
        norm_type:  ``"layernorm"`` or ``"rmsnorm"``.
        activation: FFN activation — ``"gelu"`` or ``"swiglu"``.
        norm_placement: ``"pre"`` (default) or ``"post"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        norm_placement: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_placement = norm_placement.lower()
        assert self.norm_placement in ("pre", "post"), (
            "norm_placement must be 'pre' or 'post'"
        )

        self.attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads,
            block_size=block_size, dropout=dropout,
        )
        self.ffn  = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
