"""
nanomind/pos/factory.py — Attention factory keyed by positional embedding type.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

_ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "learned": CausalSelfAttention,
    "rope":    RoPECausalSelfAttention,
    "alibi":   ALiBiCausalSelfAttention,
}


def get_attention(
    pos_type: str,
    d_model: int,
    n_heads: int,
    block_size: int,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a causal self-attention module by positional embedding type.

    Args:
        pos_type:   ``"learned"`` (default), ``"rope"``, or ``"alibi"``.
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        **kwargs:   Extra arguments forwarded to the attention constructor.

    Returns:
        Configured attention :class:`nn.Module`.

    Raises:
        ValueError: If ``pos_type`` is not recognised.
    """
    key = pos_type.lower()
    if key not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown pos_type '{pos_type}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY)}"
        )
    return _ATTENTION_REGISTRY[key](
        d_model=d_model, n_heads=n_heads,
        block_size=block_size, dropout=dropout,
        **kwargs,
    )


def list_pos_types() -> list[str]:
    """Return a sorted list of all registered positional embedding types."""
    return sorted(_ATTENTION_REGISTRY)
