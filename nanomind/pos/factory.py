"""
nanomind/pos/factory.py — Attention factory keyed by positional embedding type.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.attention.gqa import GroupedQueryAttention, MultiQueryAttention
from nanomind.attention.gqa_rope import GQARoPEAttention
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

_ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "learned":  CausalSelfAttention,
    "rope":     RoPECausalSelfAttention,
    "alibi":    ALiBiCausalSelfAttention,
    "gqa":      GroupedQueryAttention,
    "mqa":      MultiQueryAttention,
    "gqa_rope": GQARoPEAttention,
}


def get_attention(
    pos_type: str,
    d_model: int,
    n_heads: int,
    block_size: int,
    dropout: float = 0.1,
    n_kv_heads: int | None = None,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a causal self-attention module by type.

    Args:
        pos_type:   ``"learned"``, ``"rope"``, ``"alibi"``,
                    ``"gqa"``, ``"mqa"``, or ``"gqa_rope"``.
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        n_kv_heads: KV head count for GQA/MQA (ignored for MHA/MQA).
        **kwargs:   Extra arguments forwarded to the attention constructor.

    Returns:
        Configured attention :class:`nn.Module`.
    """
    key = pos_type.lower()
    if key not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown pos_type '{pos_type}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY)}"
        )

    # GQA variants need n_kv_heads
    if key in ("gqa", "gqa_rope") and n_kv_heads is not None:
        return _ATTENTION_REGISTRY[key](
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            block_size=block_size, dropout=dropout, **kwargs,
        )

    return _ATTENTION_REGISTRY[key](
        d_model=d_model, n_heads=n_heads,
        block_size=block_size, dropout=dropout, **kwargs,
    )


def list_pos_types() -> list[str]:
    """Return sorted list of all registered attention/positional types."""
    return sorted(_ATTENTION_REGISTRY)
