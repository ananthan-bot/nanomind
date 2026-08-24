"""NanoMind positional embedding sub-package.

Positional Embedding Types:
    - ``"learned"`` : Learned absolute positional embeddings (original Transformer / GPT)
    - ``"rope"``    : Rotary Position Embeddings — LLaMA, Mistral, PaLM 2
    - ``"alibi"``   : Attention with Linear Biases — BLOOM, MPT

Primary exports:
    - :func:`get_attention`          — factory: return attention module by pos_type
    - :func:`list_pos_types`         — list all registered positional types
    - :class:`RotaryEmbedding`       — RoPE module
    - :func:`apply_rotary_emb`       — apply RoPE to Q, K tensors
    - :func:`precompute_rope_freqs`  — precompute cos/sin frequency tables
    - :func:`rotate_half`            — 90-degree rotation helper
    - :func:`build_alibi_bias`       — build ALiBi position bias tensor
    - :func:`get_alibi_slopes`       — compute per-head ALiBi slopes
    - :class:`RoPECausalSelfAttention`   — attention with RoPE
    - :class:`ALiBiCausalSelfAttention`  — attention with ALiBi
"""

from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.pos.rope import (
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_rope_freqs,
    rotate_half,
)
from nanomind.pos.alibi import build_alibi_bias, get_alibi_slopes
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

__all__ = [
    "get_attention",
    "list_pos_types",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "precompute_rope_freqs",
    "rotate_half",
    "build_alibi_bias",
    "get_alibi_slopes",
    "RoPECausalSelfAttention",
    "ALiBiCausalSelfAttention",
]
