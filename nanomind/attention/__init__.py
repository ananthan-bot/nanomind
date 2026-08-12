"""NanoMind attention sub-package.

Core components:
    - :class:`CausalSelfAttention` — multi-head causal self-attention layer
    - :class:`KVCache`             — key-value cache for fast inference
    - :class:`AttentionConfig`     — attention configuration dataclass
    - :func:`make_causal_mask`     — causal mask utility
    - :func:`scaled_dot_product_attention` — pure-function attention math
"""

from nanomind.attention.attention import CausalSelfAttention
from nanomind.attention.kv_cache import KVCache
from nanomind.attention.config import AttentionConfig
from nanomind.attention.functional import (
    scaled_dot_product_attention,
    fast_scaled_dot_product_attention,
    make_causal_mask,
)

__all__ = [
    "CausalSelfAttention",
    "KVCache",
    "AttentionConfig",
    "scaled_dot_product_attention",
    "fast_scaled_dot_product_attention",
    "make_causal_mask",
]
