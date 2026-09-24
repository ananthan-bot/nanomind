"""NanoMind Long-Context sub-package — Efficient Attention for long sequences.

Implements the full suite of long-context attention mechanisms:
  1. RotaryEmbedding    — RoPE with linear/NTK scaling, apply(q,k)
  2. ALiBi              — linear bias attention, no position embeddings
  3. SlidingWindowAttention — O(T×W) local+sink attention (Mistral-style)
  4. LinearAttention    — O(T) kernel attention (ELU feature map)
  5. RetNetDecay        — decayed linear attention (RetNet-style)
  6. GroupedQueryAttention — GQA/MQA with RoPE, kv_cache_factor
  7. ChunkedAttention   — FlashAttention-style tiled memory-efficient attention
  8. LongContextConfig  — LM configuration
  9. LongContextLM      — full LM with pluggable attention

Primary exports:
    - :class:`RotaryEmbedding`         — apply, extend, linear/NTK scaling
    - :class:`ALiBi`                   — bias, apply_to_scores, slopes
    - :class:`SlidingWindowAttention`  — window_mask, sinks, past_kv
    - :class:`LinearAttention`         — ELU/ReLU feature map, O(T)
    - :class:`RetNetDecay`             — per-head gamma decay
    - :class:`GroupedQueryAttention`   — GQA, MQA, RoPE, kv_cache_factor
    - :class:`ChunkedAttention`        — tiled attention, online softmax
    - :class:`LongContextConfig`       — vocab, attn_type, pos_encoding
    - :class:`LongContextLM`           — n_params, effective_context
"""

from nanomind.longctx.rope import RotaryEmbedding
from nanomind.longctx.alibi import ALiBi
from nanomind.longctx.sliding_window import SlidingWindowAttention
from nanomind.longctx.linear_attn import LinearAttention, RetNetDecay
from nanomind.longctx.gqa import GroupedQueryAttention
from nanomind.longctx.chunked import ChunkedAttention
from nanomind.longctx.model import LongContextConfig, LongContextLM

__all__ = [
    "RotaryEmbedding",
    "ALiBi",
    "SlidingWindowAttention",
    "LinearAttention", "RetNetDecay",
    "GroupedQueryAttention",
    "ChunkedAttention",
    "LongContextConfig", "LongContextLM",
]
