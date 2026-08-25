"""
nanomind/attention/gqa.py — Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

Standard multi-head attention (MHA) uses n_heads query, key, and value heads.
GQA and MQA reduce memory pressure by sharing fewer KV heads across query heads:

  ┌─────────────────────────────────────────────────────┐
  │  MHA  (Multi-Head Attention)  → n_kv = n_heads      │
  │  GQA  (Grouped-Query)        → 1 < n_kv < n_heads   │
  │  MQA  (Multi-Query)          → n_kv = 1             │
  └─────────────────────────────────────────────────────┘

Benefits:
  - GQA/MQA dramatically reduce KV-cache size at inference time
  - KV-cache grows as: n_kv_heads × head_dim × seq_len × 2 (K+V)
  - Mistral 7B uses 8 KV heads for 32 query heads (4× cache reduction)
  - Llama 2 70B uses 8 KV heads for 64 query heads (8× cache reduction)

References:
  - GQA: Ainslie et al. (2023) — https://arxiv.org/abs/2305.13245
  - MQA: Shazeer (2019) — https://arxiv.org/abs/1911.02150
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of query heads for GQA.

    Each KV head is repeated ``n_rep`` times along the head dimension.
    This is equivalent to broadcasting each KV group across its query heads.

    Args:
        x:     KV tensor ``(B, n_kv_heads, T, head_dim)``
        n_rep: Number of times to repeat each KV head (= n_heads // n_kv_heads).

    Returns:
        Expanded tensor ``(B, n_heads, T, head_dim)``
    """
    if n_rep == 1:
        return x
    B, n_kv, T, head_dim = x.shape
    return (
        x.unsqueeze(2)                           # (B, n_kv, 1, T, head_dim)
         .expand(B, n_kv, n_rep, T, head_dim)   # (B, n_kv, n_rep, T, head_dim)
         .reshape(B, n_kv * n_rep, T, head_dim) # (B, n_heads, T, head_dim)
    )
