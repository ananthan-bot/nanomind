"""
nanomind/attention/swa.py — Sliding Window Attention (SWA).

Standard full attention has O(T²) memory and compute with sequence length T.
Sliding Window Attention limits each token to attending only to the W
previous tokens, reducing complexity to O(T × W):

  ┌──────────────────────────────────────────────────────┐
  │  Full attention       :  O(T²)   memory & compute    │
  │  Sliding Window (SWA) :  O(T·W)  memory & compute    │
  └──────────────────────────────────────────────────────┘

Each token can attend to up to W tokens in its local causal window.
Tokens more than W steps away are masked out (set to -∞ before softmax).

With multiple layers:
  - Layer 1: each token "sees" W tokens
  - Layer 2: each token effectively sees W² tokens (via layer 1 outputs)
  - Layer L: receptive field grows as W^L → full context with enough layers

Used in:
  - Mistral 7B (window_size=4096 on 8192 block_size)
  - Longformer (local + global attention)

Reference: Jiang et al. (2023) — https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a causal sliding-window attention mask.

    Token at position ``i`` can attend to positions ``j`` where::

        max(0, i - window_size + 1) <= j <= i

    Positions outside the window are masked to ``-inf``.

    Args:
        seq_len:     Sequence length T.
        window_size: Maximum look-back window W (number of tokens visible).
        device:      Target device.

    Returns:
        Boolean mask ``(T, T)`` where ``True`` means **allowed** to attend.
        (Consistent with PyTorch's ``scaled_dot_product_attention`` convention.)

    Example (T=5, W=3)::

        token 0: sees [0]
        token 1: sees [0, 1]
        token 2: sees [0, 1, 2]
        token 3: sees [1, 2, 3]
        token 4: sees [2, 3, 4]
    """
    # Start from causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Build band mask: only allow positions within window_size
    # distance[i, j] = i - j   (negative means j > i → already masked by causal)
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = rows - cols                                      # (T, T)
    window_mask = distance < window_size                        # j >= i - W + 1

    return causal & window_mask
