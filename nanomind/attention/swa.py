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


class SlidingWindowAttention(nn.Module):
    """
    Multi-head causal self-attention with a sliding local window.

    Each position can only attend to the ``window_size`` most recent tokens,
    drastically reducing memory from O(T²) to O(T × window_size).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length (for mask precomputation).
        window_size: Local attention window (W). Each token sees W past tokens.
        dropout:     Attention dropout probability.
        bias:        Whether projections have bias terms.

    Note:
        With ``window_size >= block_size``, this reduces to standard full attention.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        window_size: int = 256,
        dropout:     float = 0.1,
        bias:        bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = min(window_size, block_size)
        self.dropout     = dropout

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # Precompute the sliding window mask for the full block_size
        mask = build_sliding_window_mask(block_size, self.window_size)
        self.register_buffer("swa_mask", mask)   # (block_size, block_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sliding window attention forward pass.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Not used during training.

        Returns:
            ``(output, attention_weights)``
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Slice mask to current sequence length
        mask = self.swa_mask[:T, :T]                      # (T, T)

        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply sliding window mask: positions outside window → -inf
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size}"
        )
