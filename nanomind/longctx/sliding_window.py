"""
nanomind/longctx/sliding_window.py — Sliding Window Attention.

## Sliding Window Attention (Longformer, Mistral)

Standard attention: O(T²) memory and compute.
For long sequences (T=32K), this becomes infeasible.

Sliding window attention restricts each token to attend only to
a local window of W neighbours:
  token i attends to tokens [i-W/2, ..., i+W/2]

This reduces complexity to O(T × W).

Key insight: information still propagates globally through stacking.
With L layers and window W, effective receptive field = L × W.

With window W=4096 and L=32 layers:
  Effective field = 32 × 4096 = 131,072 tokens

Used by: Mistral 7B (W=4096), Longformer (W=512), BigBird.

## Attention Sinks

(Xiao et al., 2023) observed that LLMs always attend strongly to
the very first tokens (attention sinks). This is because:
  - Initial tokens see all future tokens during training
  - Models use them as "resting state" or "memory dump"

StreamingLLM keeps the first K_sink tokens (sinks) + recent W tokens:
  Total window = K_sink + W
  This enables infinite-length generation with bounded memory!

References:
  Beltagy et al. (2020) Longformer: https://arxiv.org/abs/2004.05150
  Jiang et al. (2023) Mistral: https://arxiv.org/abs/2310.06825
  Xiao et al. (2023) StreamingLLM: https://arxiv.org/abs/2309.17453
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Multi-head attention with sliding window constraint.

    Each token only attends to the nearest W tokens,
    with optional attention sink tokens (always attended to).

    Args:
        d_model:     Model dimension.
        n_heads:     Number of attention heads.
        window_size: Local attention window W.
        n_sinks:     Number of sink tokens (always attended to).
        dropout:     Attention dropout.

    Example::

        swa = SlidingWindowAttention(d_model=512, n_heads=8, window_size=256)
        out = swa(x)   # x: (B, T, D)  — efficient O(T×W) attention
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        window_size: int,
        n_sinks:     int   = 4,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.window_size = window_size
        self.n_sinks     = n_sinks

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_head ** -0.5

    def _make_window_mask(self, T: int) -> torch.Tensor:
        """
        Create sliding window causal mask.

        mask[i, j] = True if token j is in window of token i (and j <= i).

        Returns:
            ``(T, T)`` boolean tensor.
        """
        # Causal mask
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        # Window mask: only attend to last W positions
        dist   = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        in_win = dist >= -self.window_size
        # Sink tokens: always attended to
        sinks  = torch.zeros(T, T, dtype=torch.bool)
        if self.n_sinks > 0:
            sinks[:, :self.n_sinks] = True
        return causal & (in_win | sinks)

    def forward(
        self,
        x:            torch.Tensor,
        past_kv:      tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Sliding window attention forward.

        Args:
            x:       ``(B, T, D)`` input tensor.
            past_kv: Optional cached ``(k, v)`` for generation.

        Returns:
            ``(output, (k, v))``
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)   # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        T_full = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale        # (B,H,T,T_full)

        # Apply sliding window + causal mask
        mask   = self._make_window_mask(T_full)                 # (T_full, T_full)
        # For generation (T < T_full), take last T rows
        if T < T_full:
            mask = mask[T_full - T:]
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out     = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), (k, v)

    def effective_context(self, n_layers: int) -> int:
        """Effective receptive field across L layers."""
        return n_layers * self.window_size + self.n_sinks
