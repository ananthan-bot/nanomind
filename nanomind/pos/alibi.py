"""
nanomind/pos/alibi.py — Attention with Linear Biases (ALiBi).

ALiBi replaces positional embeddings with a simple linear position bias
added to attention scores *before* softmax::

    scores = QK^T / sqrt(d) - m * |i - j|

where ``m`` is a head-specific slope and |i-j| is the distance between positions.

Key properties:
  - No position embedding parameters
  - Excellent length extrapolation (models trained on short sequences
    can generate longer sequences at inference time)
  - Used in: BLOOM, MPT

Reference: Press et al. (2021) — https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute the per-head ALiBi slope values.

    Slopes are the geometric sequence: 2^(-8/n) for n_heads heads.

    Args:
        n_heads: Number of attention heads.

    Returns:
        Slope tensor ``(n_heads,)``
    """

    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_slopes_power_of_2(n_heads))

    # For non-power-of-2 heads, interpolate
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    base_slopes  = _slopes_power_of_2(closest_pow2)
    extra_slopes = _slopes_power_of_2(2 * closest_pow2)[0::2]
    slopes = base_slopes + extra_slopes[: n_heads - closest_pow2]
    return torch.tensor(slopes)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build the ALiBi position bias tensor.

    The bias for position pair (i, j) is ``-slope * |i - j|``.

    Args:
        n_heads: Number of attention heads.
        seq_len: Current sequence length.
        device:  Target device.

    Returns:
        Bias tensor ``(1, n_heads, seq_len, seq_len)``
    """
    slopes = get_alibi_slopes(n_heads).to(device)  # (n_heads,)

    # Build distance matrix: |i - j| for i, j in [0, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)
    distances = (positions - positions.T).abs().float()              # (T, T)

    # Outer product of slopes and distances: (n_heads, T, T)
    bias = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
    return bias.unsqueeze(0)   # (1, n_heads, T, T)
