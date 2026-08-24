"""
nanomind/pos/rope.py — Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating query and key vectors in complex space.
Unlike learned or sinusoidal embeddings, RoPE is applied *inside* attention
directly to Q and K — position information is baked into the dot product.

Key properties:
  - Relative position awareness: the dot product <Rq, Rk> depends only on (m-n)
  - Extrapolation: generalises better to longer sequences than learned embeddings
  - No additional parameters
  - Used in: LLaMA, Mistral, PaLM 2, Falcon, GPT-NeoX

Reference: Su et al. (2021) — https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of ``x`` by 90 degrees (half-dimension rotation).

    Splits the last dimension into two halves and returns::

        [-x2, x1]  where x = [x1, x2]

    This implements the complex-number rotation step in RoPE.

    Args:
        x: Input tensor of shape ``(..., d)`` where d is even.

    Returns:
        Rotated tensor of the same shape.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine frequency matrices for RoPE.

    The frequencies follow the formula::

        theta_i = 1 / base^(2i / head_dim),  i in [0, head_dim/2)

    Args:
        head_dim:    Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        base:        RoPE base frequency (default: 10000, as in original paper).
        device:      Target device.

    Returns:
        Tuple of ``(cos, sin)`` tensors of shape ``(max_seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Inverse frequencies: shape (head_dim/2,)
    theta = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # Positions: shape (max_seq_len,)
    t = torch.arange(max_seq_len, device=device).float()

    # Outer product: (max_seq_len, head_dim/2)
    freqs = torch.outer(t, theta)

    # Duplicate to match head_dim: (max_seq_len, head_dim)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Rotates Q and K in-place using precomputed cos/sin frequencies::

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    Args:
        q:   Query tensor ``(B, n_heads, T, head_dim)``
        k:   Key tensor   ``(B, n_heads, T, head_dim)``
        cos: Cosine freq  ``(T, head_dim)``
        sin: Sine freq    ``(T, head_dim)``

    Returns:
        Tuple of rotated ``(q_rot, k_rot)`` with same shapes as inputs.
    """
    # Broadcast cos/sin over batch and head dims
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
