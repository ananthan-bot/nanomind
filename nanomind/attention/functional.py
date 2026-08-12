"""
nanomind/attention/functional.py — Pure-function attention operations.

Implements the core mathematical operations of attention independently
of any nn.Module, so they can be tested in isolation.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Implements: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        q:         Query tensor  ``(B, n_heads, T, head_dim)``
        k:         Key tensor    ``(B, n_heads, T, head_dim)``
        v:         Value tensor  ``(B, n_heads, T, head_dim)``
        mask:      Boolean mask ``(1, 1, T, T)`` — True positions are MASKED OUT
        dropout_p: Attention dropout probability
        training:  Whether the model is in training mode

    Returns:
        Tuple of:
        - ``out``    : attended output ``(B, n_heads, T, head_dim)``
        - ``weights``: attention weights ``(B, n_heads, T, T)``
    """
    d_k = q.size(-1)
    scale = math.sqrt(d_k)

    # (B, n_heads, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    out = torch.matmul(weights, v)
    return out, weights


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention mask.

    A True value at position (i, j) means position i is NOT allowed
    to attend to position j (i.e., j > i is masked out).

    Args:
        seq_len: Sequence length T.
        device:  Target device for the mask tensor.

    Returns:
        Boolean tensor of shape ``(1, 1, T, T)``.
        Upper triangle (excluding diagonal) is True (masked).

    Example::

        mask = make_causal_mask(4, device)
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
    """
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(ones, diagonal=1)          # Upper triangle = True (masked)
    return mask.unsqueeze(0).unsqueeze(0)        # (1, 1, T, T)
