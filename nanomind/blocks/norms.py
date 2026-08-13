"""
nanomind/blocks/norms.py — Normalization layers for NanoMind.

Provides LayerNorm and RMSNorm, along with a registry so the
normalization type can be swapped via config without code changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (Ba et al., 2016).

    Thin wrapper around :class:`torch.nn.LayerNorm` that matches
    the RMSNorm interface so they can be used interchangeably.

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
        bias:    If True, learn an additive bias parameter.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=True)
        if not bias:
            # Zero-out and freeze the bias
            nn.init.zeros_(self.norm.bias)
            self.norm.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over the last dimension."""
        return self.norm(x)

    def extra_repr(self) -> str:
        return f"d_model={self.norm.normalized_shape[0]}"
