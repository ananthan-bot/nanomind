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


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler than LayerNorm — no mean subtraction, only RMS scaling.
    Used in LLaMA, Mistral, and other modern LLMs.

    Formula: x / RMS(x) * weight,  where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the last dimension of ``x``."""
        # Compute RMS along the last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"


# ── Norm registry ─────────────────────────────────────────────────────────────

_NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "layernorm": LayerNorm,
    "rmsnorm":   RMSNorm,
}


def get_norm(name: str, d_model: int, **kwargs) -> nn.Module:
    """
    Instantiate a normalization layer by name.

    Args:
        name:    Norm type — ``"layernorm"`` or ``"rmsnorm"``.
        d_model: Feature dimension.
        **kwargs: Extra arguments forwarded to the norm constructor.

    Returns:
        An :class:`nn.Module` normalization layer.

    Raises:
        ValueError: If the name is not recognised.

    Example::

        norm = get_norm("rmsnorm", d_model=256)
        out  = norm(x)
    """
    key = name.lower().replace("_", "")
    if key not in _NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm '{name}'. Available: {sorted(_NORM_REGISTRY)}"
        )
    return _NORM_REGISTRY[key](d_model, **kwargs)


def list_norms() -> list[str]:
    """Return a sorted list of all registered normalization names."""
    return sorted(_NORM_REGISTRY)
