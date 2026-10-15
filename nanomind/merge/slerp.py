"""
nanomind/merge/slerp.py — Spherical Linear Interpolation (SLERP) for high-dimensional parameter tensors.
Interpolates along geodesic on hypersphere preserving directional variance.
"""
import math
from typing import Optional
import torch


def slerp(t1: torch.Tensor, t2: torch.Tensor, t: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two parameter tensors:
    SLERP(t1, t2; t) = (sin((1 - t)*Omega) / sin(Omega)) * t1 + (sin(t*Omega) / sin(Omega)) * t2
    t: interpolation factor in [0, 1] (0 = t1, 1 = t2)
    """
    if t <= 0.0:
        return t1
    if t >= 1.0:
        return t2

    norm_t1 = t1 / (torch.norm(t1) + eps)
    norm_t2 = t2 / (torch.norm(t2) + eps)

    # Cosine of angle between vectors
    dot = torch.sum(norm_t1 * norm_t2)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    # Angle Omega
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if torch.abs(sin_omega) < eps:
        # Collinear vectors: fall back to linear interpolation
        return (1.0 - t) * t1 + t * t2

    scale_1 = torch.sin((1.0 - t) * omega) / sin_omega
    scale_2 = torch.sin(t * omega) / sin_omega

    interpolated = scale_1 * t1 + scale_2 * t2
    return interpolated
