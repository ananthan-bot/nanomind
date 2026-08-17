"""
nanomind/optim/grad_utils.py — Gradient analysis utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the global gradient norm across all parameters.

    Equivalent to :func:`torch.nn.utils.clip_grad_norm_` but without clipping.
    Useful for monitoring gradient health during training.

    Args:
        model:     The model whose gradients to measure.
        norm_type: The norm type (default: L2).

    Returns:
        Total gradient norm as a float. Returns 0.0 if no gradients exist.
    """
    params_with_grad = [
        p for p in model.parameters()
        if p.grad is not None
    ]
    if not params_with_grad:
        return 0.0

    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type)
            for p in params_with_grad
        ]),
        norm_type,
    )
    return total_norm.item()


def get_grad_stats(model: nn.Module) -> dict[str, float]:
    """
    Compute gradient statistics for debugging.

    Returns:
        Dict with ``"max"``, ``"min"``, ``"mean"``, ``"l2_norm"`` values.
    """
    grads = [
        p.grad.detach().abs()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not grads:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "l2_norm": 0.0}

    all_grads = torch.cat([g.flatten() for g in grads])
    return {
        "max":     all_grads.max().item(),
        "min":     all_grads.min().item(),
        "mean":    all_grads.mean().item(),
        "l2_norm": compute_grad_norm(model),
    }
