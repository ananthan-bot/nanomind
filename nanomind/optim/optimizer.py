"""
nanomind/optim/optimizer.py — Optimizer factory for NanoMind.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.optim.param_groups import get_param_groups


def get_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Build an optimizer for NanoMind with proper weight decay grouping.

    Automatically separates parameters into decay and no-decay groups
    so that biases, norms, and embeddings are not penalized.

    Args:
        model:          The model to optimize.
        lr:             Peak learning rate.
        weight_decay:   Coefficient applied to decayed parameters.
        betas:          AdamW beta coefficients ``(beta1, beta2)``.
        eps:            Numerical stability epsilon.
        optimizer_type: ``"adamw"`` (default) or ``"sgd"``.

    Returns:
        Configured :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If an unsupported optimizer type is requested.
    """
    groups = get_param_groups(model, weight_decay=weight_decay)

    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)

    raise ValueError(
        f"Unknown optimizer '{optimizer_type}'. Choose 'adamw' or 'sgd'."
    )
