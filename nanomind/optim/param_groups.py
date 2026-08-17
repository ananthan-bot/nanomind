"""
nanomind/optim/param_groups.py — Parameter group utilities for AdamW.

AdamW applies weight decay to all parameters by default, but it should NOT
be applied to biases, LayerNorm/RMSNorm weights, or embedding weights —
decaying these degrades performance.

This module splits parameters into two groups:
  - ``decay``:   all weight matrices (2D+)
  - ``no_decay``: biases, 1D params (norms), embeddings
"""

from __future__ import annotations

import torch.nn as nn


def get_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> list[dict]:
    """
    Split model parameters into decay and no-decay groups for AdamW.

    Rules:
    - 2D+ parameters (weight matrices) get weight decay.
    - 1D parameters (biases, norm weights) get NO weight decay.
    - Embedding weights get NO weight decay.

    Args:
        model:        The model whose parameters to group.
        weight_decay: Weight decay coefficient for the decay group.

    Returns:
        List of two param group dicts, ready to pass to an optimizer.

    Example::

        groups = get_param_groups(model, weight_decay=0.1)
        optimizer = torch.optim.AdamW(groups, lr=3e-4)
    """
    decay_params    : list = []
    no_decay_params : list = []

    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        # No decay: 1-D params (biases, norms), embedding matrices
        if param.dim() < 2 or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def count_param_groups(groups: list[dict]) -> dict[str, int]:
    """Return parameter counts per group."""
    return {
        "decay":    sum(p.numel() for p in groups[0]["params"]),
        "no_decay": sum(p.numel() for p in groups[1]["params"]),
    }
