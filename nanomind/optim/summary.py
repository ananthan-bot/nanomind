"""
nanomind/optim/summary.py — Optimizer and schedule summary utilities.
"""

from __future__ import annotations

import torch


def optimizer_summary(optimizer: torch.optim.Optimizer) -> str:
    """
    Return a human-readable summary of the optimizer and its param groups.

    Args:
        optimizer: Any :class:`torch.optim.Optimizer`.

    Returns:
        Multi-line string summary.
    """
    lines = [f"Optimizer: {type(optimizer).__name__}"]
    for i, pg in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        lr       = pg.get("lr", "?")
        wd       = pg.get("weight_decay", "?")
        lines.append(
            f"  Group {i}: {n_params:>10,} params | "
            f"lr={lr} | wd={wd}"
        )
    total = sum(
        sum(p.numel() for p in pg["params"])
        for pg in optimizer.param_groups
    )
    lines.append(f"  Total: {total:,} params")
    return "
".join(lines)


def schedule_preview(
    schedule,
    total_steps: int,
    n_points: int = 10,
) -> list[tuple[int, float]]:
    """
    Preview an LR schedule at evenly spaced steps.

    Args:
        schedule:    A callable ``schedule(step) -> float``.
        total_steps: Total number of training steps.
        n_points:    Number of steps to sample.

    Returns:
        List of ``(step, lr)`` tuples.
    """
    steps = [int(i * total_steps / (n_points - 1)) for i in range(n_points)]
    return [(s, schedule(s)) for s in steps]
