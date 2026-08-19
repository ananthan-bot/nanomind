"""
nanomind/checkpoint/resume.py — Auto-resume helpers.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from nanomind.checkpoint.io import load_checkpoint
from nanomind.utils.logger import get_logger


def auto_resume(
    out_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[int, dict | None]:
    """
    Automatically resume from the latest checkpoint in ``out_dir``.

    If no checkpoint exists, returns ``(0, None)`` so training starts fresh.

    Args:
        out_dir:   Directory to search for checkpoints.
        model:     Model to restore.
        optimizer: Optimizer to restore (None = skip).
        device:    Target device.

    Returns:
        Tuple of ``(start_step, metadata_or_None)``.

    Example::

        start_step, meta = auto_resume("checkpoints", model, optimizer, device)
        trainer.step = start_step
    """
    log = get_logger("resume")
    candidates = sorted(Path(out_dir).glob("step_*.pt"))
    if not candidates:
        log.info("No checkpoint found — starting training from step 0.")
        return 0, None

    latest = candidates[-1]
    log.info(f"Auto-resuming from: {latest.name}")
    meta = load_checkpoint(latest, model, optimizer, device)
    start_step = meta.get("step", 0) + 1
    log.info(f"Resumed at step {start_step}.")
    return start_step, meta
