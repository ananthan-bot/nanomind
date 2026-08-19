"""
nanomind/checkpoint/io.py — Low-level checkpoint save/load functions.

Checkpoints are stored as PyTorch ``.pt`` files alongside a ``.json``
metadata file. Writes are atomic: the payload is written to a ``.tmp``
file first, then renamed to the final path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from nanomind.checkpoint.metadata import make_metadata, save_metadata, load_metadata


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    train_loss: float = float("nan"),
    val_loss: float = float("nan"),
    model_config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Save a training checkpoint atomically.

    The checkpoint ``.pt`` file contains:
    - ``model_state``:     model weights
    - ``optimizer_state``: optimizer state (if provided)
    - ``step``:            training step
    - ``metadata``:        metadata dict

    Writes to a ``.tmp`` file first, then renames for atomicity.

    Args:
        path:         Destination path (e.g. ``checkpoints/step_1000.pt``).
        model:        Model to checkpoint.
        optimizer:    Optimizer (None = skip optimizer state).
        step:         Current training step.
        train_loss:   Current training loss.
        val_loss:     Current validation loss.
        model_config: Model config dict for metadata.
        extra:        Extra metadata fields.

    Returns:
        The final checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = make_metadata(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        model_config=model_config or {},
        extra=extra,
    )

    payload = {
        "model_state": model.state_dict(),
        "step":        step,
        "metadata":    meta,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    # Atomic write: .tmp -> final
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)

    # Companion metadata JSON
    save_metadata(meta, path.with_suffix(".json"))
    return path
