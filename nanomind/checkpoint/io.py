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


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path:      Path to the ``.pt`` checkpoint file.
        model:     Model to restore weights into.
        optimizer: Optimizer to restore state into (None = skip).
        device:    Device to map tensors to (None = use saved device).
        strict:    Whether to require exact key matching in state dict.

    Returns:
        The metadata dict from the checkpoint.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload.get("metadata", {})


def load_for_inference(
    path: str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> dict:
    """
    Load only model weights — no optimizer state needed for inference.

    Args:
        path:   Path to the ``.pt`` checkpoint file.
        model:  Model to restore weights into.
        device: Target device.

    Returns:
        Metadata dict from the checkpoint.
    """
    return load_checkpoint(path, model, optimizer=None, device=device)
