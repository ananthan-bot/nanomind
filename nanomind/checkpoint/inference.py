"""
nanomind/checkpoint/inference.py — Lightweight inference-only checkpoints.

Saves only model weights (no optimizer state), producing smaller files
suitable for deployment and sharing.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn


def save_for_inference(
    path: str | Path,
    model: nn.Module,
    model_config: dict | None = None,
    step: int = 0,
) -> Path:
    """
    Save a lightweight inference-only checkpoint (weights only).

    Args:
        path:         Destination file path.
        model:        Model whose weights to save.
        model_config: Model configuration dict (stored in file for loading).
        step:         Training step (metadata only).

    Returns:
        Path of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state":  model.state_dict(),
        "model_config": model_config or {},
        "step":         step,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def load_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> dict:
    """
    Load an inference-only checkpoint into a model.

    Args:
        path:   Path to the inference checkpoint.
        model:  Model to restore.
        device: Target device.

    Returns:
        Dict with ``"model_config"`` and ``"step"``.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return {
        "model_config": payload.get("model_config", {}),
        "step":         payload.get("step", 0),
    }
