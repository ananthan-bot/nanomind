"""
nanomind/quant/checkpoint.py — Save and load quantized model checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from nanomind.utils.logger import get_logger

log = get_logger("quant.checkpoint")


def save_quantized_checkpoint(
    model:    nn.Module,
    path:     str | Path,
    metadata: dict | None = None,
) -> Path:
    """
    Save a quantized model checkpoint.

    The INT8 weights (buffers) and float bias parameters are both saved.
    The resulting file is ~4x smaller than the equivalent float32 checkpoint.

    Args:
        model:    Quantized model.
        path:     Destination ``.pt`` file path.
        metadata: Optional metadata dict (config, step, etc.)

    Returns:
        Path of saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata":   metadata or {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)

    size_mb = path.stat().st_size / 1024 ** 2
    log.info(f"Saved quantized checkpoint: {path.name} ({size_mb:.2f} MB)")
    return path


def load_quantized_checkpoint(
    model:  nn.Module,
    path:   str | Path,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict:
    """
    Load a quantized model checkpoint.

    The model must already have been quantized (QuantizedLinear layers in place)
    before calling this function.

    Args:
        model:  Quantized model with matching architecture.
        path:   Path to ``.pt`` checkpoint.
        device: Device to map tensors to.
        strict: Whether to require exact state dict key matching.

    Returns:
        Metadata dict from the checkpoint.
    """
    path    = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["state_dict"], strict=strict)
    log.info(f"Loaded quantized checkpoint from: {path.name}")
    return payload.get("metadata", {})
