"""
nanomind/lora/checkpoint.py — Save and load LoRA-only checkpoints.

LoRA checkpoints store only the A and B matrices — typically just a few MB
compared to hundreds of MB for the full model. The base model is loaded
separately and the LoRA weights are applied on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from nanomind.lora.utils import get_lora_state_dict
from nanomind.utils.logger import get_logger

log = get_logger("lora.checkpoint")


def save_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """
    Save only the LoRA weights (A and B matrices) to a file.

    Args:
        model:    Model with LoRA layers.
        path:     Destination file path (e.g. ``lora_weights.pt``).
        metadata: Optional dict with experiment info (step, loss, config).

    Returns:
        Path of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lora_state = get_lora_state_dict(model)
    payload = {
        "lora_state": lora_state,
        "metadata":   metadata or {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)

    n_params = sum(v.numel() for v in lora_state.values())
    size_kb   = path.stat().st_size / 1024
    log.info(f"Saved LoRA checkpoint: {path.name} ({n_params:,} params, {size_kb:.1f} KB)")
    return path


def load_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict:
    """
    Load LoRA weights into a model with LoRA layers already injected.

    Args:
        model:   Model with LoRA layers (injected but not trained).
        path:    Path to the LoRA ``.pt`` checkpoint.
        device:  Device to map weights to.
        strict:  Whether to require exact LoRA key matching.

    Returns:
        Metadata dict from the checkpoint.
    """
    path    = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    lora_state = payload.get("lora_state", payload)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)

    lora_missing = [k for k in missing    if "lora" in k]
    non_lora_unexp = [k for k in unexpected if "lora" not in k]

    if strict and lora_missing:
        raise RuntimeError(f"Missing LoRA keys: {lora_missing}")
    if non_lora_unexp:
        log.warning(f"Unexpected non-LoRA keys: {non_lora_unexp}")

    log.info(f"Loaded LoRA checkpoint from: {path.name}")
    return payload.get("metadata", {})
