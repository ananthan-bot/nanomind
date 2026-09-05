"""
nanomind/amp/context.py — Mixed precision autocast context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch

from nanomind.amp.config import AMPConfig


@contextmanager
def mixed_precision_context(
    cfg:    AMPConfig,
    device: str | torch.device = "cpu",
) -> Generator:
    """
    Context manager that enables mixed precision (autocast) when configured.

    Uses ``torch.amp.autocast`` to automatically cast eligible operations to
    the lower-precision dtype while keeping master weights in float32.

    Args:
        cfg:    AMP configuration.
        device: Device type (``"cpu"`` or ``"cuda"``).

    Example::

        amp_cfg = AMPConfig(enabled=True, dtype="bfloat16")
        with mixed_precision_context(amp_cfg, device="cuda"):
            logits, loss = model(x, y)
            # logits and intermediate activations are bfloat16
            # model weights (master copy) remain float32
    """
    device_str = str(device).split(":")[0]   # "cuda:0" → "cuda"

    if cfg.enabled and device_str in ("cuda", "cpu"):
        with torch.amp.autocast(device_type=device_str, dtype=cfg.torch_dtype):
            yield
    else:
        yield


def is_amp_available(device: str | torch.device = "cpu") -> bool:
    """Check whether AMP is available on the given device."""
    device_str = str(device).split(":")[0]
    if device_str == "cuda":
        return torch.cuda.is_available()
    if device_str == "cpu":
        return True   # CPU autocast always available (bfloat16 on modern CPUs)
    return False
