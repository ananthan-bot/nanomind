"""
nanomind/cli/device.py — Device resolution for CLI commands.
"""

from __future__ import annotations

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"auto"`` (pick best), ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
