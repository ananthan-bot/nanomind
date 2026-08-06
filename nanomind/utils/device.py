"""
nanomind/utils/device.py — Device detection and management for NanoMind.

Automatically selects the best available compute device:
  CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
"""

import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``,
                or ``"mps"``. When ``"auto"``, the best available
                device is selected automatically.

    Returns:
        A :class:`torch.device` object ready to pass to ``.to()``.

    Example::

        device = get_device()          # auto-select
        device = get_device("cuda:1")  # explicit GPU
        model = model.to(device)
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """
    Return a human-readable description of a device.

    Args:
        device: A :class:`torch.device`.

    Returns:
        String describing the device (e.g. ``"CUDA — NVIDIA A100"``,
        ``"Apple MPS"``, ``"CPU"``)
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory
        mem_gb = mem / (1024 ** 3)
        return f"CUDA — {name} ({mem_gb:.1f} GB)"

    if device.type == "mps":
        return "Apple Silicon MPS"

    return "CPU"


def is_cuda(device: torch.device) -> bool:
    """Return True if the device is a CUDA GPU."""
    return device.type == "cuda"
