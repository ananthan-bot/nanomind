"""
nanomind/amp/memory.py — Memory profiling utilities for AMP training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Generator


def get_memory_mb(device: torch.device | str = "cpu") -> float:
    """
    Get current GPU allocated memory in MB (0 on CPU).

    Args:
        device: torch.device or device string.

    Returns:
        Allocated memory in MB, or 0.0 if not CUDA.
    """
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.memory_allocated(d) / (1024 ** 2)
    return 0.0


def get_peak_memory_mb(device: torch.device | str = "cpu") -> float:
    """
    Get peak GPU memory allocated since last reset (MB).

    Args:
        device: torch.device or device string.

    Returns:
        Peak allocated memory in MB, or 0.0 if not CUDA.
    """
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.max_memory_allocated(d) / (1024 ** 2)
    return 0.0


@contextmanager
def memory_tracker(
    device: torch.device | str = "cpu",
    label:  str = "",
) -> Generator[dict, None, None]:
    """
    Context manager that measures memory usage of a code block.

    Args:
        device: Device to measure.
        label:  Optional label for the output dict.

    Yields:
        Dict with ``before_mb``, ``after_mb``, ``delta_mb`` (filled on exit).

    Example::

        with memory_tracker("cuda", label="forward") as mem:
            logits, loss = model(x, y)
        print(f"Forward used: {mem['delta_mb']:.1f} MB")
    """
    d      = torch.device(device)
    result = {"label": label, "before_mb": 0.0, "after_mb": 0.0, "delta_mb": 0.0}
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)
        result["before_mb"] = get_memory_mb(d)

    yield result

    if d.type == "cuda":
        result["after_mb"] = get_memory_mb(d)
        result["delta_mb"] = result["after_mb"] - result["before_mb"]


def model_parameter_memory_mb(model: nn.Module) -> dict:
    """
    Compute memory used by model parameters and buffers.

    Args:
        model: PyTorch model.

    Returns:
        Dict with ``params_mb``, ``buffers_mb``, ``total_mb``, ``n_params``.
    """
    params_bytes  = sum(p.nbytes for p in model.parameters())
    buffers_bytes = sum(b.nbytes for b in model.buffers())
    return {
        "params_mb":  params_bytes  / (1024 ** 2),
        "buffers_mb": buffers_bytes / (1024 ** 2),
        "total_mb":   (params_bytes + buffers_bytes) / (1024 ** 2),
        "n_params":   sum(p.numel() for p in model.parameters()),
    }
