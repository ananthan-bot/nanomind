"""
nanomind/utils/seed.py — Reproducibility utilities for NanoMind.

Sets seeds across Python, NumPy (if installed), and PyTorch (CPU + CUDA)
so that training runs are deterministic and reproducible.
"""

import random
from typing import Optional

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for full reproducibility.

    Args:
        seed:          Integer seed value.
        deterministic: If True, also enable CUDA deterministic mode.
                       Slower but ensures bit-exact reproducibility on GPU.

    Example::

        set_seed(42)
        # All subsequent random operations are reproducible
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # NumPy (optional dependency)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_rng_state() -> dict:
    """
    Capture the full RNG state for later restoration.

    Useful for checkpointing so training can be resumed exactly.

    Returns:
        Dictionary with Python, torch, and cuda RNG states.
    """
    state: dict = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """
    Restore RNG state captured by :func:`get_rng_state`.

    Args:
        state: Dictionary returned by :func:`get_rng_state`.
    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
