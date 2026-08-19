"""
nanomind/checkpoint/manager.py — High-level checkpoint manager.

CheckpointManager wraps the low-level save/load functions and adds:
- Automatic naming by step number
- Retention policy (keep last N)
- Best checkpoint tracking
- Auto-resume from latest
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from nanomind.checkpoint.config import CheckpointConfig
from nanomind.checkpoint.io import save_checkpoint, load_checkpoint
from nanomind.utils.logger import get_logger


class CheckpointManager:
    """
    Manages checkpoint saving, tracking, and cleanup.

    Args:
        cfg: :class:`~nanomind.checkpoint.CheckpointConfig` instance.
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg      = cfg
        self.out_dir  = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log      = get_logger("checkpoint")
        self._best_val: float = float("inf")
        self._saved:    list[Path] = []
