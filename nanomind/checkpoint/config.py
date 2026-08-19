"""
nanomind/checkpoint/config.py — Checkpoint configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    """
    Configuration for the :class:`~nanomind.checkpoint.CheckpointManager`.

    Attributes:
        out_dir:        Directory to write checkpoints.
        save_interval:  Save a checkpoint every N training steps.
        keep_last_n:    Keep only the N most recent checkpoints (0 = keep all).
        save_best:      Always keep the checkpoint with the lowest val loss.
        save_optimizer: Whether to include optimizer state in checkpoints.
    """

    out_dir:         str  = "checkpoints"
    save_interval:   int  = 500
    keep_last_n:     int  = 3
    save_best:       bool = True
    save_optimizer:  bool = True

    def __post_init__(self) -> None:
        assert self.save_interval > 0, "save_interval must be positive"
        assert self.keep_last_n >= 0, "keep_last_n must be >= 0"
