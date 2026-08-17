"""
nanomind/trainer/config.py — Training configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """
    All hyperparameters needed to run NanoMind training.

    Attributes:
        max_iters:        Total number of training iterations.
        eval_interval:    Run validation every N iterations.
        eval_iters:       Number of batches to average for eval loss.
        log_interval:     Print training loss every N iterations.
        grad_accum_steps: Accumulate gradients over N steps before update.
        grad_clip:        Max gradient norm (0 = no clipping).
        use_amp:          Use automatic mixed precision (CUDA only).
        early_stop_patience: Stop if val loss doesn't improve for N evals (0 = off).
        seed:             Random seed.
        device:           Device string — ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.
        out_dir:          Directory to write checkpoints and logs.
    """

    max_iters:           int   = 5000
    eval_interval:       int   = 200
    eval_iters:          int   = 50
    log_interval:        int   = 10
    grad_accum_steps:    int   = 1
    grad_clip:           float = 1.0
    use_amp:             bool  = False
    early_stop_patience: int   = 0
    seed:                int   = 42
    device:              str   = "auto"
    out_dir:             str   = "checkpoints"

    def __post_init__(self) -> None:
        assert self.max_iters > 0
        assert self.eval_interval > 0
        assert self.grad_accum_steps >= 1
        assert self.grad_clip >= 0.0
