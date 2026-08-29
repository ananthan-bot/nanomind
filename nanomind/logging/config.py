"""
nanomind/logging/config.py — Logging configuration for training runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """
    Configuration for training experiment logging.

    Attributes:
        backend:       Logging backend(s). Options: ``"console"``, ``"tensorboard"``,
                       ``"wandb"``, or a list of multiple backends.
        log_dir:       Directory for TensorBoard event files and local logs.
        project:       W&B project name (used when backend includes ``"wandb"``).
        run_name:      Human-readable name for this training run.
        log_interval:  Log training metrics every N steps.
        log_grad_norm: Whether to log gradient norm at each log step.
        log_lr:        Whether to log learning rate at each log step.
        log_params:    Whether to log parameter histograms (expensive; TB only).
        tags:          Optional list of tags for W&B run organisation.
        notes:         Optional notes string for W&B.
    """

    backend:       str | list[str]  = "console"
    log_dir:       str              = "logs"
    project:       str              = "nanomind"
    run_name:      str              = "run"
    log_interval:  int              = 50
    log_grad_norm: bool             = True
    log_lr:        bool             = True
    log_params:    bool             = False
    tags:          list[str]        = field(default_factory=list)
    notes:         str              = ""

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = [self.backend]
        valid = {"console", "tensorboard", "wandb"}
        for b in self.backend:
            assert b in valid, f"Unknown backend '{b}'. Choose from {valid}."
        assert self.log_interval >= 1

    @property
    def backends(self) -> list[str]:
        return self.backend if isinstance(self.backend, list) else [self.backend]
