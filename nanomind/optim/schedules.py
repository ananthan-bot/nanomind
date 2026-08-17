"""
nanomind/optim/schedules.py — Learning rate schedule functions for NanoMind.

All schedules are implemented as plain callables: ``lr = schedule(step)``.
This keeps them independent of any optimizer or framework scheduler,
making them easy to test and compose.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRSchedule(ABC):
    """
    Abstract base class for all NanoMind LR schedules.

    Subclasses implement ``__call__(step) -> float``.
    """

    @abstractmethod
    def __call__(self, step: int) -> float:
        """Return the learning rate for the given training step."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConstantLR(LRSchedule):
    """
    Constant learning rate — returns the same LR for every step.

    Args:
        lr: The fixed learning rate.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, step: int) -> float:
        return self.lr

    def __repr__(self) -> str:
        return f"ConstantLR(lr={self.lr})"


class LinearWarmup(LRSchedule):
    """
    Linear warmup from 0 to ``max_lr`` over ``warmup_steps`` steps.

    After warmup, delegates to ``post_warmup_schedule`` (or holds ``max_lr``).

    Args:
        max_lr:                Peak learning rate after warmup.
        warmup_steps:          Number of warmup steps.
        post_warmup_schedule:  Optional schedule called after warmup.
    """

    def __init__(
        self,
        max_lr: float,
        warmup_steps: int,
        post_warmup_schedule: LRSchedule | None = None,
    ) -> None:
        self.max_lr               = max_lr
        self.warmup_steps         = warmup_steps
        self.post_warmup_schedule = post_warmup_schedule

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear ramp: 0 -> max_lr
            return self.max_lr * (step + 1) / self.warmup_steps
        if self.post_warmup_schedule is not None:
            return self.post_warmup_schedule(step)
        return self.max_lr

    def __repr__(self) -> str:
        return (
            f"LinearWarmup(max_lr={self.max_lr}, "
            f"warmup_steps={self.warmup_steps})"
        )
