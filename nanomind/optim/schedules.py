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


class CosineDecay(LRSchedule):
    """
    Cosine annealing from ``max_lr`` down to ``min_lr`` over ``total_steps``.

    Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t/T))

    Args:
        max_lr:       Starting (peak) learning rate.
        min_lr:       Minimum learning rate (floor).
        total_steps:  Total number of steps for the decay.
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        total_steps: int,
    ) -> None:
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"CosineDecay(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"total_steps={self.total_steps})"
        )


class WarmupCosine(LRSchedule):
    """
    Linear warmup followed by cosine decay — the default NanoMind schedule.

    This is equivalent to composing :class:`LinearWarmup` with
    :class:`CosineDecay`, but provided as a single convenience class.

    Args:
        max_lr:       Peak learning rate (after warmup, before decay).
        min_lr:       Minimum learning rate at end of cosine decay.
        warmup_steps: Number of linear warmup steps.
        total_steps:  Total training steps (including warmup).
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self._cosine      = CosineDecay(max_lr, min_lr, total_steps - warmup_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        return self._cosine(step - self.warmup_steps)

    def __repr__(self) -> str:
        return (
            f"WarmupCosine(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup={self.warmup_steps}, total={self.total_steps})"
        )


class LinearDecay(LRSchedule):
    """
    Linear decay from ``max_lr`` to ``min_lr`` over ``total_steps``.

    Args:
        max_lr:      Starting learning rate.
        min_lr:      Ending learning rate.
        total_steps: Total number of decay steps.
    """

    def __init__(self, max_lr: float, min_lr: float, total_steps: int) -> None:
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return self.max_lr - progress * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"LinearDecay(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"total_steps={self.total_steps})"
        )


class WarmupLinear(LRSchedule):
    """
    Linear warmup followed by linear decay.

    Args:
        max_lr:       Peak learning rate after warmup.
        min_lr:       Minimum LR at end of decay.
        warmup_steps: Number of warmup steps.
        total_steps:  Total training steps (warmup + decay).
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        decay_step  = step - self.warmup_steps
        if decay_step >= decay_steps:
            return self.min_lr
        progress = decay_step / decay_steps
        return self.max_lr - progress * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"WarmupLinear(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup={self.warmup_steps}, total={self.total_steps})"
        )
