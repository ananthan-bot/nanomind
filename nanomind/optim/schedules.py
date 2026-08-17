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
