"""
nanomind/logging/base.py — Abstract base class for all training loggers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Abstract base class for NanoMind training loggers.

    All logger backends (console, TensorBoard, W&B) implement this interface,
    allowing the :class:`TrainingLogger` to multiplex across multiple backends.
    """

    @abstractmethod
    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at a given step."""

    @abstractmethod
    def log_config(self, config: dict) -> None:
        """Log hyperparameter configuration at the start of training."""

    def log_histogram(self, name: str, values, step: int) -> None:
        """Log a histogram (optional — not all backends support it)."""

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text sample (optional — not all backends support it)."""

    def finish(self) -> None:
        """Called at end of training run to flush and close the logger."""
