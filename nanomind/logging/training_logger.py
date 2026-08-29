"""
nanomind/logging/training_logger.py — High-level training logger multiplexer.

TrainingLogger fans out all log calls to multiple backends simultaneously.
It also manages the MetricsBuffer for step-level accumulation and provides
convenience methods for training-specific events (epoch start, validation, etc.)
"""

from __future__ import annotations

import time
import torch.nn as nn

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.factory import build_loggers
from nanomind.logging.metrics import MetricsBuffer
from nanomind.utils.logger import get_logger


class TrainingLogger:
    """
    High-level training logger that multiplexes across multiple backends.

    Accepts a :class:`LogConfig` and fans out all log calls to every enabled
    backend (console, TensorBoard, W&B) simultaneously.

    Args:
        cfg: Logging configuration.

    Example::

        log_cfg = LogConfig(backend=["console", "tensorboard"], run_name="my_run")
        logger  = TrainingLogger(log_cfg)
        logger.log_config({"lr": 3e-4, "batch_size": 32})

        for step, (x, y) in enumerate(loader):
            loss = train_step(x, y)
            logger.log_step(step, {"train/loss": loss, "lr": get_lr()})

        logger.finish()
    """

    def __init__(self, cfg: LogConfig) -> None:
        self.cfg      = cfg
        self._loggers = build_loggers(cfg)
        self._buffer  = MetricsBuffer()
        self._t0      = time.perf_counter()
        self._log     = get_logger("training.logger")

    def log_config(self, config: dict) -> None:
        """Broadcast config to all backends at run start."""
        for lg in self._loggers:
            try:
                lg.log_config(config)
            except Exception as e:
                self._log.warning(f"log_config failed for {type(lg).__name__}: {e}")

    def log_step(
        self,
        step:    int,
        metrics: dict[str, float],
    ) -> None:
        """
        Accumulate step metrics; broadcast to backends every ``log_interval`` steps.

        Args:
            step:    Current training step.
            metrics: Scalar metrics dict (e.g. ``{"train/loss": 2.3, "lr": 3e-4}``).
        """
        self._buffer.update(metrics)

        if step > 0 and step % self.cfg.log_interval == 0:
            averaged = self._buffer.averages()
            averaged["step"] = step
            averaged["elapsed_s"] = time.perf_counter() - self._t0
            self._broadcast(averaged, step)
            self._buffer.reset()

    def log_validation(self, step: int, metrics: dict[str, float]) -> None:
        """Log validation metrics immediately (not buffered)."""
        prefixed = {f"val/{k}": v for k, v in metrics.items()}
        self._broadcast(prefixed, step)

    def log_histogram(self, name: str, values, step: int) -> None:
        """Log a parameter histogram to all backends that support it."""
        for lg in self._loggers:
            try:
                lg.log_histogram(name, values, step)
            except Exception:
                pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text sample to all backends."""
        for lg in self._loggers:
            try:
                lg.log_text(tag, text, step)
            except Exception:
                pass

    def log_model_params(self, model: nn.Module, step: int) -> None:
        """Log weight histograms for all named parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"params/{name}", param.data, step)

    def finish(self) -> None:
        """Flush and close all backends."""
        for lg in self._loggers:
            try:
                lg.finish()
            except Exception:
                pass

    def _broadcast(self, metrics: dict[str, float], step: int) -> None:
        for lg in self._loggers:
            try:
                lg.log_scalars(metrics, step)
            except Exception as e:
                self._log.warning(f"log_scalars failed for {type(lg).__name__}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
