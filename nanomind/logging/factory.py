"""
nanomind/logging/factory.py — Logger factory: build loggers from LogConfig.
"""

from __future__ import annotations

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.console import ConsoleLogger
from nanomind.logging.tensorboard import TensorBoardLogger
from nanomind.logging.wandb_logger import WandbLogger


def build_loggers(cfg: LogConfig) -> list[BaseLogger]:
    """
    Build a list of logger instances from a :class:`LogConfig`.

    Args:
        cfg: Logging configuration.

    Returns:
        List of instantiated logger backends.
    """
    loggers: list[BaseLogger] = []

    for backend in cfg.backends:
        if backend == "console":
            loggers.append(
                ConsoleLogger(run_name=cfg.run_name, log_interval=cfg.log_interval)
            )
        elif backend == "tensorboard":
            loggers.append(
                TensorBoardLogger(
                    log_dir=cfg.log_dir,
                    run_name=cfg.run_name,
                    log_params=cfg.log_params,
                )
            )
        elif backend == "wandb":
            loggers.append(
                WandbLogger(
                    project=cfg.project,
                    run_name=cfg.run_name,
                    tags=cfg.tags,
                    notes=cfg.notes,
                )
            )

    return loggers
