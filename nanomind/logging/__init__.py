"""NanoMind training logging sub-package.

Provides unified logging to console, TensorBoard, and Weights & Biases,
with a single high-level :class:`TrainingLogger` that multiplexes across
all enabled backends simultaneously.

Backends:
    - ``"console"``     — always available; rich formatted terminal output
    - ``"tensorboard"`` — requires ``pip install tensorboard``
    - ``"wandb"``       — requires ``pip install wandb`` + ``wandb login``

Primary exports:
    - :class:`TrainingLogger`   — high-level multiplex logger (recommended entry point)
    - :class:`LogConfig`        — backend, log_dir, project, run_name, log_interval
    - :class:`MetricsBuffer`    — accumulate and average metrics over steps
    - :class:`ConsoleLogger`    — formatted stdout logger
    - :class:`TensorBoardLogger`— TensorBoard backend
    - :class:`WandbLogger`      — Weights & Biases backend
    - :func:`build_loggers`     — instantiate backends from LogConfig
"""

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.console import ConsoleLogger
from nanomind.logging.tensorboard import TensorBoardLogger
from nanomind.logging.wandb_logger import WandbLogger
from nanomind.logging.metrics import MetricsBuffer
from nanomind.logging.factory import build_loggers
from nanomind.logging.training_logger import TrainingLogger

__all__ = [
    "LogConfig",
    "BaseLogger",
    "ConsoleLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "MetricsBuffer",
    "build_loggers",
    "TrainingLogger",
]
