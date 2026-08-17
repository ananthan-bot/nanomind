"""NanoMind trainer sub-package.

Primary exports:
    - :class:`Trainer`     — full training loop with eval, logging, early stop
    - :class:`TrainConfig` — training hyperparameter configuration
    - :func:`estimate_training_time` — pre-training throughput benchmark
"""

from nanomind.trainer.config import TrainConfig
from nanomind.trainer.trainer import Trainer
from nanomind.trainer.estimate import estimate_training_time

__all__ = ["Trainer", "TrainConfig", "estimate_training_time"]
