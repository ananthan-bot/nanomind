"""
nanomind/trainer/trainer.py — Training loop for NanoMind.

The Trainer class owns the full training lifecycle:
    - train_step()     : one forward + backward + optimizer step
    - eval_step()      : one forward without gradients
    - estimate_loss()  : average loss over N batches
    - train()          : the full training loop
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.trainer.config import TrainConfig
from nanomind.utils.logger import get_logger
from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr
from nanomind.utils.timer import Timer, tokens_per_second


class Trainer:
    """
    Handles the NanoMind training loop.

    Args:
        model:        The :class:`~nanomind.model.NanoMind` model.
        optimizer:    Configured optimizer (e.g. AdamW).
        train_loader: DataLoader for training batches.
        val_loader:   DataLoader for validation batches.
        cfg:          :class:`~nanomind.trainer.TrainConfig`.
        device:       Resolved :class:`torch.device`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.log          = get_logger("trainer")

        # State
        self.step:     int   = 0
        self.best_val: float = float("inf")
        self._timer          = Timer()

        # AMP scaler (only active when use_amp=True and CUDA available)
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if cfg.use_amp and device.type == "cuda"
            else None
        )
