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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        """Yield batches endlessly, restarting the loader when exhausted."""
        while True:
            yield from loader

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform one forward + backward pass (no optimizer step).

        Supports AMP via :class:`torch.cuda.amp.GradScaler` when enabled.

        Args:
            x: Input tokens  ``(B, T)``
            y: Target tokens ``(B, T)``

        Returns:
            Loss value as a Python float.
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        if self._scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = self.model(x, y)
            self._scaler.scale(loss).backward()
        else:
            _, loss = self.model(x, y)
            loss.backward()

        return loss.item()

    def _optimizer_step(self) -> None:
        """
        Apply accumulated gradients, clip norms, and step the optimizer.

        Handles both standard and AMP (GradScaler) paths.
        Resets gradients after the update.
        """
        if self._scaler is not None:
            if self.cfg.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
