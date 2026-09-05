"""
nanomind/amp/trainer.py — AMP-aware training step with gradient accumulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.amp.config import AMPConfig
from nanomind.amp.context import mixed_precision_context
from nanomind.amp.scaler import NanoGradScaler
from nanomind.amp.accumulation import GradAccumulator
from nanomind.utils.logger import get_logger

log = get_logger("amp.trainer")


class AMPTrainer:
    """
    Mixed precision training loop with gradient accumulation and grad clipping.

    Integrates AMP autocast, GradScaler, gradient accumulation, and gradient
    norm clipping into a clean ``train_epoch()`` API.

    Args:
        model:     Language model.
        optimizer: PyTorch optimizer.
        cfg:       AMP configuration.
        device:    Training device.

    Example::

        trainer = AMPTrainer(model, optimizer, AMPConfig(dtype="bfloat16",
                                                          grad_accum_steps=4))
        for epoch in range(n_epochs):
            metrics = trainer.train_epoch(train_loader)
            print(metrics)
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       AMPConfig,
        device:    torch.device | str = "cpu",
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg
        self.device    = torch.device(device)
        self.scaler    = NanoGradScaler(cfg)
        self.accum     = GradAccumulator(cfg.grad_accum_steps)

        log.info(
            f"AMPTrainer: dtype={cfg.dtype}, "
            f"accum={cfg.grad_accum_steps}, "
            f"clip={cfg.clip_grad_norm}"
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Run one micro-batch forward + backward.

        Handles loss scaling, accumulation dividing, and skips optimizer.step()
        for non-final accumulation steps.

        Args:
            x: Input token IDs ``(B, T)``.
            y: Target token IDs ``(B, T)``.

        Returns:
            Raw (unscaled) loss value for this micro-batch.
        """
        x, y = x.to(self.device), y.to(self.device)
        is_last = self.accum.should_step()

        # Forward with autocast
        with mixed_precision_context(self.cfg, self.device):
            _, loss = self.model(x, y)

        # Scale loss for accumulation (mean, not sum)
        scaled_loss = loss / self.accum.loss_scale

        # Backward
        self.scaler.scale(scaled_loss).backward()

        if is_last:
            # Unscale before clipping
            if self.cfg.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.accum.reset()
        else:
            self.accum.step()

        return loss.item()

    def train_epoch(self, loader: DataLoader) -> dict:
        """
        Run one full training epoch.

        Args:
            loader: DataLoader of (x, y) batches.

        Returns:
            Dict with ``loss`` (mean), ``steps``, ``scale``.
        """
        self.model.train()
        total_loss, steps = 0.0, 0
        for x, y in loader:
            total_loss += self.train_step(x, y)
            steps      += 1
        return {
            "loss":  total_loss / max(steps, 1),
            "steps": steps,
            "scale": self.scaler.scale_factor,
        }
