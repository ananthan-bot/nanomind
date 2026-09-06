"""
nanomind/rlhf/rm_trainer.py — Reward Model training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.rlhf.reward_model import RewardModel
from nanomind.rlhf.preference_loss import preference_loss, reward_stats
from nanomind.utils.logger import get_logger

log = get_logger("rlhf.rm_trainer")


class RewardModelTrainer:
    """
    Training loop for the Reward Model on preference data.

    Args:
        model:     RewardModel to train.
        optimizer: PyTorch optimizer.
        device:    Training device.

    Example::

        trainer = RewardModelTrainer(rm, optimizer, device)
        metrics = trainer.train_epoch(preference_loader)
    """

    def __init__(
        self,
        model:     RewardModel,
        optimizer: torch.optim.Optimizer,
        device:    torch.device | str = "cpu",
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.device    = torch.device(device)

    def train_step(
        self,
        chosen:   torch.Tensor,
        rejected: torch.Tensor,
    ) -> dict:
        """
        One gradient step on a (chosen, rejected) preference batch.

        Args:
            chosen:   ``(B, T)`` chosen completion token IDs.
            rejected: ``(B, T)`` rejected completion token IDs.

        Returns:
            Dict with ``loss`` and ``accuracy``.
        """
        chosen, rejected = chosen.to(self.device), rejected.to(self.device)

        r_w  = self.model(chosen)
        r_l  = self.model(rejected)
        loss = preference_loss(r_w, r_l)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {**reward_stats(r_w.detach(), r_l.detach()), "loss": loss.item()}

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one full training epoch over the preference dataset."""
        self.model.train()
        total_loss, total_acc, steps = 0.0, 0.0, 0
        for chosen, rejected in loader:
            m = self.train_step(chosen, rejected)
            total_loss += m["loss"]
            total_acc  += m["accuracy"]
            steps      += 1
        return {
            "loss":     total_loss / max(steps, 1),
            "accuracy": total_acc  / max(steps, 1),
            "steps":    steps,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on a preference dataset without gradient updates."""
        self.model.eval()
        total_loss, total_acc, steps = 0.0, 0.0, 0
        for chosen, rejected in loader:
            chosen, rejected = chosen.to(self.device), rejected.to(self.device)
            r_w  = self.model(chosen)
            r_l  = self.model(rejected)
            loss = preference_loss(r_w, r_l)
            total_loss += loss.item()
            total_acc  += (r_w > r_l).float().mean().item()
            steps      += 1
        return {
            "val_loss":     total_loss / max(steps, 1),
            "val_accuracy": total_acc  / max(steps, 1),
        }
