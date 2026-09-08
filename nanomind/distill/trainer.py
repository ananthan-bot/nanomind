"""
nanomind/distill/trainer.py — Knowledge Distillation training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.distill.config import DistillConfig
from nanomind.distill.loss import distillation_loss
from nanomind.utils.logger import get_logger

log = get_logger("distill.trainer")


class DistillTrainer:
    """
    Knowledge distillation trainer: train a student to mimic a teacher.

    Args:
        teacher:   Large pre-trained teacher model (frozen during distillation).
        student:   Smaller student model to train.
        optimizer: Optimizer for the student.
        cfg:       Distillation configuration.
        device:    Training device.

    Example::

        trainer = DistillTrainer(teacher, student, optimizer,
                                  DistillConfig(temperature=4.0, alpha=0.5))
        for epoch in range(5):
            metrics = trainer.train_epoch(loader)
            print(metrics)
    """

    def __init__(
        self,
        teacher:   nn.Module,
        student:   nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       DistillConfig | None = None,
        device:    torch.device | str = "cpu",
    ) -> None:
        self.teacher   = teacher.eval()
        self.student   = student
        self.optimizer = optimizer
        self.cfg       = cfg or DistillConfig()
        self.device    = torch.device(device)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        log.info(
            f"DistillTrainer: T={self.cfg.temperature}, "
            f"alpha={self.cfg.alpha}, "
            f"teacher={sum(p.numel() for p in teacher.parameters()):,} params, "
            f"student={sum(p.numel() for p in student.parameters()):,} params"
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        One distillation gradient step.

        Args:
            x: Input token IDs ``(B, T)``.
            y: Target token IDs ``(B, T)``.

        Returns:
            Dict with ``loss``, ``ce_loss``, ``kd_loss``.
        """
        x, y = x.to(self.device), y.to(self.device)

        # Teacher logits (no grad, cached)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(x)

        # Student logits
        student_logits, _ = self.student(x)

        # Flatten for loss computation
        N, T, V = student_logits.shape
        s_flat  = student_logits.reshape(N * T, V)
        t_flat  = teacher_logits.reshape(N * T, V)
        y_flat  = y.reshape(N * T)

        loss, info = distillation_loss(
            s_flat, t_flat, y_flat,
            temperature=self.cfg.temperature,
            alpha=self.cfg.alpha,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        return info

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one full distillation epoch."""
        self.student.train()
        totals: dict = {}
        steps = 0
        for x, y in loader:
            m = self.train_step(x, y)
            for k, v in m.items():
                totals[k] = totals.get(k, 0.0) + v
            steps += 1
        return {k: v / max(steps, 1) for k, v in totals.items()} | {"steps": steps}

    def compression_ratio(self) -> float:
        """Student / Teacher parameter ratio."""
        t = sum(p.numel() for p in self.teacher.parameters())
        s = sum(p.numel() for p in self.student.parameters())
        return s / max(t, 1)
