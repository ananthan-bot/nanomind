"""
nanomind/continual/trainer.py — Unified continual learning trainer.

Orchestrates training across sequential tasks with the chosen strategy.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.continual.config import ContinualConfig
from nanomind.continual.ewc import EWC
from nanomind.continual.si import SynapticIntelligence
from nanomind.continual.replay import ReplayBuffer
from nanomind.continual.packnet import PackNet
from nanomind.utils.logger import get_logger

log = get_logger("continual.trainer")


class ContinualTrainer:
    """
    Unified trainer for continual learning strategies.

    Supports: ``"ewc"``, ``"si"``, ``"replay"``, ``"packnet"``, ``"naive"``.

    Args:
        model:  Language model.
        cfg:    :class:`ContinualConfig`.

    Example::

        trainer = ContinualTrainer(model, ContinualConfig(strategy="ewc"))
        for task_id, task_batches in enumerate(tasks):
            trainer.train_task(task_id, task_batches)
        metrics = trainer.evaluator.compute()
    """

    def __init__(self, model: nn.Module, cfg: ContinualConfig) -> None:
        self.model   = model
        self.cfg     = cfg
        self._ewc:    EWC | None = None
        self._si:     SynapticIntelligence | None = None
        self._replay: ReplayBuffer | None = None
        self._packnet: PackNet | None     = None
        self._task_logs: list[dict] = []

        if cfg.strategy == "ewc":
            self._ewc = EWC(model, lambda_=cfg.ewc_lambda, n_samples=cfg.ewc_n_samples)
        elif cfg.strategy == "si":
            self._si  = SynapticIntelligence(model, lambda_=cfg.ewc_lambda)
        elif cfg.strategy == "replay":
            self._replay = ReplayBuffer(max_size=cfg.replay_buffer_size)
        elif cfg.strategy == "packnet":
            self._packnet = PackNet(model, prune_ratio=cfg.packnet_prune_ratio)

    def _build_opt(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_task(
        self,
        task_id: int,
        batches: list,
        epochs:  int = 3,
        lr:      float = 1e-3,
    ) -> dict:
        """
        Train on one task.

        Args:
            task_id: Task index (0-based).
            batches: List of ``(x, y)`` batches.
            epochs:  Training epochs.
            lr:      Learning rate.

        Returns:
            Log dict with ``task_id``, ``final_loss``, ``strategy``.
        """
        opt      = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        if self._si:
            self._si.begin_task()

        total_loss = 0.0
        n_steps    = 0

        for _ in range(epochs):
            for x, y in batches:
                opt.zero_grad()
                logits, loss = self.model(x, y)
                if loss is None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                # Add regularisation
                if self._ewc:
                    loss = loss + self._ewc.penalty()
                if self._si:
                    loss = loss + self._si.penalty()

                # Replay mixing
                if self._replay and len(self._replay) > 0:
                    try:
                        rx, ry = self._replay.sample_batch(max(1, int(x.shape[0] * self.cfg.replay_fraction)))
                        r_logits, r_loss = self.model(rx, ry)
                        if r_loss is None:
                            r_loss = F.cross_entropy(r_logits.view(-1, r_logits.size(-1)), ry.view(-1))
                        loss = loss + r_loss
                    except RuntimeError:
                        pass

                loss.backward()

                if self._si:
                    self._si.update_importances()
                if self._packnet:
                    self._packnet.freeze_past_weights()

                opt.step()
                total_loss += loss.item()
                n_steps    += 1

        # Post-task hooks
        if self._ewc:
            self._ewc.register_task(batches, task_id)
        if self._si:
            self._si.end_task()
        if self._replay:
            for x, y in batches:
                self._replay.add_batch(x, y, task_id)
        if self._packnet:
            self._packnet.prune_and_pack(task_id)

        entry = {
            "task_id":    task_id,
            "final_loss": round(total_loss / max(n_steps, 1), 6),
            "strategy":   self.cfg.strategy,
        }
        self._task_logs.append(entry)
        log.info(f"Task {task_id} done: loss={entry['final_loss']:.4f}")
        return entry

    @property
    def task_logs(self) -> list[dict]:
        return list(self._task_logs)
