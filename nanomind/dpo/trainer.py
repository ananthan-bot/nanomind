"""
nanomind/dpo/trainer.py — DPO training loop.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.dpo.config import DPOConfig
from nanomind.dpo.loss import compute_log_probs, dpo_loss
from nanomind.utils.logger import get_logger

log = get_logger("dpo.trainer")


class DPOTrainer:
    """
    DPO fine-tuning trainer.

    Maintains a frozen reference model (copy of initial policy) alongside
    the trainable policy model. Computes DPO loss by comparing log-ratios.

    Args:
        model:     Policy model to fine-tune.
        optimizer: PyTorch optimizer.
        cfg:       DPO configuration.
        device:    Training device.

    Example::

        trainer = DPOTrainer(model, optimizer, DPOConfig(beta=0.1))
        for epoch in range(3):
            metrics = trainer.train_epoch(dpo_loader)
            print(metrics)
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       DPOConfig | None = None,
        device:    torch.device | str = "cpu",
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg or DPOConfig()
        self.device    = torch.device(device)

        # Freeze a copy of the model as the reference policy
        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        log.info(f"DPOTrainer: beta={self.cfg.beta}, loss={self.cfg.loss_type}")

    def _get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        mask:      torch.Tensor,
    ) -> torch.Tensor:
        """Get sequence log-probs from model for completion tokens."""
        # Forward pass: get logits
        logits, _ = model(input_ids)
        # Shift: logits[t] predicts input[t+1]
        shift_logits = logits[:, :-1, :]
        shift_ids    = input_ids[:, 1:]
        shift_mask   = mask[:, 1:]
        return compute_log_probs(shift_logits, shift_ids, shift_mask)

    def train_step(self, batch: dict) -> dict:
        """
        One DPO gradient step.

        Args:
            batch: Dict from DPODataset.collate_fn.

        Returns:
            Dict with loss, reward_accuracy, reward_margin.
        """
        chosen   = batch["chosen_ids"].to(self.device)
        rejected = batch["rejected_ids"].to(self.device)
        c_mask   = batch["chosen_mask"].to(self.device)
        r_mask   = batch["rejected_mask"].to(self.device)

        # Policy log-probs
        pi_c = self._get_log_probs(self.model, chosen,   c_mask)
        pi_r = self._get_log_probs(self.model, rejected, r_mask)

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_c = self._get_log_probs(self.ref_model, chosen,   c_mask)
            ref_r = self._get_log_probs(self.ref_model, rejected, r_mask)

        # Log ratios
        log_ratio_c = pi_c - ref_c
        log_ratio_r = pi_r - ref_r

        loss, info = dpo_loss(
            log_ratio_c, log_ratio_r,
            beta=self.cfg.beta,
            label_smoothing=self.cfg.label_smoothing,
            loss_type=self.cfg.loss_type,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return info

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one full DPO training epoch."""
        self.model.train()
        totals: dict = {}
        steps = 0
        for batch in loader:
            m = self.train_step(batch)
            for k, v in m.items():
                totals[k] = totals.get(k, 0.0) + v
            steps += 1
        return {k: v / max(steps, 1) for k, v in totals.items()} | {"steps": steps}
