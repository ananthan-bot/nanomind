"""
nanomind/continual/packnet.py — PackNet: prune and pack parameters for each task.

PackNet (Mallya & Lazebnik, 2018) allocates disjoint subnetworks:
  Task 1: train full network → prune P% of weights → "pack" mask_1
  Task 2: train remaining (1-P)% → prune again → pack mask_2
  Task k: remaining (1-P)^k of weights

At inference, the task ID selects which binary mask to activate.
Pruned weights can be reused for future tasks (weight recycling).

This achieves zero forgetting by hard parameter isolation.
Limitation: capacity decreases with each task.

Reference:
  Mallya & Lazebnik (2018) "PackNet: Adding Multiple Tasks to a Single Network
  by Iterative Pruning" https://arxiv.org/abs/1711.05769
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nanomind.utils.logger import get_logger

log = get_logger("continual.packnet")


class PackNet:
    """
    PackNet: iterative pruning for parameter isolation across tasks.

    Args:
        model:       Language model.
        prune_ratio: Fraction of free weights to prune per task.

    Example::

        pn = PackNet(model, prune_ratio=0.5)
        # After training task 0:
        pn.prune_and_pack(task_id=0)
        # Free weights are now available for task 1.
        # At inference:
        pn.apply_mask(task_id=0)   # restore task 0 weights
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5) -> None:
        self.model       = model
        self.prune_ratio = prune_ratio
        # Binary masks per task: 1 = owned by task, 0 = free
        self._task_masks:  list[dict[str, torch.Tensor]] = []
        # Accumulated frozen mask (union of all past tasks)
        self._frozen_mask: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.data, dtype=torch.bool)
            for n, p in model.named_parameters() if p.requires_grad
        }
        # Saved weights per task
        self._task_weights: list[dict[str, torch.Tensor]] = []

    def _free_magnitude(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Magnitude of free (non-frozen) weights for a parameter."""
        frozen = self._frozen_mask.get(name, torch.zeros_like(param, dtype=torch.bool))
        mag    = param.data.abs().clone()
        mag[frozen] = float("inf")   # exclude frozen from pruning
        return mag

    def prune_and_pack(self, task_id: int) -> dict:
        """
        Prune the model and assign a mask to this task.

        Args:
            task_id: Current task ID.

        Returns:
            Dict with pruning statistics.
        """
        task_mask = {}
        n_pruned  = 0
        n_total   = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            frozen  = self._frozen_mask[name]
            free    = ~frozen
            n_free  = free.sum().item()
            n_prune = int(n_free * self.prune_ratio)

            # Prune smallest-magnitude free weights
            mag     = param.data.abs().clone()
            mag[frozen] = float("inf")
            flat_mag = mag.view(-1)
            if n_prune > 0:
                threshold = flat_mag.kthvalue(n_prune).values.item()
                new_mask  = (mag <= threshold) & free
            else:
                new_mask  = torch.zeros_like(free)

            task_mask[name]          = new_mask.clone()
            self._frozen_mask[name] |= new_mask
            n_pruned += new_mask.sum().item()
            n_total  += param.numel()

        self._task_masks.append(task_mask)
        self._task_weights.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()
             if p.requires_grad}
        )

        log.info(f"PackNet task {task_id}: pruned {n_pruned}/{n_total} "
                 f"({100*n_pruned/max(n_total,1):.1f}%)")
        return {"n_pruned": n_pruned, "n_total": n_total,
                "prune_pct": n_pruned / max(n_total, 1)}

    def apply_mask(self, task_id: int) -> None:
        """
        Restore model weights for a specific task.

        Args:
            task_id: Task to restore.
        """
        if task_id >= len(self._task_weights):
            raise ValueError(f"Task {task_id} not registered")
        weights = self._task_weights[task_id]
        for n, p in self.model.named_parameters():
            if n in weights:
                p.data.copy_(weights[n])

    def freeze_past_weights(self) -> None:
        """Zero gradients for all frozen (past task) weights during training."""
        for n, p in self.model.named_parameters():
            if p.grad is not None and n in self._frozen_mask:
                p.grad.data[self._frozen_mask[n]] = 0.0

    @property
    def n_tasks(self) -> int:
        return len(self._task_masks)

    @property
    def free_ratio(self) -> float:
        """Fraction of weights still available for new tasks."""
        total  = sum(v.numel() for v in self._frozen_mask.values())
        frozen = sum(v.sum().item() for v in self._frozen_mask.values())
        return 1.0 - frozen / max(total, 1)
