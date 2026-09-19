"""
nanomind/continual/ewc.py — Elastic Weight Consolidation (EWC).

EWC treats continual learning as Bayesian inference:
  P(θ | D_A, D_B) ∝ P(D_B | θ) × P(θ | D_A)

P(θ | D_A) is approximated as a diagonal Gaussian:
  P(θ | D_A) ≈ N(θ_A*, F_A)

where F_A is the diagonal Fisher Information Matrix — estimated as:
  F_i = E[(∂ log P(y|x,θ) / ∂θ_i)²]

The EWC loss for task B is:
  L(θ) = L_B(θ) + (λ/2) × Σ_i F_i × (θ_i - θ_A*_i)²

High F_i → parameter i was important for task A → constrain it.
Low F_i  → parameter i was unimportant   → allow free learning.

Reference:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
  https://arxiv.org/abs/1612.00796
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("continual.ewc")


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    After each task, call :meth:`register_task` to compute the Fisher
    Information Matrix and store the optimal weights.
    During training on the next task, call :meth:`penalty` to get the
    EWC regularisation term to add to the loss.

    Args:
        model:      Language model.
        lambda_:    Regularisation strength (higher = less forgetting).
        n_samples:  Samples for Fisher estimation.

    Example::

        ewc = EWC(model, lambda_=1000.0)
        # After training task A:
        ewc.register_task(train_loader_A, task_id=0)
        # During training task B:
        loss = task_loss + ewc.penalty()
    """

    def __init__(
        self,
        model:    nn.Module,
        lambda_:  float = 1000.0,
        n_samples: int  = 200,
    ) -> None:
        self.model     = model
        self.lambda_   = lambda_
        self.n_samples = n_samples
        # Store per-task: optimal weights + Fisher diagonals
        self._means:   list[dict] = []   # θ*_A per task
        self._fishers: list[dict] = []   # F_A per task

    def _compute_fisher(self, batches: list) -> dict:
        """
        Compute diagonal Fisher Information Matrix.

        Args:
            batches: List of (x, y) batches.

        Returns:
            Dict mapping param name → Fisher diagonal tensor.
        """
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        n_done = 0

        for x, y in batches:
            if n_done >= self.n_samples:
                break
            self.model.zero_grad()
            logits, loss = self.model(x, y)
            if loss is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            n_done += x.shape[0]

        # Normalise
        for n in fisher:
            fisher[n] /= max(n_done, 1)
        return fisher

    def register_task(self, batches: list, task_id: int = None) -> None:
        """
        Record task completion: compute Fisher and save optimal weights.

        Args:
            batches: List of ``(x, y)`` from the completed task.
            task_id: Optional task identifier (for logging).
        """
        task_id = task_id if task_id is not None else len(self._means)
        log.info(f"EWC: registering task {task_id} with {self.n_samples} samples")

        fisher = self._compute_fisher(batches)
        means  = {n: p.data.clone()
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self._fishers.append(fisher)
        self._means.append(means)

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty for all registered tasks.

        Returns:
            Scalar EWC loss term (add to main loss before backward).
        """
        if not self._fishers:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for fisher, means in zip(self._fishers, self._means):
            for n, p in self.model.named_parameters():
                if n in fisher and p.requires_grad:
                    loss += (fisher[n] * (p - means[n]).pow(2)).sum()

        return (self.lambda_ / 2) * loss

    @property
    def n_tasks_registered(self) -> int:
        return len(self._means)

    def fisher_summary(self, task_id: int = 0) -> dict:
        """Summary statistics of Fisher matrix for a task."""
        if task_id >= len(self._fishers):
            return {}
        f     = self._fishers[task_id]
        total = sum(v.sum().item() for v in f.values())
        nparams = sum(v.numel() for v in f.values())
        return {
            "task_id":      task_id,
            "total_fisher": round(total, 4),
            "n_params":     nparams,
            "mean_fisher":  round(total / max(nparams, 1), 8),
        }
