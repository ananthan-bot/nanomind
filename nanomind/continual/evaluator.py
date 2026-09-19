"""
nanomind/continual/evaluator.py — Continual learning metrics.

Key metrics for continual learning:

  Average Accuracy (AA):
    AA = (1/T) × Σ_t a_t,T
    where a_t,T = accuracy on task t after training on all T tasks

  Backward Transfer (BWT):
    BWT = (1/(T-1)) × Σ_t (a_t,T - a_t,t)
    Negative BWT = forgetting  (accuracy dropped after learning new tasks)
    Positive BWT = backward transfer (new tasks helped old ones)

  Forward Transfer (FWT):
    FWT = (1/(T-1)) × Σ_t (a_t,t - b_t)
    where b_t = random baseline accuracy on task t before training
    Positive FWT = new tasks help future tasks via transfer

  Forgetting (F):
    F = (1/(T-1)) × Σ_t max_{t'≤T} a_t,t' - a_t,T
    How much accuracy was lost from the peak.

Reference:
  Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"
  https://arxiv.org/abs/1706.08840
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ContinualMetrics:
    """Continual learning evaluation metrics."""
    accuracy_matrix: list[list[float]]   # a[t][s] = acc on task t after task s
    n_tasks:         int

    @property
    def average_accuracy(self) -> float:
        """AA after all tasks: mean accuracy on all past tasks."""
        n = self.n_tasks
        if n == 0:
            return 0.0
        return sum(self.accuracy_matrix[t][n - 1] for t in range(n)) / n

    @property
    def backward_transfer(self) -> float:
        """BWT: negative = forgetting, positive = improvement."""
        n = self.n_tasks
        if n <= 1:
            return 0.0
        bwt = sum(
            self.accuracy_matrix[t][n - 1] - self.accuracy_matrix[t][t]
            for t in range(n - 1)
        )
        return bwt / max(n - 1, 1)

    @property
    def forgetting(self) -> float:
        """Average forgetting across tasks."""
        n = self.n_tasks
        if n <= 1:
            return 0.0
        total = 0.0
        for t in range(n - 1):
            row   = self.accuracy_matrix[t][:n]
            peak  = max(row[:t + 1]) if row else 0.0
            final = row[n - 1]
            total += peak - final
        return total / max(n - 1, 1)

    def to_dict(self) -> dict:
        return {
            "average_accuracy":  round(self.average_accuracy, 4),
            "backward_transfer": round(self.backward_transfer, 4),
            "forgetting":        round(self.forgetting, 4),
            "n_tasks":           self.n_tasks,
        }


class ContinualEvaluator:
    """
    Evaluate continual learning performance across tasks.

    Args:
        model:       The language model.

    Example::

        eval_ = ContinualEvaluator(model)
        eval_.record(task_id=0, after_task=0, accuracy=0.92)
        eval_.record(task_id=0, after_task=1, accuracy=0.74)  # forgetting!
        metrics = eval_.compute()
        print(metrics.backward_transfer)  # negative = forgot
    """

    def __init__(self, model: nn.Module, n_tasks: int) -> None:
        self.model   = model
        self.n_tasks = n_tasks
        # acc_matrix[task_id][after_task] = accuracy
        self._acc: list[list[float | None]] = [
            [None] * n_tasks for _ in range(n_tasks)
        ]

    def record(self, task_id: int, after_task: int, accuracy: float) -> None:
        """Record accuracy on task_id evaluated after training on after_task."""
        self._acc[task_id][after_task] = accuracy

    @torch.no_grad()
    def evaluate_task(
        self,
        task_id:  int,
        batches:  list,
        after_task: int,
    ) -> float:
        """
        Evaluate model accuracy on a task.

        Args:
            task_id:    Task to evaluate.
            batches:    List of ``(x, y)`` batches.
            after_task: Current training step (for logging).

        Returns:
            Accuracy in [0, 1].
        """
        self.model.eval()
        n_correct = 0
        n_total   = 0
        for x, y in batches:
            logits, _ = self.model(x, y)
            preds     = logits.argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total   += y.numel()
        acc = n_correct / max(n_total, 1)
        self.record(task_id, after_task, acc)
        return acc

    def compute(self) -> ContinualMetrics:
        """Compute final continual learning metrics."""
        filled = [
            [self._acc[t][s] or 0.0 for s in range(self.n_tasks)]
            for t in range(self.n_tasks)
        ]
        return ContinualMetrics(accuracy_matrix=filled, n_tasks=self.n_tasks)
