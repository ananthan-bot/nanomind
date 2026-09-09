"""
nanomind/eval/accuracy.py — Token prediction accuracy metrics.

Top-K accuracy: the fraction of time the correct next token appears
in the model's top-K predictions. Standard metrics in LLM evaluation.

  Top-1 accuracy = argmax match (greedy accuracy)
  Top-5 accuracy = correct token in the 5 most probable predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def top_k_accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    k:       int = 1,
) -> float:
    """
    Compute top-K token prediction accuracy.

    Args:
        logits:  ``(N, V)`` model output logits.
        targets: ``(N,)`` ground-truth token IDs.
        k:       Number of top predictions to consider.

    Returns:
        Accuracy in [0, 1].
    """
    topk = logits.topk(k, dim=-1).indices   # (N, k)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def multi_k_accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    k_values: list[int] = (1, 5, 10),
) -> dict:
    """
    Compute top-K accuracy for multiple values of K.

    Args:
        logits:   ``(N, V)`` model output logits.
        targets:  ``(N,)`` ground-truth token IDs.
        k_values: List of K values to compute.

    Returns:
        Dict mapping ``top_{k}`` → accuracy float.
    """
    return {f"top_{k}": top_k_accuracy(logits, targets, k) for k in k_values}


@torch.no_grad()
def evaluate_accuracy(
    model:       nn.Module,
    loader:      DataLoader,
    device:      str | torch.device = "cpu",
    k_values:    list[int] = (1, 5),
    max_batches: int | None = None,
) -> dict:
    """
    Evaluate top-K accuracy of a model over a full dataset.

    Args:
        model:       Language model returning ``(logits, loss)``.
        loader:      DataLoader of ``(x, y)`` batches.
        device:      Evaluation device.
        k_values:    List of K values.
        max_batches: Optionally limit evaluation batches.

    Returns:
        Dict with ``top_1``, ``top_5``, etc., plus ``n_tokens``.
    """
    model.eval()
    device     = torch.device(device)
    totals     = {f"top_{k}": 0.0 for k in k_values}
    n_batches  = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y    = x.to(device), y.to(device)
        logits, _ = model(x)
        # Flatten: each position predicts the next token
        N, T, V = logits.shape
        flat_logits  = logits[:, :-1].reshape(-1, V)
        flat_targets = y[:, 1:].reshape(-1)

        for k in k_values:
            totals[f"top_{k}"] += top_k_accuracy(flat_logits, flat_targets, k)
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in totals.items()} | {"n_batches": n_batches}
