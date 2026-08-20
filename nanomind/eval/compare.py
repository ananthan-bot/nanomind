"""
nanomind/eval/compare.py — Side-by-side model comparison utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.evaluator import Evaluator
from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult


def compare_models(
    models: dict[str, nn.Module],
    loader: DataLoader,
    cfg: EvalConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, EvalResult]:
    """
    Evaluate multiple models on the same DataLoader and return results.

    Args:
        models: Dict mapping model name -> model.
        loader: DataLoader for evaluation.
        cfg:    EvalConfig (shared across all models).
        device: Evaluation device.

    Returns:
        Dict mapping model name -> :class:`EvalResult`.
    """
    results: dict[str, EvalResult] = {}
    for name, model in models.items():
        evaluator = Evaluator(model, cfg or EvalConfig(), device)
        results[name] = evaluator.full_eval(loader)
    return results


def print_comparison(results: dict[str, "EvalResult"]) -> None:
    """Pretty-print a model comparison table."""
    header = f"{'Model':<20} {'Loss':>8} {'PPL':>8} {'BPC':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        acc_str = f"{r.accuracy:.4f}" if r.accuracy == r.accuracy else "  N/A"
        print(
            f"{name:<20} {r.loss:>8.4f} {r.ppl:>8.2f} "
            f"{r.bpc:>8.4f} {acc_str:>8}"
        )
