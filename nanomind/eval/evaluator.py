"""
nanomind/eval/evaluator.py — High-level evaluation runner for NanoMind.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult
from nanomind.eval.metrics import token_accuracy, top_k_accuracy
from nanomind.utils.logger import get_logger


class Evaluator:
    """
    Runs evaluation over a DataLoader and collects metrics.

    Args:
        model:  The NanoMind model to evaluate.
        cfg:    :class:`~nanomind.eval.EvalConfig`.
        device: Target device (None = auto-detect from model).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: EvalConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg or EvalConfig()
        self.device = device or next(model.parameters()).device
        self.log    = get_logger("evaluator")

    @torch.no_grad()
    def evaluate_perplexity(self, loader: DataLoader) -> EvalResult:
        """
        Evaluate cross-entropy loss and perplexity over a DataLoader.

        Args:
            loader: DataLoader yielding ``(x, y)`` batches.

        Returns:
            :class:`~nanomind.eval.EvalResult` with loss, ppl, and bpc.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for i, (x, y) in enumerate(loader):
            if self.cfg.max_batches > 0 and i >= self.cfg.max_batches:
                break
            x, y    = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches  += 1

        mean_loss = total_loss / max(n_batches, 1)
        return EvalResult.from_loss(mean_loss, n_batches=n_batches)
