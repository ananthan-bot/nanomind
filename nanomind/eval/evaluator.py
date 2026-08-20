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
