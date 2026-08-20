"""
nanomind/eval/config.py — Evaluation configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Configuration for :class:`~nanomind.eval.Evaluator`.

    Attributes:
        max_batches:    Maximum number of batches to evaluate (0 = all).
        compute_acc:    Whether to compute token accuracy.
        compute_top_k:  Whether to compute top-K accuracy.
        top_k:          K value for top-K accuracy.
        compute_bpc:    Whether to compute bits-per-character.
        device:         Device for evaluation (``"auto"`` = use model device).
    """

    max_batches:   int   = 0
    compute_acc:   bool  = True
    compute_top_k: bool  = True
    top_k:         int   = 5
    compute_bpc:   bool  = True
    device:        str   = "auto"

    def __post_init__(self) -> None:
        assert self.max_batches >= 0
        assert self.top_k >= 1
