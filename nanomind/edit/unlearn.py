"""
nanomind/edit/unlearn.py — Targeted knowledge erasure / factual unlearning.
Suppresses specific memorized associations by projecting target probabilities towards uniform entropy.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from nanomind.edit.rome import ROMEEngine
from nanomind.edit.config import EditConfig


class FactualUnlearner:
    """
    Unlearns private or copyrighted factual associations by redirecting the value vector
    towards a neutral or randomized representation.
    """

    def __init__(self, config: Optional[EditConfig] = None):
        self.rome = ROMEEngine(config)

    def unlearn_fact(
        self,
        layer: nn.Linear,
        key: torch.Tensor,
        c_inv: torch.Tensor,
    ):
        """Erases association by zeroing or neutralizing output projection for key k*."""
        zero_value = torch.zeros(layer.weight.shape[0], device=layer.weight.device)
        self.rome.apply_edit(layer, key, zero_value, c_inv)
