"""
nanomind/merge/dare.py — DARE: Drop And REscale parameter sparsification (Yu et al., ICML 2024).
Randomly drops redundant fine-tuned parameters and rescales remaining weights to preserve expectation.
"""
from typing import List, Optional
import torch


class DAREPruner:
    """
    Drop And REscale:
    With probability drop_rate, parameter delta is set to 0.
    Remaining parameters are rescaled by 1 / (1 - drop_rate) such that E[tilde_tau] = tau.
    """

    def __init__(self, drop_rate: float = 0.5, rescale: bool = True):
        self.drop_rate = drop_rate
        self.rescale = rescale

    def prune_tensor(self, delta: torch.Tensor) -> torch.Tensor:
        """Applies DARE Bernoulli drop and rescaling."""
        if self.drop_rate <= 0.0:
            return delta
        if self.drop_rate >= 1.0:
            return torch.zeros_like(delta)

        keep_prob = 1.0 - self.drop_rate
        # Bernoulli mask where 1 = keep, 0 = drop
        mask = torch.bernoulli(torch.full_like(delta, keep_prob))
        pruned = delta * mask

        if self.rescale:
            pruned = pruned / keep_prob

        return pruned
