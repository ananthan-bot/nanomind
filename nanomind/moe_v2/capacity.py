"""
nanomind/moe_v2/capacity.py — Expert capacity and token overflow handling.

## The Capacity Problem

If one expert receives more tokens than its capacity allows,
some tokens must be "dropped" (passed through unchanged or zero-padded).

Capacity = capacity_factor × (total_tokens / n_experts)

Example:
  1024 tokens, 8 experts, capacity_factor=1.25
  capacity = 1.25 × (1024/8) = 160 tokens per expert

  If expert 3 gets 200 tokens → 40 tokens dropped (overflow).

Strategies:
  - Drop & replace with residual (Switch Transformer default)
  - Auxiliary loss to prevent overflow (load-balance)
  - Flexible capacity (scale factor per batch)
  - Expert choice (perfect balance, no overflow by design)

This module implements capacity buffers and overflow statistics.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class CapacityStats:
    """Statistics about token capacity utilisation."""
    n_tokens:       int
    n_experts:      int
    capacity:       int
    overflow_tokens: int
    overflow_frac:  float
    expert_loads:   list[int]

    def to_dict(self) -> dict:
        return {
            "n_tokens":       self.n_tokens,
            "capacity":       self.capacity,
            "overflow_tokens": self.overflow_tokens,
            "overflow_frac":  round(self.overflow_frac, 4),
            "max_load":       max(self.expert_loads),
            "min_load":       min(self.expert_loads),
        }


class CapacityBuffer:
    """
    Manages token capacity for each expert.

    Args:
        n_experts:       Number of experts.
        capacity_factor: Tokens per expert = cf × (N / n_experts).

    Example::

        buf   = CapacityBuffer(n_experts=8, capacity_factor=1.25)
        masks = buf.compute_masks(routing_indices, n_tokens=512)
        stats = buf.stats(routing_indices, n_tokens=512)
    """

    def __init__(
        self,
        n_experts:       int,
        capacity_factor: float = 1.25,
    ) -> None:
        self.n_experts       = n_experts
        self.capacity_factor = capacity_factor

    def capacity(self, n_tokens: int) -> int:
        """Compute expert capacity for a given token count."""
        return max(1, int(self.capacity_factor * n_tokens / self.n_experts))

    def compute_masks(
        self,
        indices: torch.Tensor,
        n_tokens: int,
    ) -> list[torch.Tensor]:
        """
        Compute per-expert boolean masks (True = token accepted).

        Tokens that overflow capacity are dropped.

        Args:
            indices:  ``(N, K)`` or ``(N, 1)`` expert assignment indices.
            n_tokens: Total token count.

        Returns:
            List of ``(N,)`` boolean masks, one per expert.
        """
        cap   = self.capacity(n_tokens)
        N     = indices.shape[0]
        masks = []
        for e in range(self.n_experts):
            # Find tokens assigned to expert e (top-1 only for simplicity)
            assigned = (indices[:, 0] == e).nonzero(as_tuple=True)[0]
            accepted = torch.zeros(N, dtype=torch.bool)
            if len(assigned) > 0:
                chosen   = assigned[:cap]   # keep only up to capacity
                accepted[chosen] = True
            masks.append(accepted)
        return masks

    def stats(
        self,
        indices:  torch.Tensor,
        n_tokens: int,
    ) -> CapacityStats:
        """Compute capacity utilisation statistics."""
        cap    = self.capacity(n_tokens)
        loads  = []
        overflow = 0
        for e in range(self.n_experts):
            n = (indices[:, 0] == e).sum().item()
            loads.append(n)
            overflow += max(0, n - cap)
        return CapacityStats(
            n_tokens        = n_tokens,
            n_experts       = self.n_experts,
            capacity        = cap,
            overflow_tokens = overflow,
            overflow_frac   = overflow / max(n_tokens, 1),
            expert_loads    = loads,
        )
