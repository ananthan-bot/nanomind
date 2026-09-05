"""
nanomind/amp/block.py — Transformer block with activation checkpointing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nanomind.blocks.block import TransformerBlock


class CheckpointedTransformerBlock(TransformerBlock):
    """
    TransformerBlock with activation checkpointing applied to the full block.

    On forward: runs the block normally, discards activations.
    On backward: re-runs the block forward to recover activations.

    Memory: O(1) per block (only I/O tensors kept), vs O(T) for standard.
    Compute: ~33% more (one extra forward per backward).

    Args:
        Same as :class:`TransformerBlock`.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        """Checkpointed forward — recomputes activations during backward."""
        def _forward(x_):
            return super(CheckpointedTransformerBlock, self).forward(x_, **kwargs)
        return checkpoint(_forward, x, use_reentrant=False)
