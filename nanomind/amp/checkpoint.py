"""
nanomind/amp/checkpoint.py — Gradient checkpointing for transformer layers.

Gradient checkpointing (Chen et al. 2016) trades activation memory for compute:

  Normal training:      store all activations during forward → O(N) memory
  Checkpointing:        discard activations at checkpoint boundaries
                        recompute them during backward → O(√N) or O(1) memory
                        cost: ~33% additional compute

Usage in transformer training:
  - Segment the model into checkpointed segments (e.g., every 2 layers)
  - PyTorch re-runs the segment forward during backward
  - Result: can train ~2-4× larger models with same GPU memory

API:
  torch.utils.checkpoint.checkpoint(function, *inputs)
  → runs function(*inputs) normally in forward
  → during backward, re-runs function(*inputs) to get activations
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def checkpointed_forward(
    module: nn.Module,
    x:      torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Run a module forward with gradient checkpointing.

    Wraps ``torch.utils.checkpoint.checkpoint`` to handle arbitrary module
    kwargs by using a closure.

    Args:
        module: Module to run with checkpointing.
        x:      Input tensor.
        **kwargs: Additional keyword arguments forwarded to module.forward().

    Returns:
        Module output tensor.
    """
    def _forward(x_):
        return module(x_, **kwargs)

    return checkpoint(_forward, x, use_reentrant=False)


def apply_gradient_checkpointing(
    model:           nn.Module,
    block_class:     type,
    every_n_layers:  int = 1,
) -> int:
    """
    Wrap every N-th transformer block with gradient checkpointing.

    Replaces the forward method of matching blocks with a checkpointed version.
    Modifies the model in-place.

    Args:
        model:          Transformer model.
        block_class:    Block class to target (e.g., TransformerBlock).
        every_n_layers: Checkpoint every N matching blocks (1 = all blocks).

    Returns:
        Number of blocks that were checkpointed.
    """
    count = 0
    for i, module in enumerate(model.modules()):
        if isinstance(module, block_class) and i % every_n_layers == 0:
            original_forward = module.forward

            def make_checkpointed_forward(orig):
                def new_forward(x, *args, **kwargs):
                    def fn(x_):
                        return orig(x_, *args, **kwargs)
                    return checkpoint(fn, x, use_reentrant=False)
                return new_forward

            module.forward = make_checkpointed_forward(original_forward)
            count += 1

    return count


def estimate_activation_memory(
    batch:    int,
    seq_len:  int,
    d_model:  int,
    n_layers: int,
    dtype_bytes: int = 4,
) -> dict:
    """
    Estimate activation memory with and without gradient checkpointing.

    Args:
        batch, seq_len, d_model, n_layers: Model dimensions.
        dtype_bytes: Bytes per element.

    Returns:
        Dict with ``standard_mb``, ``checkpointed_mb``, ``savings_ratio``.
    """
    # Very rough estimate: main activations per layer ≈ 12 × batch × seq × d_model
    per_layer = 12 * batch * seq_len * d_model * dtype_bytes
    std_mb    = per_layer * n_layers / (1024 ** 2)
    # With checkpointing: store ~O(√n_layers) checkpoints
    import math
    ckpt_mb   = per_layer * math.ceil(math.sqrt(n_layers)) / (1024 ** 2)
    return {
        "standard_mb":    std_mb,
        "checkpointed_mb": ckpt_mb,
        "savings_ratio":  std_mb / max(ckpt_mb, 1e-9),
    }
