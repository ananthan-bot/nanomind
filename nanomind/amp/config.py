"""
nanomind/amp/config.py — Mixed Precision and Checkpointing configuration.

## Mixed Precision Training (AMP)
Standard training uses float32 for all tensors. Mixed Precision Training uses
float16 or bfloat16 for forward/backward passes and float32 for the master weights:

  float32 activations  → 4 bytes/element
  float16 activations  → 2 bytes/element  (2× memory saving)

On modern NVIDIA GPUs (Volta+), float16 matrix multiplications are computed on
Tensor Cores at 2-8× higher throughput than float32.

Risk: float16 has small dynamic range → gradient underflow / overflow.
Fix:  GradScaler scales the loss before backward, unscales gradients before
      the optimizer step, and skips the step if inf/nan gradients are detected.

## Gradient Checkpointing (Activation Checkpointing)
During the forward pass, PyTorch caches all intermediate activations for
use in the backward pass. This uses O(N_layers) memory.

Gradient checkpointing trades compute for memory: only checkpoint boundaries
are saved; activations between checkpoints are RECOMPUTED during backward.

  Memory: O(√N_layers)  — reduced by discarding and recomputing
  Compute: ~33% extra   — one extra forward pass per checkpointed segment

## Gradient Accumulation
Simulate large batch sizes without increasing memory by accumulating gradients
over multiple micro-batches before calling optimizer.step():

  Effective batch = batch_size × grad_accum_steps
  Memory per step = unchanged (only one micro-batch at a time)

References:
  AMP:                   Micikevicius et al. (2017) https://arxiv.org/abs/1710.03740
  Gradient checkpointing: Chen et al. (2016) https://arxiv.org/abs/1604.06174
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class AMPConfig:
    """
    Configuration for Mixed Precision Training and Gradient Checkpointing.

    Attributes:
        enabled:          Master switch — enable AMP.
        dtype:            Compute dtype (``"float16"`` or ``"bfloat16"``).
        grad_scaler:      Use GradScaler for loss scaling (float16 only).
        init_scale:       Initial GradScaler loss scale.
        growth_interval:  Steps between scale increases.
        grad_accum_steps: Gradient accumulation micro-batches.
        checkpoint_layers:Use gradient checkpointing on transformer blocks.
        clip_grad_norm:   Max gradient norm (0.0 = off).
    """

    enabled:           bool  = True
    dtype:             str   = "bfloat16"
    grad_scaler:       bool  = True
    init_scale:        float = 65536.0
    growth_interval:   int   = 2000
    grad_accum_steps:  int   = 1
    checkpoint_layers: bool  = False
    clip_grad_norm:    float = 1.0

    def __post_init__(self) -> None:
        assert self.dtype in ("float16", "bfloat16", "float32")
        assert self.grad_accum_steps >= 1
        assert self.clip_grad_norm  >= 0.0
        assert self.init_scale      > 0.0
        # GradScaler is only needed for float16 (bfloat16 has wider range)
        if self.dtype == "bfloat16":
            self.grad_scaler = False

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.dtype]
