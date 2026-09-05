"""NanoMind AMP sub-package — Mixed Precision Training and Gradient Checkpointing.

Techniques for training larger models on the same GPU memory budget:

  1. Mixed Precision (AMP):    float16/bfloat16 activations → 2× memory saving
  2. Gradient Checkpointing:  discard and recompute activations → O(√N) memory
  3. Gradient Accumulation:   large effective batch without large GPU batch

Primary exports:
    - :class:`AMPTrainer`              — autocast + GradScaler + accum + clipping
    - :class:`AMPConfig`               — dtype, grad_scaler, accum_steps, checkpointing
    - :class:`NanoGradScaler`          — float16 loss scaling (scale/step/update)
    - :class:`GradAccumulator`         — micro-batch counter with should_step()
    - :class:`CheckpointedTransformerBlock` — block with activation checkpointing
    - :func:`mixed_precision_context`  — autocast context manager
    - :func:`checkpointed_forward`     — run any module with checkpointing
    - :func:`apply_gradient_checkpointing` — patch all blocks in-place
    - :func:`estimate_activation_memory`   — memory estimate with/without checkpointing
    - :func:`memory_tracker`           — context manager for memory profiling
    - :func:`model_parameter_memory_mb`    — parameter memory breakdown
"""

from nanomind.amp.config import AMPConfig
from nanomind.amp.context import mixed_precision_context, is_amp_available
from nanomind.amp.scaler import NanoGradScaler
from nanomind.amp.accumulation import GradAccumulator
from nanomind.amp.checkpoint import (
    checkpointed_forward,
    apply_gradient_checkpointing,
    estimate_activation_memory,
)
from nanomind.amp.trainer import AMPTrainer
from nanomind.amp.block import CheckpointedTransformerBlock
from nanomind.amp.memory import (
    get_memory_mb,
    get_peak_memory_mb,
    memory_tracker,
    model_parameter_memory_mb,
)

__all__ = [
    "AMPConfig",
    "mixed_precision_context",
    "is_amp_available",
    "NanoGradScaler",
    "GradAccumulator",
    "checkpointed_forward",
    "apply_gradient_checkpointing",
    "estimate_activation_memory",
    "AMPTrainer",
    "CheckpointedTransformerBlock",
    "get_memory_mb",
    "get_peak_memory_mb",
    "memory_tracker",
    "model_parameter_memory_mb",
]
