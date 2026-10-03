"""
nanomind/distributed/checkpointing.py — Gradient checkpointing & activation recomputation.

## The Memory-Compute Tradeoff

During backward pass, PyTorch needs activation tensors from the forward pass.
For a Transformer with L layers:
  Without checkpointing: store ALL activations → O(L × B × T × D) memory
  With checkpointing:    store ONLY checkpointed activations → O(√L) memory

## Gradient Checkpointing (Chen et al., 2016)

Instead of storing ALL activations:
  1. Forward: compute normally, but don't save intermediate activations
  2. Save only "checkpoint" tensors at intervals
  3. Backward: recompute activations from last checkpoint when needed

Trade: 33% more compute, but O(√L) memory instead of O(L).

This allows training much larger batches / longer sequences!

## Selective Activation Recomputation

Re-checkpoint only expensive operations:
  - Full self-attention: O(T² × D)
  - Recompute: only store Q,K,V, recompute attention scores

## Activation Offloading

For extreme memory savings: offload activations to CPU RAM during forward.
Bring back to GPU during backward.
  → 4× larger batch sizes!
  → ~20% throughput loss due to CPU↔GPU transfer

Reference:
  Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost"
  https://arxiv.org/abs/1604.06174
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CheckpointingConfig:
    """Configuration for gradient checkpointing."""
    enabled:            bool  = True
    checkpoint_ratio:   float = 1.0    # 1.0 = all layers, 0.5 = every other layer
    offload_to_cpu:     bool  = False  # offload activations to CPU
    recompute_attention: bool = True   # recompute attention in backward
    use_reentrant:      bool  = False  # PyTorch checkpoint() flag

    def __post_init__(self):
        assert 0.0 <= self.checkpoint_ratio <= 1.0


class CheckpointedLayer(nn.Module):
    """
    Wrapper that applies gradient checkpointing to a layer.

    Instead of storing activations, recomputes them during backward.
    Reduces memory by not storing intermediate activations.

    Args:
        layer:      The wrapped layer.
        enabled:    Whether checkpointing is active.

    Example::

        layer     = CheckpointedLayer(transformer_block, enabled=True)
        output    = layer(hidden_states)   # activations NOT stored
        # → 33% more compute, but much less memory
    """

    def __init__(self, layer: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.layer   = layer
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and x.requires_grad:
            # torch.utils.checkpoint.checkpoint recomputes in backward
            return torch.utils.checkpoint.checkpoint(
                self.layer, x, use_reentrant=False
            )
        return self.layer(x)


class SelectiveCheckpointing:
    """
    Apply gradient checkpointing to a subset of layers.

    Checkpoints every ceil(1/ratio) layers.

    Args:
        layers:  List of transformer layers.
        cfg:     :class:`CheckpointingConfig`.

    Example::

        sc     = SelectiveCheckpointing(transformer_layers, cfg)
        layers = sc.apply()
        # Some layers now wrapped with CheckpointedLayer
    """

    def __init__(
        self,
        layers: list[nn.Module],
        cfg:    CheckpointingConfig | None = None,
    ) -> None:
        self.layers = layers
        self.cfg    = cfg or CheckpointingConfig()

    def apply(self) -> list[nn.Module]:
        """Wrap layers with checkpointing according to ratio."""
        if not self.cfg.enabled:
            return self.layers

        step      = max(1, round(1.0 / self.cfg.checkpoint_ratio))
        wrapped   = []
        for i, layer in enumerate(self.layers):
            if i % step == 0:
                wrapped.append(CheckpointedLayer(layer, enabled=True))
            else:
                wrapped.append(layer)
        return wrapped

    @property
    def n_checkpointed(self) -> int:
        step = max(1, round(1.0 / self.cfg.checkpoint_ratio))
        return (len(self.layers) + step - 1) // step

    def memory_savings(self, baseline_mb: float) -> dict:
        """Estimate memory savings from checkpointing."""
        ratio = self.cfg.checkpoint_ratio
        saved_fraction = 1.0 - ratio   # store ratio, free the rest
        return {
            "baseline_mb":    round(baseline_mb, 2),
            "with_ckpt_mb":   round(baseline_mb * ratio, 2),
            "saved_mb":       round(baseline_mb * saved_fraction, 2),
            "saved_fraction":  round(saved_fraction, 3),
            "compute_overhead": "33%",
        }


def estimate_activation_memory(
    n_layers:   int,
    batch_size: int,
    seq_len:    int,
    d_model:    int,
    n_heads:    int,
    dtype_bytes: int = 2,   # fp16
) -> dict:
    """
    Estimate transformer activation memory (bytes).

    Args:
        n_layers:    Number of transformer layers.
        batch_size:  Batch size.
        seq_len:     Sequence length.
        d_model:     Model hidden dimension.
        n_heads:     Number of attention heads.
        dtype_bytes: Bytes per element (2=fp16, 4=fp32).

    Returns:
        Dict with memory estimates with/without checkpointing.
    """
    B, T, D, H = batch_size, seq_len, d_model, n_heads

    # Per-layer activations:
    # - Input: B × T × D
    # - Attention QKV: B × T × 3D
    # - Attention scores: B × H × T × T
    # - MLP: B × T × 4D
    per_layer = dtype_bytes * (
        B * T * D +           # input
        B * T * 3 * D +       # QKV
        B * H * T * T +       # attention scores
        B * T * 4 * D         # MLP
    )

    total        = n_layers * per_layer
    with_ckpt    = int(total ** 0.5)  # √L memory with full checkpointing

    return {
        "per_layer_mb":   round(per_layer / 1e6, 2),
        "total_mb":       round(total / 1e6, 2),
        "with_ckpt_mb":   round(with_ckpt / 1e6, 2),
        "savings_x":      round(total / max(with_ckpt, 1), 1),
    }
