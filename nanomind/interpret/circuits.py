"""
nanomind/interpret/circuits.py — Circuit analysis via attention head ablation.

## Mechanistic Interpretability

Circuits (Elhage et al., 2021) are minimal subgraphs of a neural network
that implement a specific behaviour. Finding circuits involves:
  1. Identify which attention heads are important for a task
  2. Ablate (zero-out) heads and measure performance degradation
  3. Build a causal graph of head interactions

Head ablation types:
  Zero ablation:  set head output to zero
  Mean ablation:  replace head output with its mean across the dataset
  Activation patching: replace head activations from a "clean" run

This module implements zero-ablation for head importance scoring.

References:
  Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
  https://transformer-circuits.pub/2021/framework/index.html

  Wang et al. (2022) "Interpretability in the Wild: IOI circuit"
  https://arxiv.org/abs/2211.00593
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result of ablating one attention head."""
    layer:       int
    head:        int
    loss_delta:  float    # change in loss (positive = head was important)
    base_loss:   float
    ablated_loss: float

    @property
    def importance(self) -> float:
        """Importance score (higher = more important)."""
        return max(0.0, self.loss_delta)

    def to_dict(self) -> dict:
        return {
            "layer":        self.layer,
            "head":         self.head,
            "loss_delta":   round(self.loss_delta, 4),
            "importance":   round(self.importance, 4),
        }


class HeadAblator:
    """
    Ablate individual attention heads and measure impact.

    Finds which heads are critical for a given input by zeroing each
    head's output and measuring the resulting loss increase.

    Args:
        model:  Language model with MultiheadAttention layers.

    Example::

        ablator = HeadAblator(model)
        results = ablator.ablate_all(input_ids, target_ids)
        # Sort by importance to find the most critical heads
        results.sort(key=lambda r: r.importance, reverse=True)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _mha_layers(self) -> list[nn.MultiheadAttention]:
        """Collect all MultiheadAttention layers."""
        return [m for m in self.model.modules()
                if isinstance(m, nn.MultiheadAttention)]

    def _base_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            logits, loss = self.model(input_ids, target_ids)
            if loss is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )
        return loss.item()

    def ablate_head(
        self,
        layer:      nn.MultiheadAttention,
        head_idx:   int,
        input_ids:  torch.Tensor,
        target_ids: torch.Tensor,
        base_loss:  float,
        layer_idx:  int = 0,
    ) -> AblationResult:
        """
        Zero-ablate a single head and measure loss delta.

        Args:
            layer:      The MultiheadAttention module.
            head_idx:   Head index to ablate.
            input_ids:  Input token IDs.
            target_ids: Target token IDs.
            base_loss:  Reference loss without ablation.
            layer_idx:  Layer index (for logging).
        """
        n_heads = layer.num_heads
        d_head  = layer.head_dim

        # Hook to zero one head's output
        handle  = None
        def ablate_hook(module, inp, out):
            if isinstance(out, tuple):
                # out = (output, weights)
                output = out[0].clone()
                # Zero head h: output is (B, T, D), head occupies columns
                start  = head_idx * d_head
                end    = start + d_head
                output[:, :, start:end] = 0.0
                return (output,) + out[1:]
            return out

        handle = layer.register_forward_hook(ablate_hook)
        try:
            with torch.no_grad():
                logits, loss = self.model(input_ids, target_ids)
                if loss is None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), target_ids.view(-1)
                    )
            abl_loss = loss.item()
        finally:
            handle.remove()

        return AblationResult(
            layer        = layer_idx,
            head         = head_idx,
            loss_delta   = abl_loss - base_loss,
            base_loss    = base_loss,
            ablated_loss = abl_loss,
        )

    def ablate_all(
        self,
        input_ids:  torch.Tensor,
        target_ids: torch.Tensor,
    ) -> list[AblationResult]:
        """
        Ablate every head in every layer.

        Returns:
            List of :class:`AblationResult` sorted by layer then head.
        """
        base_loss = self._base_loss(input_ids, target_ids)
        results   = []
        for l_idx, layer in enumerate(self._mha_layers()):
            for h_idx in range(layer.num_heads):
                result = self.ablate_head(
                    layer, h_idx, input_ids, target_ids, base_loss, l_idx
                )
                results.append(result)
        return results
