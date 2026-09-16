"""
nanomind/federated/compression.py — Gradient compression for communication efficiency.

In federated learning, transmitting full gradients is expensive:
  - GPT-3 (175B params) × 4 bytes = 700 GB per round per client!

Gradient compression reduces bandwidth:
  Top-K sparsification:  keep only the K largest gradient components
    → 100× compression at 1% sparsity with minimal accuracy loss
  Random sparsification: keep K random components (unbiased)
  Quantisation:          reduce float32 to int8/int4

Error feedback: accumulate the dropped gradients in a local buffer
  and add them to the next round's gradients to ensure convergence.

References:
  Lin et al. (2017) "Deep Gradient Compression" https://arxiv.org/abs/1712.01887
  Stich et al. (2018) "Sparsified SGD with Memory" https://arxiv.org/abs/1809.07599
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterator


def _flat_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate all gradients into a single flat tensor."""
    return torch.cat([
        p.grad.data.view(-1) for p in model.parameters()
        if p.grad is not None
    ])


def _set_flat_grads(model: nn.Module, flat: torch.Tensor) -> None:
    """Restore a flat gradient tensor back to model.grad."""
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.numel()
            p.grad.data.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n


class TopKCompressor:
    """
    Top-K gradient sparsification with error feedback.

    Keeps the K largest-magnitude gradient elements per round,
    accumulates the rest in an error buffer for the next step.

    Args:
        ratio:       Fraction of gradients to keep (0.01 = top 1%).
        error_feedback: Accumulate compressed-out gradients.

    Example::

        comp = TopKCompressor(ratio=0.01)
        comp.compress(model)
        sparse = comp.last_sparse        # indices + values
        comp.decompress(model, sparse)   # restore to model.grad
    """

    def __init__(self, ratio: float = 0.1, error_feedback: bool = True) -> None:
        self.ratio          = ratio
        self.error_feedback = error_feedback
        self._error_buffer: torch.Tensor | None = None
        self.last_sparse:   dict | None = None

    def compress(self, model: nn.Module) -> dict:
        """
        Compress model gradients via top-K sparsification.

        Returns:
            Dict with ``indices`` and ``values`` (sparse representation).
        """
        flat = _flat_grads(model)

        # Add error feedback
        if self.error_feedback and self._error_buffer is None:
            self._error_buffer = torch.zeros_like(flat)
        if self.error_feedback:
            flat = flat + self._error_buffer

        # Select top-K by magnitude
        k     = max(1, int(flat.numel() * self.ratio))
        _, idx = torch.topk(flat.abs(), k)
        vals   = flat[idx]

        # Store residual in error buffer
        if self.error_feedback:
            residual = flat.clone()
            residual[idx] = 0.0
            self._error_buffer = residual

        sparse = {"indices": idx, "values": vals, "n_total": flat.numel()}
        self.last_sparse = sparse

        # Zero out non-top-K gradients
        flat_new = torch.zeros_like(flat)
        flat_new[idx] = vals
        _set_flat_grads(model, flat_new)
        return sparse

    def decompress(self, model: nn.Module, sparse: dict) -> None:
        """
        Restore sparse gradients to model (for aggregation).

        Args:
            model:  Target model.
            sparse: Dict from :meth:`compress`.
        """
        flat = torch.zeros(sparse["n_total"])
        flat[sparse["indices"]] = sparse["values"]
        _set_flat_grads(model, flat)

    def compression_ratio(self) -> float:
        """Actual compression ratio of the last compress() call."""
        if self.last_sparse is None:
            return 1.0
        return len(self.last_sparse["values"]) / self.last_sparse["n_total"]


class QuantisedCompressor:
    """
    Gradient quantisation: float32 → int8 with scale/zero-point.

    Args:
        bits: Number of bits (8 or 4).
    """

    def __init__(self, bits: int = 8) -> None:
        assert bits in (4, 8)
        self.bits  = bits
        self.qmax  = (1 << bits) - 1

    def quantise(self, flat: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Quantise a flat gradient tensor to int8."""
        min_v  = flat.min().item()
        max_v  = flat.max().item()
        scale  = (max_v - min_v) / max(self.qmax, 1e-8)
        zp     = -min_v / max(scale, 1e-8)
        q      = ((flat - min_v) / max(scale, 1e-8)).round().clamp(0, self.qmax).to(torch.uint8)
        return q, scale, min_v

    def dequantise(self, q: torch.Tensor, scale: float, min_v: float) -> torch.Tensor:
        """Restore float32 from quantised tensor."""
        return q.float() * scale + min_v
