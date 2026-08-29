"""
nanomind/quant/calibrate.py — Calibration for quantization scale computation.

For weight-only quantization, calibration is not strictly needed (scales
are computed from weight statistics alone). For activation quantization,
calibration data is used to estimate the typical range of activations,
providing better scales than per-batch dynamic quantization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

from nanomind.utils.logger import get_logger

log = get_logger("quant.calibrate")


class ActivationCalibrator:
    """
    Collect activation statistics for calibration.

    Hooks into ``nn.Linear`` forward calls to record the range
    of activations seen on calibration data. The collected statistics
    can then be used to set quantization scales.

    Args:
        model: Model to calibrate.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model   = model
        self.stats:  dict[str, dict] = {}
        self._hooks  = []

    def _make_hook(self, name: str) -> Callable:
        def hook(module, inp, out):
            x = inp[0].detach().float()
            if name not in self.stats:
                self.stats[name] = {"min": x.min().item(), "max": x.max().item(),
                                    "abs_max": x.abs().max().item()}
            else:
                self.stats[name]["min"]     = min(self.stats[name]["min"],     x.min().item())
                self.stats[name]["max"]     = max(self.stats[name]["max"],     x.max().item())
                self.stats[name]["abs_max"] = max(self.stats[name]["abs_max"], x.abs().max().item())
        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on all Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def calibrate(
        self,
        loader:    DataLoader,
        max_batches: int = 8,
    ) -> dict:
        """
        Run calibration data through the model and collect statistics.

        Args:
            loader:      DataLoader yielding (x, y) or x batches.
            max_batches: Maximum number of batches to process.

        Returns:
            Activation statistics dict keyed by layer name.
        """
        self.model.eval()
        self.register_hooks()
        processed = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(next(self.model.parameters()).device)
            self.model(x)
            processed += 1
            if processed >= max_batches:
                break
        self.remove_hooks()
        log.info(
            f"Calibrated on {processed} batches, "
            f"collected stats for {len(self.stats)} layers."
        )
        return self.stats
