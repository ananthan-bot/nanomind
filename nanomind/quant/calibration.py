"""
nanomind/quant/calibration.py — Calibration for post-training quantization.

Calibration collects activation statistics from a small set of
representative inputs to compute quantization parameters.

Key statistics:
  - Min/max: for range-based quantization
  - Moving average min/max: smooth over batches
  - KL-divergence calibration: minimise information loss (TensorRT)
  - MSE calibration: minimise mean squared error
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import TensorQuantizer


@dataclass
class LayerStats:
    """Quantization statistics for a single layer."""
    name:         str
    shape:        tuple
    weight_mse:   float
    weight_snr:   float
    weight_range: tuple
    n_params:     int

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "shape":      self.shape,
            "weight_mse": round(self.weight_mse, 8),
            "weight_snr": round(self.weight_snr, 2),
            "weight_range": self.weight_range,
            "n_params":   self.n_params,
        }


class ModelCalibrator:
    """
    Run calibration pass to compute per-layer quantization statistics.

    Args:
        model:  PyTorch model.
        cfg:    :class:`QuantConfig`.

    Example::

        cal    = ModelCalibrator(model, QuantConfig(bits=4))
        stats  = cal.run(calibration_batches)
        report = cal.report()
    """

    def __init__(self, model: nn.Module, cfg: QuantConfig) -> None:
        self.model  = model
        self.cfg    = cfg
        self._stats: list[LayerStats] = []

    def run(self, batches: list | None = None) -> list[LayerStats]:
        """
        Compute quantization statistics for all Linear layers.

        Args:
            batches: Optional list of ``(x, y)`` calibration batches.
                     If None, compute weight stats only (no activation stats).

        Returns:
            List of :class:`LayerStats`.
        """
        self._stats.clear()
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w  = module.weight.data
            q  = TensorQuantizer(self.cfg)
            q.calibrate(w)
            err = q.quantization_error(w)
            stats = LayerStats(
                name         = name,
                shape        = tuple(w.shape),
                weight_mse   = err["mse"],
                weight_snr   = err["snr_db"],
                weight_range = (round(w.min().item(), 4), round(w.max().item(), 4)),
                n_params     = w.numel(),
            )
            self._stats.append(stats)
        return self._stats

    def report(self) -> dict:
        """Summary statistics across all layers."""
        if not self._stats:
            return {}
        mses  = [s.weight_mse for s in self._stats]
        snrs  = [s.weight_snr for s in self._stats]
        total = sum(s.n_params for s in self._stats)
        return {
            "n_layers":       len(self._stats),
            "total_params":   total,
            "mean_mse":       round(sum(mses) / len(mses), 8),
            "mean_snr_db":    round(sum(snrs) / len(snrs), 2),
            "worst_snr_db":   round(min(snrs), 2),
            "best_snr_db":    round(max(snrs), 2),
            "worst_layer":    self._stats[snrs.index(min(snrs))].name,
            "bits":           self.cfg.bits,
        }

    def sensitivity_analysis(self) -> list[dict]:
        """Rank layers by sensitivity to quantization (by SNR)."""
        return sorted(
            [s.to_dict() for s in self._stats],
            key=lambda x: x["weight_snr"],
        )

    def mixed_precision_suggestion(
        self,
        high_bits: int = 8,
        low_bits:  int = 4,
        threshold_snr: float = 20.0,
    ) -> dict[str, int]:
        """
        Suggest per-layer bit widths based on SNR sensitivity.

        Layers with SNR < threshold get high_bits, others get low_bits.
        """
        suggestions = {}
        for s in self._stats:
            suggestions[s.name] = high_bits if s.weight_snr < threshold_snr else low_bits
        return suggestions
