"""
nanomind/quant/rtn.py — Round-To-Nearest (RTN) post-training quantization.

RTN is the simplest PTQ method:
  1. For each weight tensor: compute scale from abs-max
  2. Round weights to nearest integer: q = round(w / scale)
  3. Store quantized integers + scale for dequantization at inference

Quality: fast but suboptimal — ignores inter-weight dependencies.
Better methods (GPTQ, AWQ) compensate for quantization error.

RTN is baseline for all other PTQ methods.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import TensorQuantizer
from nanomind.utils.logger import get_logger

log = get_logger("quant.rtn")


class RTNQuantizer:
    """
    Round-To-Nearest post-training quantization.

    Quantizes all nn.Linear weight matrices in a model.

    Args:
        model:  PyTorch model.
        cfg:    :class:`QuantConfig`.

    Example::

        q_model = RTNQuantizer(model, QuantConfig(bits=4))
        q_model.quantize()
        stats = q_model.layer_stats()
    """

    def __init__(self, model: nn.Module, cfg: QuantConfig) -> None:
        self.model  = model
        self.cfg    = cfg
        self._qdata: dict[str, dict] = {}   # name → {q_weight, scale, zero}

    def quantize(self) -> dict:
        """
        Quantize all Linear layers in the model.

        Returns:
            Dict of {layer_name: error_metrics}.
        """
        results = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            q = TensorQuantizer(self.cfg)
            q.calibrate(w)
            err = q.quantization_error(w)
            self._qdata[name] = {
                "q_weight": q.quantize(w),
                "scale":    q.scale,
                "zero":     q.zero,
                "shape":    tuple(w.shape),
            }
            results[name] = err
            log.info(f"RTN: {name} → SNR={err['snr_db']:.1f}dB, MSE={err['mse']:.6f}")
        return results

    def apply(self) -> None:
        """Apply quantization: replace Linear weights with dequantized versions."""
        for name, module in self.model.named_modules():
            if name not in self._qdata:
                continue
            d    = self._qdata[name]
            w_q  = dequantize_weight(d["q_weight"], d["scale"], d["zero"])
            module.weight.data = w_q

    def model_size_bytes(self) -> int:
        """Estimated compressed model size in bytes."""
        total = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            n = module.weight.numel()
            if name in self._qdata:
                total += n * self.cfg.bytes_per_param
            else:
                total += n * 4   # FP32
        return int(total)

    def layer_stats(self) -> list[dict]:
        """Summary of quantized layers."""
        return [
            {"name": name, "shape": d["shape"],
             "bits": self.cfg.bits,
             "params": d["shape"][0] * d["shape"][1]}
            for name, d in self._qdata.items()
        ]


def dequantize_weight(q, scale, zero):
    return scale * (q.float() - zero)
