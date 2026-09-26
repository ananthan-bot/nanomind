"""NanoMind Quantization sub-package — Model compression via quantization.

Implements the full quantization pipeline:
  1. QuantConfig          — bits, scheme, granularity, method, compression_ratio
  2. TensorQuantizer      — compute_scale_zero, quantize/dequantize, per-group
  3. RTNQuantizer         — Round-To-Nearest PTQ, apply(), model_size_bytes()
  4. GPTQQuantizer        — column-wise OBS quantization, HessianCollector
  5. AWQQuantizer         — activation-aware scaling, ActivationScaleCollector
  6. STEQuantize          — straight-through estimator for QAT
  7. QATLinear            — fake-quantized Linear layer with STE
  8. convert_to_qat       — convert model to QAT in-place
  9. ModelCalibrator      — sensitivity analysis, mixed-precision suggestions
  10. LayerStats          — per-layer quantization statistics

Primary exports:
    - :class:`QuantConfig`           — quantization configuration
    - :class:`TensorQuantizer`       — calibrate, quantize, fake_quantize, error
    - :func:`compute_scale_zero`     — scale and zero point computation
    - :func:`quantize`               — float → int
    - :func:`dequantize`             — int → float
    - :func:`quantize_dequantize`    — QDQ (fake quantization)
    - :class:`RTNQuantizer`          — RTN PTQ, apply, layer_stats
    - :class:`GPTQQuantizer`         — GPTQ column-wise, quantize_blocks
    - :class:`HessianCollector`      — X^TX Hessian from forward passes
    - :class:`AWQQuantizer`          — activation-aware weight quantization
    - :class:`ActivationScaleCollector` — mean activation magnitude per channel
    - :class:`QATLinear`             — QAT-aware linear, effective_bits
    - :func:`convert_to_qat`         — convert all Linear → QATLinear
    - :class:`ModelCalibrator`       — run, report, sensitivity_analysis
    - :class:`LayerStats`            — name, mse, snr, shape, n_params
"""

from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import (
    TensorQuantizer, compute_scale_zero, quantize, dequantize, quantize_dequantize,
)
from nanomind.quant.rtn import RTNQuantizer
from nanomind.quant.gptq import GPTQQuantizer, HessianCollector
from nanomind.quant.awq import AWQQuantizer, ActivationScaleCollector
from nanomind.quant.qat import STEQuantize, QATLinear, convert_to_qat, fake_quant_ste
from nanomind.quant.calibration import ModelCalibrator, LayerStats

__all__ = [
    "QuantConfig",
    "TensorQuantizer", "compute_scale_zero", "quantize", "dequantize", "quantize_dequantize",
    "RTNQuantizer",
    "GPTQQuantizer", "HessianCollector",
    "AWQQuantizer", "ActivationScaleCollector",
    "STEQuantize", "QATLinear", "convert_to_qat", "fake_quant_ste",
    "ModelCalibrator", "LayerStats",
]
