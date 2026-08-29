"""NanoMind quantization sub-package — INT8 post-training quantization.

Reduces model size 4x (float32 → int8) with minimal accuracy impact.

Modes:
    ``"weight_only"`` — weights quantized offline, activations remain float
    ``"dynamic"``     — weights offline + activations quantized at runtime

Granularity:
    ``"per_tensor"``  — one scale per weight matrix (fastest)
    ``"per_channel"`` — one scale per output channel (more accurate)

Primary exports:
    - :func:`quantize_model`              — replace Linear layers with QuantizedLinear
    - :class:`QuantConfig`                — mode, granularity, skip_modules config
    - :class:`QuantizedLinear`            — INT8 weight storage + float32 forward
    - :class:`DynamicQuantizedLinear`     — INT8 weights + runtime activation quant
    - :func:`quantize_tensor`             — quantize a float tensor to INT8
    - :func:`dequantize_tensor`           — dequantize INT8 back to float
    - :func:`quantization_stats`          — size and compression statistics
    - :func:`quantization_error`          — weight reconstruction MSE
    - :func:`print_quantization_report`   — pretty-print size + accuracy report
    - :func:`save_quantized_checkpoint`   — save INT8 model (4x smaller)
    - :func:`load_quantized_checkpoint`   — load INT8 model checkpoint
    - :class:`ActivationCalibrator`       — collect activation stats for calibration
"""

from nanomind.quant.config import QuantConfig
from nanomind.quant.ops import quantize_tensor, dequantize_tensor
from nanomind.quant.layer import QuantizedLinear
from nanomind.quant.dynamic import DynamicQuantizedLinear
from nanomind.quant.quantize import quantize_model
from nanomind.quant.stats import (
    quantization_stats,
    quantization_error,
    print_quantization_report,
    model_size_bytes,
)
from nanomind.quant.checkpoint import save_quantized_checkpoint, load_quantized_checkpoint
from nanomind.quant.calibrate import ActivationCalibrator

__all__ = [
    "QuantConfig",
    "QuantizedLinear",
    "DynamicQuantizedLinear",
    "quantize_model",
    "quantize_tensor",
    "dequantize_tensor",
    "quantization_stats",
    "quantization_error",
    "print_quantization_report",
    "model_size_bytes",
    "save_quantized_checkpoint",
    "load_quantized_checkpoint",
    "ActivationCalibrator",
]
