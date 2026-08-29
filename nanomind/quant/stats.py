"""
nanomind/quant/stats.py — Quantization size and accuracy impact analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.quant.layer import QuantizedLinear
from nanomind.quant.dynamic import DynamicQuantizedLinear
from nanomind.quant.ops import dequantize_tensor


def model_size_bytes(model: nn.Module) -> int:
    """
    Estimate model parameter storage size in bytes.

    INT8 parameters count as 1 byte; float32 as 4; float16 as 2.
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def quantization_stats(
    original: nn.Module,
    quantized: nn.Module,
) -> dict:
    """
    Compare original and quantized model sizes.

    Args:
        original:  Float32 model before quantization.
        quantized: INT8 model after quantization.

    Returns:
        Dict with:
        - ``original_mb``   : float32 model size in MB
        - ``quantized_mb``  : quantized model size in MB
        - ``compression``   : compression ratio (original / quantized)
        - ``size_reduction``: percentage size reduction
        - ``n_quant_layers``: number of quantized layers
    """
    orig_bytes = model_size_bytes(original)
    quant_bytes = model_size_bytes(quantized)

    n_ql = sum(
        1 for m in quantized.modules()
        if isinstance(m, (QuantizedLinear, DynamicQuantizedLinear))
    )

    compression = orig_bytes / max(quant_bytes, 1)
    reduction   = 1.0 - quant_bytes / max(orig_bytes, 1)

    return {
        "original_mb":   orig_bytes / 1024 ** 2,
        "quantized_mb":  quant_bytes / 1024 ** 2,
        "compression":   compression,
        "size_reduction": reduction,
        "n_quant_layers": n_ql,
    }


def quantization_error(
    original: nn.Module,
    quantized: nn.Module,
) -> dict:
    """
    Compute mean squared error between original and dequantized weights.

    Args:
        original:  Float32 model.
        quantized: Quantized model.

    Returns:
        Dict with ``mean_mse`` and ``max_mse`` across all quantized layers.
    """
    mse_list = []
    for (n1, m1), (n2, m2) in zip(
        original.named_modules(), quantized.named_modules()
    ):
        if isinstance(m2, (QuantizedLinear, DynamicQuantizedLinear)):
            w_orig = m1.weight.data.float()
            w_dq   = dequantize_tensor(m2.weight_int8, m2.scales, m2.granularity)
            mse    = ((w_orig - w_dq) ** 2).mean().item()
            mse_list.append(mse)

    if not mse_list:
        return {"mean_mse": 0.0, "max_mse": 0.0}
    return {
        "mean_mse": sum(mse_list) / len(mse_list),
        "max_mse":  max(mse_list),
    }


def print_quantization_report(original: nn.Module, quantized: nn.Module) -> None:
    """Print a side-by-side quantization summary."""
    stats = quantization_stats(original, quantized)
    err   = quantization_error(original, quantized)
    print("=" * 55)
    print("Quantization Report")
    print("=" * 55)
    print(f"  Original size   : {stats['original_mb']:.2f} MB")
    print(f"  Quantized size  : {stats['quantized_mb']:.2f} MB")
    print(f"  Compression     : {stats['compression']:.2f}x")
    print(f"  Size reduction  : {stats['size_reduction']:.1%}")
    print(f"  Quant layers    : {stats['n_quant_layers']}")
    print(f"  Mean weight MSE : {err['mean_mse']:.2e}")
    print(f"  Max  weight MSE : {err['max_mse']:.2e}")
    print("=" * 55)
