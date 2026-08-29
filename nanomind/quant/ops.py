"""
nanomind/quant/ops.py — Core INT8 quantization and dequantization operations.

Symmetric quantization maps float values to the range [-127, 127]:

    scale  = max(|x|) / 127
    x_int8 = round(x / scale).clamp(-127, 127)
    x_fp32 = x_int8 * scale   (dequantize)

Per-channel quantization computes one scale per output channel (row),
which reduces quantization error significantly for weight matrices.
"""

from __future__ import annotations

import torch


INT8_MAX = 127.0
INT8_MIN = -128.0


def quantize_per_tensor(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-tensor INT8 quantization.

    Args:
        x: Float tensor to quantize.

    Returns:
        Tuple of ``(x_int8, scale)`` where scale is a scalar tensor.
    """
    scale   = x.abs().max() / INT8_MAX
    scale   = scale.clamp(min=1e-8)
    x_int8  = (x / scale).round().clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return x_int8, scale


def dequantize_per_tensor(
    x_int8: torch.Tensor,
    scale:  torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a per-tensor INT8 tensor back to float32.

    Args:
        x_int8: INT8 quantized tensor.
        scale:  Scalar scale factor.

    Returns:
        Dequantized float32 tensor.
    """
    return x_int8.to(torch.float32) * scale


def quantize_per_channel(
    x: torch.Tensor,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel INT8 quantization.

    Computes one scale per slice along ``dim`` (typically output channels).

    Args:
        x:   2D float weight tensor ``(out_features, in_features)``.
        dim: Dimension along which to compute per-channel scales (default: 0).

    Returns:
        Tuple of:
        - ``x_int8`` : INT8 quantized tensor, same shape as input
        - ``scales`` : Scale tensor ``(out_features,)``
    """
    assert x.dim() == 2, "per_channel quantization requires 2D tensor"
    # Max absolute value per output channel
    scales = x.abs().max(dim=1 - dim).values / INT8_MAX   # (out_features,)
    scales = scales.clamp(min=1e-8)

    # Divide each row by its scale
    x_fp   = x / scales.unsqueeze(1)                     # broadcast over in_features
    x_int8 = x_fp.round().clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return x_int8, scales


def dequantize_per_channel(
    x_int8: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a per-channel INT8 tensor back to float32.

    Args:
        x_int8: INT8 quantized tensor ``(out_features, in_features)``.
        scales: Per-channel scale factors ``(out_features,)``.

    Returns:
        Dequantized float32 tensor.
    """
    return x_int8.to(torch.float32) * scales.unsqueeze(1)


def quantize_tensor(
    x:           torch.Tensor,
    granularity: str = "per_channel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a float tensor to INT8 using the specified granularity.

    Args:
        x:           Weight tensor to quantize (1D or 2D).
        granularity: ``"per_tensor"`` or ``"per_channel"``.

    Returns:
        Tuple of ``(x_int8, scale_or_scales)``.
    """
    if granularity == "per_channel" and x.dim() == 2:
        return quantize_per_channel(x)
    return quantize_per_tensor(x)


def dequantize_tensor(
    x_int8:  torch.Tensor,
    scales:  torch.Tensor,
    granularity: str = "per_channel",
) -> torch.Tensor:
    """
    Dequantize an INT8 tensor using per-tensor or per-channel scales.

    Args:
        x_int8:      INT8 quantized tensor.
        scales:      Scale(s) from quantization.
        granularity: Must match the granularity used during quantization.

    Returns:
        Dequantized float32 tensor.
    """
    if granularity == "per_channel" and x_int8.dim() == 2:
        return dequantize_per_channel(x_int8, scales)
    return dequantize_per_tensor(x_int8, scales)
