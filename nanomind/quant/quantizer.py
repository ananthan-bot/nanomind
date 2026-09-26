"""
nanomind/quant/quantizer.py — Core quantization and dequantization operations.

Implements the fundamental operations:
  1. Compute scale and zero point from weight statistics
  2. Quantize: float32 → int (round-to-nearest)
  3. Dequantize: int → float32 (reconstruction)
  4. Quantize-Dequantize (QDQ): simulate quantization in float32

The QDQ operation is used in both PTQ evaluation and QAT.
"""

from __future__ import annotations
import torch
from nanomind.quant.config import QuantConfig


def compute_scale_zero(
    x:      torch.Tensor,
    cfg:    QuantConfig,
    dim:    int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale and zero point.

    Args:
        x:   Tensor to quantize.
        cfg: :class:`QuantConfig`.
        dim: Dimension to reduce over (None = per-tensor).

    Returns:
        ``(scale, zero_point)`` tensors.
    """
    q_min, q_max = cfg.q_min, cfg.q_max

    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

    # Apply clip ratio (AWQ-style outlier clipping)
    if cfg.clip_ratio < 1.0:
        x_abs = x.abs()
        clip  = x_abs.amax(dim=dim, keepdim=True) if dim else x_abs.max()
        x_max = (clip * cfg.clip_ratio).clamp(min=x_max.min())
        x_min = -x_max if cfg.scheme == "symmetric" else x_min

    if cfg.scheme == "symmetric":
        x_abs_max = torch.maximum(x_min.abs(), x_max.abs())
        scale     = x_abs_max / max(abs(q_max), 1e-8)
        zero      = torch.zeros_like(scale)
    else:
        scale = (x_max - x_min) / max(q_max - q_min, 1)
        zero  = (q_min - x_min / (scale + 1e-8)).round().clamp(q_min, q_max)

    return scale.clamp(min=1e-8), zero


def quantize(
    x:     torch.Tensor,
    scale: torch.Tensor,
    zero:  torch.Tensor,
    cfg:   QuantConfig,
) -> torch.Tensor:
    """
    Quantize float tensor to integer.

    Args:
        x:     ``(*,)`` float input.
        scale: Scale tensor (broadcastable to x).
        zero:  Zero point tensor.
        cfg:   :class:`QuantConfig`.

    Returns:
        Integer quantized tensor (stored as int32 for safety).
    """
    q = (x / scale + zero).round().clamp(cfg.q_min, cfg.q_max)
    return q.to(torch.int32)


def dequantize(
    q:     torch.Tensor,
    scale: torch.Tensor,
    zero:  torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize integer tensor back to float.

    Args:
        q:     Integer quantized tensor.
        scale: Scale tensor.
        zero:  Zero point tensor.

    Returns:
        Reconstructed float tensor.
    """
    return scale * (q.float() - zero)


def quantize_dequantize(
    x:     torch.Tensor,
    scale: torch.Tensor,
    zero:  torch.Tensor,
    cfg:   QuantConfig,
) -> torch.Tensor:
    """
    Quantize then immediately dequantize (QDQ / fake quantization).

    Returns a float tensor with quantization noise applied.
    Used for calibration, QAT, and error analysis.
    """
    return dequantize(quantize(x, scale, zero, cfg), scale, zero)


class TensorQuantizer:
    """
    Quantizer for a single tensor with configurable granularity.

    Args:
        cfg: :class:`QuantConfig`.

    Example::

        q  = TensorQuantizer(QuantConfig(bits=4, granularity="per_channel"))
        xq = q.quantize(weight)          # int32 tensor
        xf = q.dequantize(xq)            # float32 reconstruction
        err = q.quantization_error(weight)
    """

    def __init__(self, cfg: QuantConfig) -> None:
        self.cfg   = cfg
        self.scale: torch.Tensor | None = None
        self.zero:  torch.Tensor | None = None

    def calibrate(self, x: torch.Tensor) -> None:
        """Compute and store scale/zero point from a reference tensor."""
        if self.cfg.granularity == "per_tensor":
            self.scale, self.zero = compute_scale_zero(x, self.cfg)
        elif self.cfg.granularity == "per_channel":
            self.scale, self.zero = compute_scale_zero(x, self.cfg, dim=tuple(range(1, x.dim())))
        elif self.cfg.granularity == "per_group":
            self.scale, self.zero = self._per_group_scale(x)

    def _per_group_scale(self, x: torch.Tensor) -> tuple:
        """Compute per-group scale and zero point."""
        orig_shape = x.shape
        G          = self.cfg.group_size
        # Reshape to (..., n_groups, G)
        flat       = x.view(-1, x.shape[-1])
        n_rows, n_cols = flat.shape
        n_groups   = (n_cols + G - 1) // G
        padded     = n_groups * G
        pad        = torch.zeros(n_rows, padded - n_cols)
        flat_p     = torch.cat([flat, pad], dim=-1).view(n_rows * n_groups, G)
        s, z       = compute_scale_zero(flat_p, self.cfg, dim=1)
        return s.view(n_rows, n_groups, 1), z.view(n_rows, n_groups, 1)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            self.calibrate(x)
        return quantize(x, self.scale, self.zero, self.cfg)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return dequantize(q, self.scale, self.zero)

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """QDQ: float → int → float with quantization noise."""
        if self.scale is None:
            self.calibrate(x)
        return quantize_dequantize(x, self.scale, self.zero, self.cfg)

    def quantization_error(self, x: torch.Tensor) -> dict:
        """Measure quantization error metrics."""
        xq = self.fake_quantize(x)
        err = x - xq
        return {
            "mse":          err.pow(2).mean().item(),
            "mae":          err.abs().mean().item(),
            "max_err":      err.abs().max().item(),
            "snr_db":       10 * torch.log10(x.pow(2).mean() / (err.pow(2).mean() + 1e-10)).item(),
        }
