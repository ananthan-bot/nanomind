"""
nanomind/quant/dynamic.py — Dynamic INT8 quantization (weights + activations).

Dynamic quantization quantizes both weights (offline) and activations
(at runtime per-batch). This gives better accuracy than weight-only
at the cost of quantizing activations dynamically during inference.

Used in: PyTorch's ``torch.quantization.quantize_dynamic()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.quant.ops import (
    quantize_tensor,
    dequantize_tensor,
    quantize_per_tensor,
    dequantize_per_tensor,
)


class DynamicQuantizedLinear(nn.Module):
    """
    Dynamically quantized linear layer.

    Weights are quantized offline and stored as INT8.
    Activations are quantized to INT8 at runtime (per batch)
    and dequantized after the operation.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to include bias.
        granularity:  Weight quantization granularity.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        granularity:  str  = "per_channel",
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.granularity  = granularity

        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        if granularity == "per_channel":
            self.register_buffer("scales", torch.ones(out_features))
        else:
            self.register_buffer("scales", torch.ones(1))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(
        cls,
        linear:      nn.Linear,
        granularity: str = "per_channel",
    ) -> "DynamicQuantizedLinear":
        has_bias = linear.bias is not None
        dql = cls(linear.in_features, linear.out_features, has_bias, granularity)
        w_int8, scales = quantize_tensor(linear.weight.data.float(), granularity)
        dql.weight_int8.copy_(w_int8)
        dql.scales.copy_(scales)
        if has_bias:
            dql.bias.data.copy_(linear.bias.data)
        return dql

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamic quantization: quantize input activations at runtime.
        """
        # Quantize activations per-tensor
        x_flat     = x.reshape(-1, x.shape[-1])
        x_int8, x_scale = quantize_per_tensor(x_flat.float())
        x_dq       = dequantize_per_tensor(x_int8, x_scale).reshape_as(x)

        # Dequantize weights
        w_fp32 = dequantize_tensor(self.weight_int8, self.scales, self.granularity)
        return F.linear(x_dq, w_fp32, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"granularity={self.granularity}, dynamic=True"
        )
