"""
nanomind/quant/layer.py — INT8 Quantized Linear layer.

Stores weights in INT8 format (4x less memory than float32).
During forward pass, dequantizes weights back to float for the matmul,
then re-quantizes (weight-only quantization) or quantizes activations
too (dynamic quantization).

Memory comparison:
    nn.Linear(768, 768):         ~ 2.25 MB  (float32)
    QuantizedLinear(768, 768):   ~ 0.58 MB  (int8 weights + float scale)
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


class QuantizedLinear(nn.Module):
    """
    INT8 weight-quantized drop-in replacement for ``nn.Linear``.

    Weights are stored as INT8 and dequantized to float32 on-the-fly
    during the forward pass. This trades a small amount of compute
    for a 4x reduction in weight memory.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to include a bias term (stored as float32).
        granularity:  ``"per_tensor"`` or ``"per_channel"``.

    Attributes:
        weight_int8: INT8 quantized weights ``(out_features, in_features)``.
        scales:      Quantization scale(s).
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

        # Weights stored as INT8 (non-trainable)
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        # Scales: scalar (per_tensor) or (out_features,) (per_channel)
        if granularity == "per_channel":
            self.register_buffer("scales", torch.ones(out_features))
        else:
            self.register_buffer("scales", torch.ones(1))

        self.bias = (
            nn.Parameter(torch.zeros(out_features))
            if bias else None
        )

    @classmethod
    def from_linear(
        cls,
        linear:      nn.Linear,
        granularity: str = "per_channel",
    ) -> "QuantizedLinear":
        """
        Create a QuantizedLinear by quantizing an existing ``nn.Linear``.

        Copies and quantizes the weight tensor; copies bias as-is.

        Args:
            linear:      Source linear layer.
            granularity: Quantization granularity.

        Returns:
            New :class:`QuantizedLinear` with INT8 weight storage.
        """
        has_bias = linear.bias is not None
        ql = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            granularity=granularity,
        )
        w_int8, scales = quantize_tensor(linear.weight.data.float(), granularity)
        ql.weight_int8.copy_(w_int8)
        ql.scales.copy_(scales)
        if has_bias:
            ql.bias.data.copy_(linear.bias.data)
        return ql

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantize weights, then standard linear.

        Args:
            x: Input ``(..., in_features)``

        Returns:
            Output ``(..., out_features)``
        """
        w_fp32 = dequantize_tensor(self.weight_int8, self.scales, self.granularity)
        return F.linear(x, w_fp32, self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Return dequantized weight for compatibility with standard Linear API."""
        return dequantize_tensor(self.weight_int8, self.scales, self.granularity)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"granularity={self.granularity}, "
            f"storage=int8"
        )
