"""
nanomind/lora/layer.py — LoRA-augmented Linear layer.

Wraps a frozen ``nn.Linear`` with two trainable low-rank matrices A and B::

    output = x @ W.T + scaling * x @ A.T @ B.T
           = base_output + lora_output

where:
  - W is the frozen pre-trained weight (shape: out_features × in_features)
  - A is the down-projection: (r × in_features) — initialised with Kaiming Normal
  - B is the up-projection:   (out_features × r) — initialised with zeros
  - B = 0 at init means LoRA output is zero at the start of training,
    so the model begins from the pre-trained state

Only A and B are updated during fine-tuning; W is kept frozen.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A drop-in replacement for ``nn.Linear`` with LoRA adaptation.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        r:            LoRA rank.
        alpha:        LoRA scaling (effective scale = alpha / r).
        dropout:      Dropout on LoRA path input.
        bias:         Whether the base linear has a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r            = r
        self.scaling      = alpha / r
        self.merged       = False

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Optional dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise A with Kaiming Normal; B stays zero (safe start)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Create a LoRALinear by wrapping an existing ``nn.Linear``.

        Copies the pre-trained weight (and bias if present) and freezes them.

        Args:
            linear: Existing linear layer to adapt.
            r:      LoRA rank.
            alpha:  LoRA scaling.
            dropout: LoRA dropout.

        Returns:
            New :class:`LoRALinear` with copied frozen base weights.
        """
        has_bias = linear.bias is not None
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )
        lora.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora.bias_param.data.copy_(linear.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base linear + LoRA delta.

        Args:
            x: Input ``(..., in_features)``

        Returns:
            Output ``(..., out_features)``
        """
        base_out = F.linear(x, self.weight, self.bias_param)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, scaling={self.scaling:.3f}, merged={self.merged}"
        )
