"""
nanomind/moe/expert.py — Single FFN Expert module.

Each expert is a standard two-layer FFN, identical in structure to the
dense FFN in a regular transformer block but with its own independent weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    A single FFN expert in a Mixture of Experts layer.

    Architecture: Linear → Activation → Linear (same as dense FFN).

    Args:
        d_model:    Input/output dimension.
        d_ff:       Hidden dimension (typically 4 × d_model).
        activation: Activation function: ``"gelu"``, ``"relu"``, or ``"swiglu"``.
        bias:       Whether to include bias terms.
    """

    def __init__(
        self,
        d_model:    int,
        d_ff:       int,
        activation: str  = "gelu",
        bias:       bool = False,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.activation = activation

        if activation == "swiglu":
            # SwiGLU: two up projections, elementwise gate × linear
            self.gate = nn.Linear(d_model, d_ff, bias=bias)
            self.up   = nn.Linear(d_model, d_ff, bias=bias)
        else:
            self.fc1  = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.fc2(F.silu(self.gate(x)) * self.up(x))
        h = self.fc1(x)
        if self.activation == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h)
        return self.fc2(h)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, act={self.activation}"
