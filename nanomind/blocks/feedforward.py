"""
nanomind/blocks/feedforward.py — Position-wise feed-forward network (FFN).

The FFN is applied identically to each position in the sequence.
It consists of two linear transformations with a non-linearity in between:

    FFN(x) = Linear_2(activation(Linear_1(x)))

The hidden dimension is typically 4x the model dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model:    Input/output dimension.
        d_ff:       Hidden dimension (default: 4 * d_model).
        dropout:    Dropout probability applied after activation.
        activation: Activation function name — ``"gelu"`` or ``"swiglu"``.
        bias:       Whether to use bias in linear layers.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.d_ff       = d_ff or 4 * d_model
        self.activation = activation.lower()
        self.dropout    = dropout

        if self.activation == "swiglu":
            # SwiGLU needs two parallel projections for the gating mechanism
            self.fc1_gate = nn.Linear(d_model, self.d_ff, bias=bias)
            self.fc1_up   = nn.Linear(d_model, self.d_ff, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)

        self.fc2     = nn.Linear(self.d_ff, d_model, bias=bias)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation to each position."""
        if self.activation == "gelu":
            return self._forward_gelu(x)
        if self.activation == "swiglu":
            return self._forward_swiglu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _forward_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Standard two-layer FFN with GELU activation."""
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

    def _forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: gate(x) * sigmoid(gate(x)) * up(x)."""
        # SwiGLU: element-wise product of SiLU(gate) and up-projection
        # Reference: Shazeer (2020) — https://arxiv.org/abs/2002.05202
        gate = F.silu(self.fc1_gate(x))   # SiLU = x * sigmoid(x) = Swish
        up   = self.fc1_up(x)
        return self.fc2(self.drop(gate * up))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"activation={self.activation}, dropout={self.dropout}"
        )
