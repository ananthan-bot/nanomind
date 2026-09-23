"""
nanomind/moe_v2/expert.py — Individual expert and expert bank.

Each expert is an independent FFN:
  Expert_i(x) = W2_i × GELU(W1_i × x + b1_i) + b2_i

Expert banks allow parallel computation of multiple experts.

SwiGLU experts (LLaMA-style):
  Expert(x) = W2 × (SiLU(W_gate × x) × W1 × x)
  Better performance than standard GELU FFN.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.moe_v2.config import MoEConfig


class Expert(nn.Module):
    """
    A single FFN expert.

    Args:
        d_model:  Input/output dimension.
        d_ff:     Hidden dimension.
        use_bias: Include bias.
        variant:  ``"gelu"`` or ``"swiglu"``.
    """

    def __init__(
        self,
        d_model:  int,
        d_ff:     int,
        use_bias: bool = True,
        variant:  str  = "gelu",
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant == "swiglu":
            # SwiGLU: gate + value projections
            self.w_gate = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w_val  = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w_out  = nn.Linear(d_ff, d_model, bias=use_bias)
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "swiglu":
            return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))
        return self.w2(F.gelu(self.w1(x)))


class ExpertBank(nn.Module):
    """
    A bank of N independent expert FFNs.

    Stores all experts in a single module list for easy iteration.

    Args:
        cfg: :class:`MoEConfig`.

    Example::

        bank = ExpertBank(MoEConfig(n_experts=8, d_model=128, d_ff=512))
        # Route x through expert 3:
        out = bank.experts[3](x)
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.experts   = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_ff, cfg.use_bias)
            for _ in range(cfg.n_experts)
        ])
        # Shared experts (always active, DeepSeek-MoE style)
        self.shared = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_ff, cfg.use_bias)
            for _ in range(cfg.shared_experts)
        ])

    def forward_expert(
        self,
        expert_idx: int,
        x:          torch.Tensor,
    ) -> torch.Tensor:
        """Run a single expert."""
        return self.experts[expert_idx](x)

    def shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum all shared expert outputs."""
        if not self.shared:
            return torch.zeros_like(x)
        return sum(e(x) for e in self.shared)

    @property
    def n_params_per_expert(self) -> int:
        if not self.experts:
            return 0
        return sum(p.numel() for p in self.experts[0].parameters())

    @property
    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
