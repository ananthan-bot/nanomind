"""
nanomind/moe/config.py — Mixture of Experts configuration.

MoE replaces the dense FFN in each transformer block with a sparse mixture:

  Dense FFN :   every token uses THE SAME d_model → d_ff → d_model transform
  MoE FFN   :   every token chooses the TOP-K of N experts (smaller FFNs)

Parameters scale with N experts but compute scales with only K:
  - Parameters : N × (d_model × d_ff × 2) — huge capacity
  - Compute    : K × (d_model × d_ff × 2) — same as K dense layers

Reference: Fedus et al. (2021) Switch Transformer — https://arxiv.org/abs/2101.03961
           Jiang et al. (2024) Mixtral 8×7B    — https://arxiv.org/abs/2401.04088
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """
    Configuration for a Sparse Mixture of Experts FFN layer.

    Attributes:
        num_experts:      Total number of expert FFNs (N). Typically 8, 16, or 64.
        top_k:            Number of experts each token routes to (K). Typically 1 or 2.
        expert_capacity:  Max tokens per expert per batch (None = no cap).
        load_balance_coef: Weight of the auxiliary load-balancing loss.
                           Set to 0.0 to disable. Typical: 0.01.
        expert_dropout:   Dropout probability applied to expert outputs during training.
        d_ff_expert:      Hidden dimension of each expert FFN.
                          Defaults to 4 × d_model if None.
        activation:       Expert FFN activation (``"gelu"`` or ``"relu"`` or ``"swiglu"``).
    """

    num_experts:      int   = 8
    top_k:            int   = 2
    expert_capacity:  int | None = None
    load_balance_coef: float = 0.01
    expert_dropout:   float = 0.0
    d_ff_expert:      int | None = None
    activation:       str   = "gelu"

    def __post_init__(self) -> None:
        assert self.num_experts >= 1
        assert 1 <= self.top_k <= self.num_experts
        assert self.load_balance_coef >= 0.0
        assert self.activation in ("gelu", "relu", "swiglu")
