"""
nanomind/moe_v2/config.py — MoE configuration.

## Mixture of Experts (MoE) in LLMs

Standard MoE replaces dense FFN layers with N expert FFN networks,
routing each token to the top-K experts via a learned gating function.

Key concepts:
  n_experts:      Total experts (e.g., 8, 64, 128, 256)
  top_k:          Active experts per token (e.g., 2, 4)
  capacity_factor: Max tokens per expert = capacity_factor × tokens/experts
  load_balance:   Auxiliary loss to prevent all tokens routing to one expert

Famous MoE models:
  GShard (Lepikhin et al., 2021):   Top-2 routing, 600B params
  Switch Transformer (Fedus et al., 2022): Top-1, 1.6T params
  GLaM (Du et al., 2022):           64 experts, Top-2
  Mixtral (Mistral AI, 2024):       8 experts, Top-2
  DeepSeek-MoE (2024):              64 experts, fine-grained routing

References:
  Shazeer et al. (2017) "Outrageously Large Neural Networks"
  https://arxiv.org/abs/1701.06538
  Fedus et al. (2022) "Switch Transformers"
  https://arxiv.org/abs/2101.03961
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """
    Configuration for a Mixture of Experts layer.

    Attributes:
        n_experts:        Total number of expert networks.
        top_k:            Number of experts each token is routed to.
        d_model:          Input/output feature dimension.
        d_ff:             Expert hidden dimension.
        capacity_factor:  Tokens per expert = cf × (total_tokens / n_experts).
        router_type:      ``"topk"``, ``"expert_choice"``, or ``"hash"``.
        load_balance_coef: Coefficient for auxiliary load balancing loss.
        router_noise:     Jitter noise std for top-K routing (prevents collapse).
        dropout:          Expert dropout rate.
        use_bias:         Whether experts use bias terms.
        shared_experts:   Number of always-active shared experts (DeepSeek-style).
    """
    n_experts:        int   = 8
    top_k:            int   = 2
    d_model:          int   = 128
    d_ff:             int   = 512
    capacity_factor:  float = 1.25
    router_type:      str   = "topk"
    load_balance_coef: float = 1e-2
    router_noise:     float = 0.01
    dropout:          float = 0.0
    use_bias:         bool  = True
    shared_experts:   int   = 0

    def __post_init__(self) -> None:
        assert self.n_experts   >= 1
        assert 1 <= self.top_k  <= self.n_experts
        assert self.d_model     > 0
        assert self.d_ff        > 0
        assert self.capacity_factor > 0.0
        assert self.router_type in ("topk", "expert_choice", "hash")
        assert self.shared_experts >= 0

    @property
    def active_ratio(self) -> float:
        """Fraction of parameters used per token (top_k / n_experts)."""
        return self.top_k / self.n_experts

    @property
    def total_params_estimate(self) -> int:
        """Rough parameter count for all experts."""
        per_expert = 2 * self.d_model * self.d_ff
        return self.n_experts * per_expert + self.d_model * self.n_experts

    def to_dict(self) -> dict:
        return {
            "n_experts":    self.n_experts,
            "top_k":        self.top_k,
            "d_model":      self.d_model,
            "d_ff":         self.d_ff,
            "router_type":  self.router_type,
            "active_ratio": round(self.active_ratio, 4),
        }
