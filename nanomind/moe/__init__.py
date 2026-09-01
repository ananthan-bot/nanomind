"""NanoMind Mixture of Experts (MoE) sub-package.

Replaces the dense FFN in transformer blocks with N independent experts,
routing each token to only K of them (sparse activation).

Parameters scale with N; compute scales with K — giving huge capacity
at the same inference cost as a K-expert dense model.

Primary exports:
    - :class:`NanoMindMoE`       — full transformer model with MoE FFNs
    - :class:`MoEConfig`         — num_experts, top_k, load_balance_coef
    - :class:`SparseMoELayer`    — router + N experts, weighted blend
    - :class:`TopKRouter`        — linear gate + top-K selection
    - :class:`Expert`            — single FFN expert (gelu/relu/swiglu)
    - :class:`MoETransformerBlock` — attention + MoE FFN block
    - :func:`load_balance_loss`  — Switch Transformer auxiliary loss
    - :func:`expert_utilization` — per-expert token fraction stats
    - :func:`get_all_router_stats`  — hook-based routing diagnostics
    - :func:`print_moe_utilization` — pretty-print utilization report
"""

from nanomind.moe.config import MoEConfig
from nanomind.moe.expert import Expert
from nanomind.moe.router import TopKRouter
from nanomind.moe.load_balance import load_balance_loss, expert_utilization
from nanomind.moe.layer import SparseMoELayer
from nanomind.moe.block import MoETransformerBlock
from nanomind.moe.model import NanoMindMoE
from nanomind.moe.utils import get_all_router_stats, print_moe_utilization

__all__ = [
    "MoEConfig",
    "Expert",
    "TopKRouter",
    "load_balance_loss",
    "expert_utilization",
    "SparseMoELayer",
    "MoETransformerBlock",
    "NanoMindMoE",
    "get_all_router_stats",
    "print_moe_utilization",
]
