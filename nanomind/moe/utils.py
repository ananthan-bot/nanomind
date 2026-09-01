"""
nanomind/moe/utils.py — MoE utility functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from nanomind.moe.layer import SparseMoELayer
from nanomind.moe.load_balance import expert_utilization


def get_all_router_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, dict]:
    """
    Run a forward pass and collect expert utilization stats from all MoE layers.

    Args:
        model:     NanoMindMoE model.
        input_ids: Token IDs ``(B, T)``.

    Returns:
        Dict mapping layer name to utilization stats dict.
    """
    stats:   dict[str, dict] = {}
    handles: list            = []

    def _make_hook(name: str):
        def hook(module, inp, out):
            # out is (output, aux_loss); we need router's expert_indices
            x_flat  = inp[0].reshape(-1, inp[0].shape[-1])
            indices, _, _ = module.router(x_flat)
            stats[name] = expert_utilization(indices, module.cfg.num_experts)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, SparseMoELayer):
            handles.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return stats


def print_moe_utilization(stats: dict[str, dict]) -> None:
    """Pretty-print expert utilization across all MoE layers."""
    print("=" * 60)
    print("Expert Utilization Report")
    print("=" * 60)
    for layer_name, s in stats.items():
        print(f"
  Layer: {layer_name}")
        print(f"    Used experts    : {s['utilization']:.0%}")
        print(f"    Min token frac  : {s['min_frac']:.3f}")
        print(f"    Max token frac  : {s['max_frac']:.3f}")
        fracs = [f"{f:.2f}" for f in s["fractions"]]
        print(f"    Per-expert frac : [{', '.join(fracs)}]")
    print("=" * 60)
