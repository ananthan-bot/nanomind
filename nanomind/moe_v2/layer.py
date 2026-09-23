"""
nanomind/moe_v2/layer.py — Full MoE Layer: dispatch, compute, combine.

MoE forward pass:
  1. Router: assign each token to top-K experts + weights
  2. Dispatch: group tokens by expert assignment
  3. Compute: run each expert on its assigned tokens
  4. Combine: weighted sum of expert outputs

The dispatch-combine pattern:
  - Tokens → experts is a sparse scatter operation
  - Expert outputs → tokens is a sparse gather + weighted sum

This module implements the full forward pass without expert parallelism
(single-device). Expert parallelism would distribute experts across devices.

References:
  Lepikhin et al. (2021) GShard: https://arxiv.org/abs/2006.16668
  Fedus et al. (2022) Switch: https://arxiv.org/abs/2101.03961
"""

from __future__ import annotations
import torch
import torch.nn as nn

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.expert import ExpertBank
from nanomind.moe_v2.router import TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput
from nanomind.moe_v2.capacity import CapacityBuffer


class MoELayer(nn.Module):
    """
    Full Mixture of Experts layer.

    Replaces a dense FFN with N expert FFNs and learned routing.

    Args:
        cfg: :class:`MoEConfig`.

    Example::

        cfg   = MoEConfig(n_experts=8, top_k=2, d_model=256, d_ff=1024)
        layer = MoELayer(cfg)
        out, aux_loss = layer(x)   # x: (B, T, D)
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg      = cfg
        self.experts  = ExpertBank(cfg)
        self.capacity = CapacityBuffer(cfg.n_experts, cfg.capacity_factor)

        if cfg.router_type == "topk":
            self.router = TopKRouter(cfg)
        elif cfg.router_type == "expert_choice":
            self.router = ExpertChoiceRouter(cfg)
        elif cfg.router_type == "hash":
            self.router = HashRouter(cfg)
        else:
            raise ValueError(f"Unknown router_type: {cfg.router_type!r}")

        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.

        Args:
            x: ``(B, T, D)`` input tensor.

        Returns:
            ``(output, aux_loss)`` where output is ``(B, T, D)``
            and aux_loss is a scalar for load balancing.
        """
        B, T, D = x.shape
        x_flat  = x.view(B * T, D)   # (N, D)

        # Shared experts always run first
        shared_out = self.experts.shared_forward(x_flat)   # (N, D)

        # Route tokens to experts
        routing: RoutingOutput = self.router(x_flat)
        indices  = routing.indices    # (N, K)
        weights  = routing.weights    # (N, K)

        # Dispatch & combine
        output   = torch.zeros_like(x_flat)

        for k in range(self.cfg.top_k):
            expert_ids = indices[:, k]    # (N,)
            w_k        = weights[:, k]    # (N,)

            for e in range(self.cfg.n_experts):
                mask  = (expert_ids == e)
                if not mask.any():
                    continue
                x_e   = x_flat[mask]                              # (n_e, D)
                out_e = self.experts.forward_expert(e, x_e)       # (n_e, D)
                output[mask] += w_k[mask].unsqueeze(-1) * out_e

        # Add shared expert contribution
        output  = output + shared_out

        # Residual + dropout
        output  = self.dropout(output)
        output  = output.view(B, T, D)
        return output, routing.aux_loss

    def routing_stats(self, x: torch.Tensor) -> dict:
        """Return routing statistics for a batch."""
        B, T, D = x.shape
        x_flat  = x.view(B * T, D)
        with torch.no_grad():
            routing = self.router(x_flat)
        util    = self.router.expert_utilisation(routing.router_probs)                   if hasattr(self.router, "expert_utilisation") else {}
        cap_st  = self.capacity.stats(routing.indices, B * T)
        return {"routing": util, "capacity": cap_st.to_dict()}
