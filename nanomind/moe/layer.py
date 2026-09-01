"""
nanomind/moe/layer.py — Sparse Mixture of Experts FFN layer.

Replaces the standard dense FFN in a transformer block with N independent
experts and a top-K router. Each token is processed by only K of the N
experts, with outputs blended by the routing weights.

Forward algorithm:
  1. Router: assign each token to its top-K experts and compute weights
  2. For each expert: gather assigned tokens, run FFN, scatter results back
  3. Blend: weighted sum of expert outputs per token
  4. Optionally compute auxiliary load-balancing loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.moe.config import MoEConfig
from nanomind.moe.expert import Expert
from nanomind.moe.router import TopKRouter
from nanomind.moe.load_balance import load_balance_loss


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts FFN layer.

    Replaces the dense FFN in a transformer block. Each token is routed to
    its top-K experts; outputs are blended by routing weights.

    Args:
        d_model:    Model embedding dimension.
        cfg:        MoE configuration.
        bias:       Whether expert projections have bias.
    """

    def __init__(
        self,
        d_model: int,
        cfg:     MoEConfig,
        bias:    bool = False,
    ) -> None:
        super().__init__()
        self.d_model     = d_model
        self.cfg         = cfg
        d_ff             = cfg.d_ff_expert or (4 * d_model)

        self.router  = TopKRouter(d_model, cfg.num_experts, cfg.top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, cfg.activation, bias)
            for _ in range(cfg.num_experts)
        ])
        self.dropout = nn.Dropout(cfg.expert_dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse MoE forward pass.

        Args:
            x: Input ``(B, T, d_model)``.

        Returns:
            Tuple of:
            - Output tensor ``(B, T, d_model)`` — weighted sum of expert outputs
            - Auxiliary load-balancing loss (scalar tensor)
        """
        B, T, D = x.shape
        x_flat  = x.reshape(B * T, D)             # (tokens, D)

        # Router
        expert_indices, expert_weights, router_logits = self.router(x_flat)
        # expert_indices: (tokens, K), expert_weights: (tokens, K)

        output = torch.zeros_like(x_flat)

        # Process each expert: gather assigned tokens, run FFN, scatter back
        for expert_id, expert in enumerate(self.experts):
            # Boolean mask: which (token, k) slots chose this expert
            mask = (expert_indices == expert_id)   # (tokens, K)

            # token positions that use this expert (may be used in multiple k slots)
            token_mask = mask.any(dim=1)           # (tokens,)
            if not token_mask.any():
                continue

            selected = x_flat[token_mask]          # (selected, D)
            expert_out = expert(selected)          # (selected, D)
            expert_out = self.dropout(expert_out)

            # Weight: sum of weights across k slots for this expert
            w = (expert_weights * mask.float()).sum(dim=1)   # (tokens,)
            output[token_mask] += expert_out * w[token_mask].unsqueeze(1)

        output = output.reshape(B, T, D)

        # Auxiliary load-balancing loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.cfg.load_balance_coef > 0.0:
            aux_loss = self.cfg.load_balance_coef * load_balance_loss(
                router_logits, expert_indices, self.cfg.num_experts
            )

        return output, aux_loss

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.cfg.num_experts}, "
            f"top_k={self.cfg.top_k}, "
            f"d_model={self.d_model}"
        )
