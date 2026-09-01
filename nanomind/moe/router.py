"""
nanomind/moe/router.py — Top-K sparse router for Mixture of Experts.

The router is a simple linear layer that maps each token to N expert logits.
The top-K experts by logit are selected; only those K experts process the token.
The routing weights (softmax over top-K logits) are used to blend expert outputs.

Routing steps:
  1. Compute router logits: (B×T, N)
  2. Take top-K experts per token
  3. Compute softmax weights over top-K logits only
  4. Return indices and weights for the MoE forward pass
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Linear top-K router for Mixture of Experts.

    Args:
        d_model:     Input token embedding dimension.
        num_experts: Total number of experts (N).
        top_k:       Number of experts each token is routed to (K).
    """

    def __init__(
        self,
        d_model:     int,
        num_experts: int,
        top_k:       int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-K routing assignments.

        Args:
            x: Token embeddings ``(B, T, d_model)`` or ``(tokens, d_model)``.

        Returns:
            Tuple of:
            - ``expert_indices`` : Top-K expert IDs per token ``(tokens, K)``
            - ``expert_weights`` : Softmax routing weights ``(tokens, K)``
            - ``router_logits``  : Raw router logits ``(tokens, N)`` for aux loss
        """
        shape    = x.shape
        x_flat   = x.reshape(-1, shape[-1])       # (tokens, d_model)
        logits   = self.gate(x_flat)               # (tokens, N)

        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        weights  = F.softmax(top_logits, dim=-1)   # (tokens, K)

        return top_indices, weights, logits

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, top_k={self.top_k}"
