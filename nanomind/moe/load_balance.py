"""
nanomind/moe/load_balance.py — Load balancing auxiliary loss for MoE.

Without an auxiliary loss, MoE collapses: a few experts receive most tokens
while the rest become unused (expert collapse). The load balancing loss
encourages uniform routing across all experts.

Switch Transformer loss (Fedus et al. 2021):
    L_aux = N × Σ_i f_i × P_i

where:
  f_i = fraction of tokens routed to expert i
  P_i = mean routing probability for expert i (from softmax)
  N   = num_experts

The loss is minimised when f_i = P_i = 1/N for all i (perfectly balanced).
"""

from __future__ import annotations

import torch


def load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute the Switch Transformer load balancing auxiliary loss.

    Args:
        router_logits:  Raw router logits ``(tokens, num_experts)``.
        expert_indices: Top-K expert indices per token ``(tokens, K)``.
        num_experts:    Total number of experts N.

    Returns:
        Scalar auxiliary loss tensor.
    """
    num_tokens = router_logits.shape[0]

    # f_i: fraction of tokens dispatched to expert i
    # Count how many times each expert appears in top-K assignments
    dispatch_mask = torch.zeros(
        num_tokens, num_experts,
        device=router_logits.device, dtype=router_logits.dtype
    )
    dispatch_mask.scatter_(
        1,
        expert_indices.reshape(num_tokens, -1),
        1.0 / expert_indices.shape[1],   # share equally among K chosen experts
    )
    f_i = dispatch_mask.mean(dim=0)      # (N,) mean fraction per expert

    # P_i: mean routing probability for expert i (across all tokens)
    p_i = torch.softmax(router_logits, dim=-1).mean(dim=0)   # (N,)

    # L_aux = N × Σ f_i × P_i
    loss = num_experts * (f_i * p_i).sum()
    return loss


def expert_utilization(
    expert_indices: torch.Tensor,
    num_experts:    int,
) -> dict:
    """
    Compute expert utilization statistics.

    Args:
        expert_indices: Token → expert assignments ``(tokens, K)``.
        num_experts:    Total experts.

    Returns:
        Dict with ``counts``, ``fractions``, ``min_frac``, ``max_frac``,
        ``utilization`` (fraction of experts with >0 tokens).
    """
    flat    = expert_indices.reshape(-1)
    counts  = torch.bincount(flat, minlength=num_experts).float()
    total   = flat.numel()
    fracs   = counts / max(total, 1)
    used    = (counts > 0).sum().item()
    return {
        "counts":      counts.tolist(),
        "fractions":   fracs.tolist(),
        "min_frac":    fracs.min().item(),
        "max_frac":    fracs.max().item(),
        "utilization": used / num_experts,
    }
