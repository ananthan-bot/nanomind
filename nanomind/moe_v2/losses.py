"""
nanomind/moe_v2/losses.py — MoE auxiliary losses.

## MoE Auxiliary Losses

MoE models suffer from routing collapse:
  All tokens route to the same 1-2 experts → others starve.
  Causes: (a) rich-get-richer, (b) gradient feedback loops.

Auxiliary losses penalise routing imbalance:

1. Load Balance Loss (Switch Transformer):
   L_lb = n_experts × Σ_i f_i × P_i
   where f_i = fraction of tokens to expert i (argmax-based)
         P_i = mean router probability for expert i

2. Z-Loss (ST-MoE, Zoph et al., 2022):
   L_z = (1/B) × Σ_b (log Σ_e exp(x_{b,e}))²
   Penalises large router logits → stabilises training.

3. Router Z-Loss (ST-MoE-32B):
   Combines load balance + z-loss for best results.

4. Entropy Regularisation:
   Maximise H(router distribution) → encourages exploration.

Reference:
  Zoph et al. (2022) "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
  https://arxiv.org/abs/2202.08906
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def load_balance_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    Switch Transformer load balance auxiliary loss.

    Args:
        router_probs:   ``(N, E)`` router softmax probabilities.
        expert_indices: ``(N, K)`` top-K expert indices per token.
        n_experts:      Number of experts E.

    Returns:
        Scalar loss.
    """
    N, E    = router_probs.shape
    # f_i: fraction of tokens assigned to each expert (top-1)
    one_hot = torch.zeros(N, E)
    one_hot.scatter_(1, expert_indices[:, :1], 1.0)
    f_i     = one_hot.mean(dim=0)        # (E,)
    P_i     = router_probs.mean(dim=0)   # (E,)
    return E * (f_i * P_i).sum()


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Z-Loss (Zoph et al., 2022): penalise large logit magnitudes.

    L_z = (1/N) Σ_n (log Σ_e exp(z_{n,e}))²

    Args:
        router_logits: ``(N, E)`` pre-softmax router logits.

    Returns:
        Scalar z-loss.
    """
    log_z   = torch.logsumexp(router_logits, dim=-1)   # (N,)
    return (log_z ** 2).mean()


def entropy_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """
    Negative entropy regularisation (encourages diverse routing).

    L_ent = -H(P) = Σ P log P   (minimise → maximise entropy)

    Args:
        router_probs: ``(N, E)`` router probabilities.

    Returns:
        Negative mean entropy (scalar).
    """
    ent = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1)
    return -ent.mean()


def combined_moe_loss(
    router_probs:   torch.Tensor,
    router_logits:  torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts:      int,
    lb_coef:        float = 1e-2,
    z_coef:         float = 1e-3,
) -> torch.Tensor:
    """
    Combined MoE auxiliary loss: load balance + Z-loss.

    Args:
        router_probs:   ``(N, E)`` router probabilities.
        router_logits:  ``(N, E)`` pre-softmax logits.
        expert_indices: ``(N, K)`` top-K indices.
        n_experts:      Number of experts.
        lb_coef:        Load balance coefficient.
        z_coef:         Z-loss coefficient.

    Returns:
        Scalar combined loss.
    """
    lb = lb_coef * load_balance_loss(router_probs, expert_indices, n_experts)
    z  = z_coef  * z_loss(router_logits)
    return lb + z
