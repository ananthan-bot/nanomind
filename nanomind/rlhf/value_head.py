"""
nanomind/rlhf/value_head.py — Value function head for PPO actor-critic.

In PPO, the policy (actor) and value function (critic) often share a backbone.
The ValueHead adds a linear layer on top of the backbone hidden states to
predict V(s) — the expected cumulative reward from state s.

V(s) is used to compute advantages:
  A(s, a) = Q(s, a) - V(s)   (advantage of taking action a in state s)

Generalised Advantage Estimation (GAE, Schulman 2016):
  δ_t     = r_t + γ · V(s_{t+1}) - V(s_t)
  A_t     = Σ_{l≥0} (γλ)^l · δ_{t+l}

Advantages normalised to zero mean and unit variance before the PPO update.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Scalar value function head added on top of a transformer backbone.

    Maps hidden states ``(B, T, d_model)`` → value estimates ``(B, T)``.

    Args:
        d_model: Backbone hidden dimension.
        dropout: Dropout before the linear layer.

    Example::

        value_head = ValueHead(d_model=256)
        values     = value_head(hidden_states)   # (B, T)
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token value estimates.

        Args:
            hidden: ``(B, T, d_model)`` hidden states from backbone.

        Returns:
            Value estimates ``(B, T)``.
        """
        return self.proj(self.drop(hidden)).squeeze(-1)


def compute_gae(
    rewards:  torch.Tensor,
    values:   torch.Tensor,
    gamma:    float = 1.0,
    lam:      float = 0.95,
    last_val: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalised Advantage Estimation (GAE).

    Args:
        rewards: ``(T,)`` per-step rewards.
        values:  ``(T,)`` value estimates per step.
        gamma:   Discount factor.
        lam:     GAE lambda.
        last_val: Bootstrap value for the step after the rollout.

    Returns:
        Tuple of ``(advantages, returns)`` each ``(T,)``
        where returns = advantages + values (used as value targets).
    """
    T           = rewards.shape[0]
    advantages  = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae         = 0.0

    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else values[t + 1].item()
        delta    = rewards[t].item() + gamma * next_val - values[t].item()
        gae      = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
