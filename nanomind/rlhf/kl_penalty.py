"""
nanomind/rlhf/kl_penalty.py — KL divergence penalty for RLHF PPO.

A critical component of RLHF is keeping the policy model close to the
SFT reference model to prevent reward hacking and maintain language quality.

The KL penalty adds a per-token penalty to the reward:
  r_total(t) = r_RM(t) - β · KL(π_θ(t) || π_ref(t))

where β is the KL coefficient and:
  KL(π_θ || π_ref) = Σ_v π_θ(v|x) · log(π_θ(v|x) / π_ref(v|x))

Approximation used in practice (for efficiency):
  KL ≈ log π_θ(a_t|s_t) - log π_ref(a_t|s_t)
  (sample-based estimate using the generated action a_t)

Adaptive KL controller (Ziegler et al. 2019):
  If KL > target:   increase β  (tighten constraint)
  If KL < target:   decrease β  (relax constraint)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kl_divergence(
    logits_policy:    torch.Tensor,
    logits_reference: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token KL divergence: KL(policy || reference).

    Args:
        logits_policy:    Policy logits ``(B, T, V)``.
        logits_reference: Reference model logits ``(B, T, V)``.

    Returns:
        Per-token KL ``(B, T)`` — non-negative.
    """
    log_p   = F.log_softmax(logits_policy,    dim=-1)
    log_q   = F.log_softmax(logits_reference, dim=-1)
    p       = log_p.exp()
    # KL(p || q) = Σ p * (log p - log q)
    kl      = (p * (log_p - log_q)).sum(dim=-1)
    return kl.clamp(min=0.0)   # numerical safety


def approx_token_kl(
    log_probs_policy:    torch.Tensor,
    log_probs_reference: torch.Tensor,
) -> torch.Tensor:
    """
    Sample-based approximate KL: log π_θ(a_t) - log π_ref(a_t).

    This is the cheap approximation used in TRL / InstructGPT:
    instead of summing over the full vocabulary, just use the
    log-prob of the actually sampled token.

    Args:
        log_probs_policy:    ``(B, T)`` log-probs of sampled tokens under policy.
        log_probs_reference: ``(B, T)`` log-probs of sampled tokens under reference.

    Returns:
        Approximate KL ``(B, T)``.
    """
    return log_probs_policy - log_probs_reference


class AdaptiveKLController:
    """
    Adaptive KL coefficient controller (Ziegler et al. 2019).

    Adjusts the KL penalty coefficient β based on the observed KL divergence
    relative to a target:
      β_new = β * (1 + 0.2 * (KL - target) / target)

    Args:
        init_kl_coef: Initial β value.
        target_kl:    Target KL divergence.
        horizon:      Adaptation horizon (number of steps).
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl:    float = 6.0,
        horizon:      int   = 10_000,
    ) -> None:
        self.kl_coef  = init_kl_coef
        self.target   = target_kl
        self.horizon  = horizon

    def update(self, current_kl: float) -> float:
        """
        Update β based on observed KL and return the new value.

        Args:
            current_kl: Observed mean KL divergence in the last batch.

        Returns:
            Updated β.
        """
        proportional_error = (current_kl - self.target) / self.target
        multiplier         = 1 + 0.2 * proportional_error
        self.kl_coef       = self.kl_coef * multiplier
        self.kl_coef       = max(0.01, min(self.kl_coef, 10.0))   # clamp
        return self.kl_coef
