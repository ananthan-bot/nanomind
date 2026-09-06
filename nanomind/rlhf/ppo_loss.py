"""
nanomind/rlhf/ppo_loss.py — PPO clipped objective and value function loss.

PPO-Clip (Schulman et al. 2017) objective per token:
  ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)  = exp(log_π - log_π_old)
  L_clip = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)

The clipping prevents too-large policy updates:
  - If ratio > 1+ε: policy moved too far → cap the gradient
  - If ratio < 1-ε: policy moved back too much → cap the gradient

Value function loss (Huber loss for stability):
  L_V = 0.5 · (V_θ(s_t) - returns_t)²
  (sometimes also clipped, but we use simple MSE here)

Entropy bonus (optional, encourages exploration):
  L_H = H(π_θ(·|s_t)) = -Σ π_θ(v) log π_θ(v)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_policy_loss(
    log_probs:     torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    clip_ratio:    float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    PPO-Clip policy loss.

    Args:
        log_probs:     Current policy log-probs ``(N,)``.
        old_log_probs: Old policy log-probs ``(N,)`` (from rollout).
        advantages:    GAE advantages ``(N,)`` (pre-normalised).
        clip_ratio:    PPO clip epsilon ε.

    Returns:
        Tuple of ``(loss, info)`` where ``info`` contains clip fraction and ratio stats.
    """
    ratio      = torch.exp(log_probs - old_log_probs)
    surr1      = ratio * advantages
    surr2      = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss       = -torch.min(surr1, surr2).mean()
    clipped    = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()

    return loss, {
        "policy_loss":    loss.item(),
        "clip_fraction":  clipped,
        "ratio_mean":     ratio.mean().item(),
        "ratio_max":      ratio.max().item(),
    }


def ppo_value_loss(
    values:  torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Value function MSE loss.

    Args:
        values:  Current value estimates ``(N,)``.
        returns: GAE returns (value targets) ``(N,)``.

    Returns:
        Scalar value loss.
    """
    return F.mse_loss(values, returns)


def ppo_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of the policy distribution (encourages exploration).

    Args:
        logits: Policy logits ``(N, vocab_size)`` or ``(vocab_size,)``.

    Returns:
        Scalar entropy (positive — we maximise entropy).
    """
    probs  = F.softmax(logits, dim=-1)
    log_p  = F.log_softmax(logits, dim=-1)
    return -(probs * log_p).sum(dim=-1).mean()


def ppo_total_loss(
    log_probs:     torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    values:        torch.Tensor,
    returns:       torch.Tensor,
    logits:        torch.Tensor | None = None,
    clip_ratio:    float = 0.2,
    value_coef:    float = 0.1,
    entropy_coef:  float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Combined PPO loss: policy + value + entropy.

    Returns:
        Tuple of ``(total_loss, info_dict)``.
    """
    policy_loss, policy_info = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_ratio)
    val_loss                 = ppo_value_loss(values, returns)
    entropy                  = ppo_entropy_bonus(logits) if logits is not None else torch.tensor(0.0)

    total = policy_loss + value_coef * val_loss - entropy_coef * entropy
    return total, {
        **policy_info,
        "value_loss":   val_loss.item(),
        "entropy":      entropy.item(),
        "total_loss":   total.item(),
    }
