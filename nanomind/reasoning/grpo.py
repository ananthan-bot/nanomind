"""
nanomind/reasoning/grpo.py — Group Relative Policy Optimization (DeepSeek-R1 core RL engine).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from nanomind.reasoning.config import GRPOConfig


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes group normalized relative advantage:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    rewards: (B, G) or (G,) where G is group size per prompt.
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)  # (1, G)

    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    return advantages.squeeze(0) if advantages.shape[0] == 1 else advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_weight: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Group Relative Policy Optimization objective (DeepSeek-R1):
    L_GRPO = - 1/G sum_{i=1}^G [ min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i) - beta * D_KL(pi_theta || pi_ref) ]

    log_probs: (G, T) log probs under current policy pi_theta
    old_log_probs: (G, T) log probs under rollout policy pi_old
    ref_log_probs: (G, T) log probs under frozen reference model pi_ref
    advantages: (G,) scalar relative group advantages
    mask: (G, T) binary token completion mask (1 for generated tokens, 0 for prompt/pad)
    """
    G, T = log_probs.shape

    # Expand advantages to token shape: (G, 1) -> (G, T)
    adv = advantages.unsqueeze(-1)

    # Importance sampling ratio: pi_theta / pi_old
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2)

    # Unbiased Schulman estimator of KL divergence: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    # or forward KL: log_pi - log_ref
    kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0

    total_per_token = policy_loss + kl_weight * kl

    if mask is not None:
        valid_tokens = mask.sum().clamp(min=1.0)
        total_loss = (total_per_token * mask).sum() / valid_tokens
        p_loss_scalar = (policy_loss * mask).sum() / valid_tokens
        kl_scalar = (kl * mask).sum() / valid_tokens
    else:
        total_loss = total_per_token.mean()
        p_loss_scalar = policy_loss.mean()
        kl_scalar = kl.mean()

    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(p_loss_scalar.item()),
        "kl_divergence": float(kl_scalar.item()),
        "mean_ratio": float(ratio.mean().item()),
    }

    return total_loss, metrics


class GRPOTrainer:
    """
    Self-contained GRPO optimization step manager.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()

    def step(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Runs one step of GRPO advantage normalization and loss computation.
        """
        advantages = compute_group_advantages(rewards, eps=self.config.eps)
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            mask=mask,
            clip_eps=self.config.clip_eps,
            kl_weight=self.config.kl_weight,
        )
        return loss, metrics
