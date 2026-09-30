"""
nanomind/alignment/kto.py — KTO: Kahneman-Tversky Optimization.

## KTO (Ethayarajh et al., 2024)

DPO needs paired (chosen, rejected) data for the SAME prompt.
This is often unavailable — real human feedback is:
  "This response was good" / "This response was bad"
without a paired comparison.

KTO uses prospect theory (Kahneman & Tversky, 1979):
  Humans are loss-averse: losing $100 feels worse than gaining $100!

KTO loss:
  For desirable responses (y_w):
    L_w = 1 - σ(r(x, y_w) - z₀)    where z₀ = KL[π || π_ref]

  For undesirable responses (y_l):
    L_l = 1 - σ(z₀ - r(x, y_l))

The z₀ reference point is the expected reward under the reference model.

Benefits:
  ✓ Works with binary feedback (good/bad), not pairs
  ✓ More natural data collection (rate responses, don't compare)
  ✓ Better data efficiency (each sample is useful independently)

Reference:
  Ethayarajh et al. (2024) "KTO: Model Alignment as Prospect Theoretic Optimization"
  https://arxiv.org/abs/2402.01306
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanomind.alignment.dpo import compute_log_probs


@dataclass
class KTOConfig:
    """Configuration for KTO training."""
    beta:        float = 0.1    # KL weight
    desirable_weight:   float = 1.0   # weight for desirable samples
    undesirable_weight: float = 1.0   # weight for undesirable samples


def kto_loss(
    policy_logp:  torch.Tensor,
    ref_logp:     torch.Tensor,
    is_desirable: torch.Tensor,   # (B,) bool
    kl_estimate:  torch.Tensor,   # scalar KL estimate
    cfg:          KTOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Compute KTO loss from log probabilities and binary labels.

    Args:
        policy_logp:  ``(B,)`` policy log probs for each response.
        ref_logp:     ``(B,)`` reference log probs.
        is_desirable: ``(B,)`` bool — True = desirable, False = undesirable.
        kl_estimate:  Scalar KL divergence estimate (z₀ reference point).
        cfg:          :class:`KTOConfig`.

    Returns:
        ``(loss, metrics_dict)``
    """
    β  = cfg.beta
    z0 = kl_estimate.detach()

    rewards = β * (policy_logp - ref_logp)   # (B,)

    # Desirable loss: 1 - σ(r - z₀)
    loss_desirable   = cfg.desirable_weight * (
        1 - F.sigmoid(rewards[is_desirable] - z0)
    )
    # Undesirable loss: 1 - σ(z₀ - r)
    loss_undesirable = cfg.undesirable_weight * (
        1 - F.sigmoid(z0 - rewards[~is_desirable])
    )

    # Combine (mean over non-empty subsets)
    total = torch.zeros(1)
    if loss_desirable.numel() > 0:
        total = total + loss_desirable.mean()
    if loss_undesirable.numel() > 0:
        total = total + loss_undesirable.mean()

    metrics = {
        "loss":             total.item(),
        "reward_desirable":   rewards[is_desirable].mean().item() if is_desirable.any() else 0.0,
        "reward_undesirable": rewards[~is_desirable].mean().item() if (~is_desirable).any() else 0.0,
        "kl_estimate":        z0.item(),
    }
    return total, metrics


class KTOTrainer:
    """
    KTO training loop.

    Args:
        policy_model:  LM to train.
        ref_model:     Frozen reference LM.
        optimizer:     Policy optimizer.
        cfg:           :class:`KTOConfig`.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model:    nn.Module,
        optimizer:    torch.optim.Optimizer,
        cfg:          KTOConfig | None = None,
    ) -> None:
        self.policy = policy_model
        self.ref    = ref_model
        self.opt    = optimizer
        self.cfg    = cfg or KTOConfig()
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def estimate_kl(self, ids: torch.Tensor) -> torch.Tensor:
        """Estimate E[log π/π_ref] on a batch for the z₀ reference."""
        with torch.no_grad():
            pol = compute_log_probs(self.policy, ids, ids)
            ref = compute_log_probs(self.ref,    ids, ids)
        return (pol - ref).mean()

    def step(
        self,
        ids:          torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> dict:
        """
        One KTO training step.

        Args:
            ids:          ``(B, T)`` response token IDs.
            is_desirable: ``(B,)`` bool labels.
        """
        self.opt.zero_grad()
        kl_est = self.estimate_kl(ids)

        policy_logp = compute_log_probs(self.policy, ids, ids)
        with torch.no_grad():
            ref_logp = compute_log_probs(self.ref, ids, ids)

        loss, metrics = kto_loss(policy_logp, ref_logp, is_desirable, kl_est, self.cfg)
        loss.backward()
        self.opt.step()
        return metrics
