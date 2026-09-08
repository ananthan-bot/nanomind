"""
nanomind/dpo/loss.py — DPO and IPO loss functions.

DPO loss (Rafailov et al. 2023):
  L = -log σ(β · (log_ratio_w - log_ratio_l))
  where log_ratio = log π_θ(y|x) - log π_ref(y|x)

IPO loss (Azar et al. 2023, "A General Theoretical Paradigm"):
  L = (log_ratio_w - log_ratio_l - 1/(2β))²
  A regression variant that avoids the sigmoid saturation issue.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_log_probs(
    logits:    torch.Tensor,
    token_ids: torch.Tensor,
    mask:      torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for the given token IDs.

    Args:
        logits:    Model logits ``(B, T, V)``.
        token_ids: Target token IDs ``(B, T)``.
        mask:      Boolean mask ``(B, T)`` — True where tokens count.

    Returns:
        Sum of log-probs per sequence ``(B,)``.
    """
    log_probs  = F.log_softmax(logits, dim=-1)          # (B, T, V)
    token_lp   = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    if mask is not None:
        token_lp = token_lp * mask.float()
    return token_lp.sum(dim=-1)                          # (B,)


def dpo_loss(
    policy_log_ratio_chosen:   torch.Tensor,
    policy_log_ratio_rejected: torch.Tensor,
    beta:            float = 0.1,
    label_smoothing: float = 0.0,
    loss_type:       str   = "sigmoid",
) -> tuple[torch.Tensor, dict]:
    """
    DPO (or IPO) loss for a batch of preference pairs.

    Args:
        policy_log_ratio_chosen:   log π_θ(y_w|x) - log π_ref(y_w|x)  ``(B,)``
        policy_log_ratio_rejected: log π_θ(y_l|x) - log π_ref(y_l|x)  ``(B,)``
        beta:            KL penalty coefficient.
        label_smoothing: Optional label smoothing (DPO only).
        loss_type:       ``"sigmoid"`` (DPO) or ``"ipo"``.

    Returns:
        Tuple of ``(loss, info_dict)``.
    """
    pi_logratios = policy_log_ratio_chosen - policy_log_ratio_rejected
    h            = beta * pi_logratios

    if loss_type == "sigmoid":
        # DPO loss with optional label smoothing
        loss = (
            -F.logsigmoid(h) * (1 - label_smoothing)
            - F.logsigmoid(-h) * label_smoothing
        ).mean()
    elif loss_type == "ipo":
        # IPO loss — avoids saturation
        loss = ((h - 1 / (2 * beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Compute reward margins for monitoring
    chosen_rewards   = beta * policy_log_ratio_chosen.detach()
    rejected_rewards = beta * policy_log_ratio_rejected.detach()

    return loss, {
        "loss":               loss.item(),
        "chosen_rewards":     chosen_rewards.mean().item(),
        "rejected_rewards":   rejected_rewards.mean().item(),
        "reward_margin":      (chosen_rewards - rejected_rewards).mean().item(),
        "reward_accuracy":    (chosen_rewards > rejected_rewards).float().mean().item(),
        "log_ratio_diff":     pi_logratios.mean().item(),
    }


def reference_free_dpo_loss(
    policy_chosen_logps:   torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Reference-free DPO: treats reference log-probs as zero.

    Equivalent to vanilla DPO with π_ref = uniform distribution.
    Useful when you don't have a reference model.

    Args:
        policy_chosen_logps:   log π_θ(y_w|x)  ``(B,)``
        policy_rejected_logps: log π_θ(y_l|x)  ``(B,)``
        beta: Temperature.

    Returns:
        Tuple of ``(loss, info_dict)``.
    """
    return dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        beta=beta, loss_type="sigmoid",
    )
