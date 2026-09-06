"""
nanomind/rlhf/preference_loss.py — Pairwise preference losses for reward model training.

## Bradley-Terry Model
The Bradley-Terry model (1952) gives the probability that item A is preferred over B:
  P(A ≻ B) = σ(r(A) - r(B)) = exp(r(A)) / (exp(r(A)) + exp(r(B)))

Maximising the log-likelihood gives the pairwise ranking loss:
  L = -log P(y_w ≻ y_l) = -log σ(r_θ(y_w) - r_θ(y_l))

This is equivalent to binary cross-entropy on the score difference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def preference_loss(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
    margin:          float = 0.0,
) -> torch.Tensor:
    """
    Bradley-Terry pairwise preference loss.

    Trains the reward model to assign higher scores to chosen completions.

    Args:
        reward_chosen:   Scalar rewards for preferred completions ``(B,)``.
        reward_rejected: Scalar rewards for rejected completions ``(B,)``.
        margin:          Optional margin: require r_w - r_l > margin.

    Returns:
        Scalar loss (mean over batch).

    Example::

        r_w  = reward_model(chosen_ids)
        r_l  = reward_model(rejected_ids)
        loss = preference_loss(r_w, r_l)
        loss.backward()
    """
    diff = reward_chosen - reward_rejected - margin
    return -F.logsigmoid(diff).mean()


def preference_accuracy(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
) -> float:
    """
    Compute the fraction of pairs where chosen reward > rejected reward.

    Args:
        reward_chosen:   ``(B,)`` rewards for preferred completions.
        reward_rejected: ``(B,)`` rewards for rejected completions.

    Returns:
        Accuracy in [0, 1] — 1.0 means perfect ranking.
    """
    return (reward_chosen > reward_rejected).float().mean().item()


def reward_stats(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
) -> dict:
    """
    Compute summary statistics for reward model monitoring.

    Args:
        reward_chosen:   ``(B,)`` rewards for preferred completions.
        reward_rejected: ``(B,)`` rewards for rejected completions.

    Returns:
        Dict with ``mean_chosen``, ``mean_rejected``, ``mean_margin``, ``accuracy``.
    """
    return {
        "mean_chosen":   reward_chosen.mean().item(),
        "mean_rejected": reward_rejected.mean().item(),
        "mean_margin":   (reward_chosen - reward_rejected).mean().item(),
        "accuracy":      preference_accuracy(reward_chosen, reward_rejected),
    }
