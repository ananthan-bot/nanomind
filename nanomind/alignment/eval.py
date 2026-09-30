"""
nanomind/alignment/eval.py — Alignment evaluation metrics.

Key alignment metrics:
  - Win rate: fraction of time aligned model beats baseline
  - Reward score: mean RM score on test prompts
  - KL divergence: how much policy drifted from reference
  - Reward hacking detection: high reward but low quality

## Goodhart's Law and Reward Hacking

"When a measure becomes a target, it ceases to be a good measure."

In RLHF: models learn to exploit reward model weaknesses:
  - Generating very long responses (RM rewards verbosity)
  - Sycophantic responses ("Great question!")
  - Repetitive token patterns that confuse RM

Mitigation:
  - KL penalty (keeps policy close to reference)
  - Diverse RM ensembles (harder to hack all)
  - Human evaluation checkpoints
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AlignmentMetrics:
    """Collected alignment evaluation metrics."""
    win_rate:    float
    mean_reward: float
    kl_div:      float
    reward_std:  float

    def to_dict(self) -> dict:
        return {
            "win_rate":    round(self.win_rate,    3),
            "mean_reward": round(self.mean_reward, 4),
            "kl_div":      round(self.kl_div,      4),
            "reward_std":  round(self.reward_std,  4),
        }

    def is_reward_hacking(self, threshold: float = 2.0) -> bool:
        """Heuristic: high reward + high KL = suspicious."""
        return self.mean_reward > threshold and self.kl_div > threshold


class AlignmentEvaluator:
    """
    Evaluate alignment quality of a trained model.

    Args:
        reward_model:  Trained reward model for scoring.
        ref_model:     Reference (SFT) model for KL estimation.

    Example::

        eval    = AlignmentEvaluator(reward_model, sft_model)
        metrics = eval.evaluate(policy_model, test_ids)
        print(metrics.to_dict())
    """

    def __init__(
        self,
        reward_model: nn.Module,
        ref_model:    nn.Module,
    ) -> None:
        self.rm  = reward_model
        self.ref = ref_model

    @torch.no_grad()
    def compute_rewards(self, model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        """Score responses with reward model."""
        return self.rm(ids)   # (B,)

    @torch.no_grad()
    def compute_kl(
        self,
        policy:    nn.Module,
        ids:       torch.Tensor,
    ) -> float:
        """Estimate KL(policy || ref) on a batch."""
        def log_probs(m, x):
            out    = m(x)
            logits = out[0] if isinstance(out, tuple) else out
            return F.log_softmax(logits[:, :-1], dim=-1)

        policy_lp = log_probs(policy, ids)
        ref_lp    = log_probs(self.ref, ids)
        kl = (torch.exp(policy_lp) * (policy_lp - ref_lp)).sum(-1).mean()
        return kl.item()

    @torch.no_grad()
    def win_rate(
        self,
        policy:    nn.Module,
        baseline:  nn.Module,
        ids:       torch.Tensor,
    ) -> float:
        """Fraction of samples where policy gets higher reward than baseline."""
        r_policy   = self.compute_rewards(policy,   ids)
        r_baseline = self.compute_rewards(baseline, ids)
        return (r_policy > r_baseline).float().mean().item()

    @torch.no_grad()
    def evaluate(
        self,
        policy:   nn.Module,
        baseline: nn.Module,
        test_ids: torch.Tensor,
    ) -> AlignmentMetrics:
        """
        Full alignment evaluation.

        Args:
            policy:   Aligned model.
            baseline: Reference/SFT model.
            test_ids: ``(B, T)`` test prompt+response token IDs.

        Returns:
            :class:`AlignmentMetrics`.
        """
        rewards  = self.compute_rewards(policy, test_ids)
        kl       = self.compute_kl(policy, test_ids)
        wr       = self.win_rate(policy, baseline, test_ids)

        return AlignmentMetrics(
            win_rate    = wr,
            mean_reward = rewards.mean().item(),
            kl_div      = kl,
            reward_std  = rewards.std().item(),
        )
