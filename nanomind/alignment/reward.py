"""
nanomind/alignment/reward.py — Reward model for alignment training.

The reward model (RM) learns to score responses based on preference data.
Used as a signal source for PPO and as a judge for RLAIF.

Training objective (Bradley-Terry):
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
  L_RM = -log σ(r(x, y_w) - r(x, y_l))

After training, RM can score any response in [roughly] (-3, +3) range:
  r > 0: response is better than average
  r < 0: response is worse than average
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class RewardModel(nn.Module):
    """
    Reward model: LM backbone + scalar regression head.

    Takes (prompt + response) token IDs and outputs a scalar reward.

    Args:
        backbone:   LM model (used as feature extractor).
        d_model:    Backbone output dimension.
        dropout:    Dropout on reward head.

    Example::

        rm      = RewardModel(backbone, d_model=128)
        rewards = rm(chosen_ids)     # (B,) scalar rewards
        loss, m = rm.preference_loss(chosen_ids, rejected_ids)
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model:  int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar reward for each sequence.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B,)`` reward scores.
        """
        out = self.backbone(input_ids)
        # Use last token hidden state as sequence representation
        if isinstance(out, tuple):
            hidden = out[0][:, -1, :]   # (B, D)
        else:
            hidden = out[:, -1, :]
        return self.reward_head(hidden).squeeze(-1)   # (B,)

    def preference_loss(
        self,
        chosen_ids:   torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Bradley-Terry preference loss.

        Args:
            chosen_ids:   ``(B, T)`` chosen responses.
            rejected_ids: ``(B, T)`` rejected responses.

        Returns:
            ``(loss, metrics)``
        """
        r_w = self(chosen_ids)     # (B,)
        r_l = self(rejected_ids)   # (B,)
        loss     = -F.logsigmoid(r_w - r_l).mean()
        accuracy = (r_w > r_l).float().mean()
        return loss, {
            "loss":        loss.item(),
            "accuracy":    accuracy.item(),
            "reward_chosen":   r_w.mean().item(),
            "reward_rejected": r_l.mean().item(),
            "reward_margin":   (r_w - r_l).mean().item(),
        }


@dataclass
class RMTrainingConfig:
    """Configuration for reward model training."""
    lr:         float = 1e-4
    beta:       float = 0.0   # margin bonus (r_w - r_l > beta)
    center_reward: bool = True  # subtract mean reward for stability
