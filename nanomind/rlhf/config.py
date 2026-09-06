"""
nanomind/rlhf/config.py — RLHF configuration (Reward Model + PPO).

## RLHF Pipeline (Ouyang et al. 2022 — InstructGPT)

Step 1 — Supervised Fine-Tuning (SFT):
  Fine-tune the base LM on high-quality human demonstrations.
  (This uses the standard NanoMind Trainer — already implemented)

Step 2 — Reward Model (RM) Training:
  Given pairs of completions (chosen, rejected) for the same prompt,
  train a scalar-output model to score completions by preference.
  Loss: Bradley-Terry pairwise ranking loss
    L = -log σ(r_θ(x, y_w) - r_θ(x, y_l))
  where y_w = preferred (won), y_l = rejected (lost).

Step 3 — PPO Fine-Tuning:
  Use the Reward Model as an environment reward signal.
  Train the SFT model (policy) with PPO to maximise expected reward
  while staying close to the SFT reference (KL penalty):
    objective = E[r_θ(x,y)] - β · KL(π_θ || π_ref)
  PPO clip: prevents large policy updates that destabilise training.

References:
  InstructGPT: Ouyang et al. (2022) https://arxiv.org/abs/2203.02155
  PPO:         Schulman et al. (2017) https://arxiv.org/abs/1707.06347
  RL4LMs:      Ramamurthy et al. (2022) https://arxiv.org/abs/2210.01241
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RewardModelConfig:
    """
    Configuration for the Reward Model.

    Attributes:
        dropout:      Dropout on reward head.
        pooling:      How to pool sequence → scalar
                      (``"last"`` = last token, ``"mean"`` = average).
    """
    dropout: float = 0.1
    pooling: str   = "last"

    def __post_init__(self) -> None:
        assert self.pooling in ("last", "mean")
        assert 0.0 <= self.dropout <= 1.0


@dataclass
class PPOConfig:
    """
    Configuration for PPO fine-tuning.

    Attributes:
        clip_ratio:     PPO clip epsilon (0.2 is standard).
        value_coef:     Weight of value function loss.
        entropy_coef:   Weight of entropy bonus (exploration).
        kl_coef:        KL penalty coefficient β (distance from reference).
        kl_target:      Target KL divergence (adaptive controller).
        gamma:          Discount factor.
        lam:            GAE lambda for advantage estimation.
        ppo_epochs:     Number of PPO update epochs per rollout.
        rollout_len:    Token length of generated rollouts.
        mini_batch_size: Mini-batch size for PPO updates.
    """
    clip_ratio:      float = 0.2
    value_coef:      float = 0.1
    entropy_coef:    float = 0.01
    kl_coef:         float = 0.1
    kl_target:       float = 6.0
    gamma:           float = 1.0
    lam:             float = 0.95
    ppo_epochs:      int   = 4
    rollout_len:     int   = 64
    mini_batch_size: int   = 4

    def __post_init__(self) -> None:
        assert 0.0 < self.clip_ratio < 1.0
        assert 0.0 <= self.kl_coef
        assert 0.0 < self.gamma <= 1.0
        assert 0.0 <= self.lam  <= 1.0
        assert self.ppo_epochs >= 1
