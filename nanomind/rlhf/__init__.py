"""NanoMind RLHF sub-package — Reward Model + PPO Fine-Tuning.

Implements the InstructGPT 3-step RLHF pipeline:
  Step 1: SFT      — supervised fine-tuning (use NanoMind Trainer)
  Step 2: RM       — reward model training on preference pairs
  Step 3: PPO      — policy optimization with RM reward + KL penalty

Primary exports:
    - :class:`RewardModel`           — backbone + scalar head for preference scoring
    - :class:`RewardModelTrainer`    — train_step() + train_epoch() on preference data
    - :class:`PreferenceDataset`     — (prompt, chosen, rejected) dataset
    - :class:`RewardModelConfig`     — pooling, dropout config
    - :class:`PPOConfig`             — clip_ratio, kl_coef, GAE lambda, ppo_epochs
    - :class:`ValueHead`             — per-token value estimates for actor-critic
    - :class:`PPORolloutBuffer`      — collect rollouts, compute GAE advantages
    - :class:`AdaptiveKLController`  — adaptive β for KL penalty
    - :func:`preference_loss`        — Bradley-Terry pairwise ranking loss
    - :func:`preference_accuracy`    — fraction of pairs ranked correctly
    - :func:`reward_stats`           — mean chosen/rejected/margin/accuracy
    - :func:`token_kl_divergence`    — per-token KL(policy || reference)
    - :func:`approx_token_kl`        — sample-based approximate KL
    - :func:`compute_gae`            — Generalised Advantage Estimation
    - :func:`ppo_total_loss`         — policy + value + entropy PPO loss
"""

from nanomind.rlhf.config import RewardModelConfig, PPOConfig
from nanomind.rlhf.reward_model import RewardModel
from nanomind.rlhf.preference_loss import preference_loss, preference_accuracy, reward_stats
from nanomind.rlhf.preference_dataset import PreferenceDataset
from nanomind.rlhf.value_head import ValueHead, compute_gae
from nanomind.rlhf.kl_penalty import (
    token_kl_divergence, approx_token_kl, AdaptiveKLController
)
from nanomind.rlhf.rollout import PPORolloutBuffer, RolloutBatch
from nanomind.rlhf.ppo_loss import (
    ppo_policy_loss, ppo_value_loss, ppo_entropy_bonus, ppo_total_loss
)
from nanomind.rlhf.rm_trainer import RewardModelTrainer

__all__ = [
    "RewardModelConfig", "PPOConfig",
    "RewardModel", "RewardModelTrainer",
    "PreferenceDataset",
    "ValueHead", "compute_gae",
    "token_kl_divergence", "approx_token_kl", "AdaptiveKLController",
    "PPORolloutBuffer", "RolloutBatch",
    "preference_loss", "preference_accuracy", "reward_stats",
    "ppo_policy_loss", "ppo_value_loss", "ppo_entropy_bonus", "ppo_total_loss",
]
