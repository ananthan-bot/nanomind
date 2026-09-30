"""NanoMind Alignment sub-package — Constitutional AI, DPO & Modern Alignment.

Implements the modern post-training alignment stack:
  1. PreferencePair / BinaryFeedback — data structures
  2. PreferenceDataset               — preference pairs, filter, stats
  3. DPOConfig / dpo_loss            — DPO/IPO/KTO-pair loss
  4. compute_log_probs               — per-token log probs
  5. DPOTrainer                      — policy + ref model training loop
  6. ConstitutionalAI                — critique/revise, RLAIF feedback
  7. CritiqueRevision                — single critique-revision record
  8. ConstitutionalConfig            — constitution, prompts
  9. KTOConfig / kto_loss            — Kahneman-Tversky prospect loss
  10. KTOTrainer                     — binary feedback training
  11. RewardModel                    — backbone + scalar head
  12. RMTrainingConfig               — RM training settings
  13. AlignmentEvaluator             — win_rate, kl_div, reward_hacking
  14. AlignmentMetrics               — win_rate, mean_reward, kl_div

Primary exports:
    - :class:`PreferencePair`        — prompt, chosen, rejected
    - :class:`BinaryFeedback`        — prompt, response, is_good
    - :class:`PreferenceDataset`     — add, filter_by_margin, stats
    - :class:`DPOConfig`             — beta, loss_type
    - :func:`dpo_loss`               — DPO/IPO/KTO_pair loss
    - :func:`compute_log_probs`      — sequence log probs
    - :class:`DPOTrainer`            — step, policy + ref training
    - :class:`ConstitutionalAI`      — critique_and_revise, ai_feedback
    - :class:`ConstitutionalConfig`  — constitution, prompt templates
    - :class:`CritiqueRevision`      — original, critique, revised
    - :class:`KTOConfig`             — beta, desirable/undesirable weight
    - :func:`kto_loss`               — prospect-theoretic loss
    - :class:`KTOTrainer`            — step, estimate_kl
    - :class:`RewardModel`           — forward, preference_loss
    - :class:`AlignmentEvaluator`    — evaluate, win_rate, kl
    - :class:`AlignmentMetrics`      — to_dict, is_reward_hacking
"""

from nanomind.alignment.preference import PreferencePair, BinaryFeedback, PreferenceDataset
from nanomind.alignment.dpo import (
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
)
from nanomind.alignment.constitutional import (
    ConstitutionalAI, ConstitutionalConfig, CritiqueRevision,
    DEFAULT_CONSTITUTION,
)
from nanomind.alignment.kto import KTOConfig, kto_loss, KTOTrainer
from nanomind.alignment.reward import RewardModel, RMTrainingConfig
from nanomind.alignment.eval import AlignmentEvaluator, AlignmentMetrics

__all__ = [
    "PreferencePair", "BinaryFeedback", "PreferenceDataset",
    "DPOConfig", "dpo_loss", "compute_log_probs", "DPOTrainer",
    "ConstitutionalAI", "ConstitutionalConfig", "CritiqueRevision", "DEFAULT_CONSTITUTION",
    "KTOConfig", "kto_loss", "KTOTrainer",
    "RewardModel", "RMTrainingConfig",
    "AlignmentEvaluator", "AlignmentMetrics",
]
