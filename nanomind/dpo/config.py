"""
nanomind/dpo/config.py — Direct Preference Optimization configuration.

## Why DPO? (vs RLHF/PPO)

RLHF (PPO) is complex:
  SFT → Reward Model → PPO training loop (4 forward passes/step!)
  Requires separate reward model, value model, reference model + policy

DPO (Rafailov et al. 2023) eliminates the reward model entirely:
  SFT → DPO fine-tuning (only 2 models: policy + frozen reference)
  Uses Bradley-Terry preference data (same as RM training)

DPO Loss (for a chosen/rejected pair):
  L_DPO = -log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x))
               - β · log(π_θ(y_l|x)/π_ref(y_l|x)))

Intuition:
  Increase the log-ratio of chosen over reference → policy prefers chosen
  Decrease the log-ratio of rejected over reference → policy avoids rejected
  β controls how tightly we stay close to the reference

Used in: LLaMA 2 Chat, Mistral Instruct, Zephyr, Phi-2

Reference: Rafailov et al. (2023) "DPO: Direct Preference Optimization"
           https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """
    Configuration for DPO fine-tuning.

    Attributes:
        beta:           Temperature controlling divergence from reference.
                        Higher β → stay closer to reference policy.
                        Typical: 0.1-0.5.
        label_smoothing: Smoothing on chosen/rejected labels (0 = off).
        loss_type:      ``"sigmoid"`` (standard DPO) or ``"ipo"``
                        (Identity Preference Optimisation, Azar 2023).
        reference_free: If True, skip reference model (β acts as regulariser).
        max_length:     Max sequence length for DPO inputs.
        max_prompt_length: Max prompt length (truncate prompt if longer).
    """

    beta:             float = 0.1
    label_smoothing:  float = 0.0
    loss_type:        str   = "sigmoid"
    reference_free:   bool  = False
    max_length:       int   = 512
    max_prompt_length:int   = 256

    def __post_init__(self) -> None:
        assert self.beta > 0.0
        assert 0.0 <= self.label_smoothing < 0.5
        assert self.loss_type in ("sigmoid", "ipo")
        assert self.max_length > 0
