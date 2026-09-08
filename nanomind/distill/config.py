"""
nanomind/distill/config.py — Knowledge Distillation configuration.

## Knowledge Distillation (Hinton et al. 2015)

A large "teacher" model trains a smaller "student" model to mimic its output
distributions — not just the hard labels but the soft probability distributions.

Student loss = α · CE(student, hard_labels)
             + (1-α) · KL(softmax(teacher/T), softmax(student/T)) · T²

Temperature T > 1 "softens" the teacher's distribution, revealing which
wrong answers the teacher considers plausible — rich training signal for
the student beyond just the one-hot ground truth.

Used in: DistilBERT (66% of BERT size, 97% of BERT performance),
         TinyBERT, MiniLM, DistilGPT-2.

Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
           https://arxiv.org/abs/1503.02531
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DistillConfig:
    """
    Configuration for knowledge distillation.

    Attributes:
        temperature:  Softening temperature T (higher = softer teacher distribution).
        alpha:        Weight of hard-label CE loss (0 = pure distillation).
        feature_distill: Also match intermediate hidden states (feature-level KD).
        feature_weight:  Weight of feature-matching loss.
        teacher_layer_map: Dict mapping student layer idx → teacher layer idx for feature KD.
    """

    temperature:      float = 4.0
    alpha:            float = 0.5
    feature_distill:  bool  = False
    feature_weight:   float = 0.1
    teacher_layer_map:dict  | None = None

    def __post_init__(self) -> None:
        assert self.temperature >= 1.0
        assert 0.0 <= self.alpha <= 1.0
        assert self.feature_weight >= 0.0
