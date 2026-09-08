"""NanoMind Knowledge Distillation sub-package.

Train a smaller student model to mimic a larger teacher model.

Primary exports:
    - :class:`DistillTrainer`        — teacher + student training loop
    - :class:`DistillConfig`         — temperature, alpha, feature_distill
    - :func:`distillation_loss`      — combined hard + soft KL loss
    - :func:`soft_cross_entropy`     — soft-label KL divergence
    - :func:`feature_distillation_loss` — hidden state MSE matching
"""

from nanomind.distill.config import DistillConfig
from nanomind.distill.loss import (
    distillation_loss, soft_cross_entropy, feature_distillation_loss
)
from nanomind.distill.trainer import DistillTrainer

__all__ = [
    "DistillConfig", "DistillTrainer",
    "distillation_loss", "soft_cross_entropy", "feature_distillation_loss",
]
