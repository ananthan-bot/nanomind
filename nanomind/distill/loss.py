"""
nanomind/distill/loss.py — Knowledge distillation loss functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature:    float = 4.0,
) -> torch.Tensor:
    """
    Soft-label KL divergence loss (student ← teacher soft labels).

    Args:
        student_logits: ``(N, V)`` student model logits.
        teacher_logits: ``(N, V)`` teacher model logits.
        temperature:    Softening temperature T.

    Returns:
        Scalar KL divergence loss (scaled by T²).
    """
    T           = temperature
    student_lp  = F.log_softmax(student_logits / T, dim=-1)
    teacher_p   = F.softmax(teacher_logits  / T, dim=-1)
    # KL(teacher || student) * T²
    return F.kl_div(student_lp, teacher_p, reduction="batchmean") * (T ** 2)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels:    torch.Tensor,
    temperature:    float = 4.0,
    alpha:          float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Combined hard + soft distillation loss.

    L = α · CE(student, hard) + (1-α) · KL(teacher_soft, student_soft) · T²

    Args:
        student_logits: ``(N, V)`` student logits.
        teacher_logits: ``(N, V)`` teacher logits (no grad expected).
        hard_labels:    ``(N,)`` ground truth token IDs.
        temperature:    Softening temperature.
        alpha:          Hard-label weight (0 = pure distillation).

    Returns:
        Tuple of ``(total_loss, info_dict)``.
    """
    ce_loss   = F.cross_entropy(student_logits, hard_labels)
    kd_loss   = soft_cross_entropy(student_logits, teacher_logits, temperature)
    total     = alpha * ce_loss + (1 - alpha) * kd_loss

    return total, {
        "loss":    total.item(),
        "ce_loss": ce_loss.item(),
        "kd_loss": kd_loss.item(),
    }


def feature_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection:     torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Feature-level distillation: MSE between student and teacher hidden states.

    If student and teacher have different hidden dimensions, supply a
    ``projection`` linear layer to align them.

    Args:
        student_hidden: ``(B, T, d_student)``
        teacher_hidden: ``(B, T, d_teacher)``
        projection:     Optional linear to project student → teacher dim.

    Returns:
        Scalar MSE loss.
    """
    if projection is not None:
        student_hidden = projection(student_hidden)
    return F.mse_loss(student_hidden, teacher_hidden.detach())
