"""
nanomind/speech/loss.py — Connectionist Temporal Classification (CTC) and Multi-Stage Codec Cross-Entropy.
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    CTC Loss for speech recognition without requiring explicit frame-level alignments.
    """

    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        log_probs: (T, B, C) log probabilities
        targets: (B, S) target indices
        input_lengths: (B,)
        target_lengths: (B,)
        """
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


class MultiStageCodecLoss(nn.Module):
    """
    Cross-entropy loss for multi-stage RVQ acoustic codes.
    Calculates weighted cross-entropy across all n_q quantizer levels.
    """

    def __init__(self, n_q: int = 8, decay_per_stage: float = 0.9):
        super().__init__()
        self.n_q = n_q
        self.decay_per_stage = decay_per_stage

    def forward(self, logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, codebook_size)
        target_codes: (B, n_q, T)
        """
        B, n_q, T = target_codes.shape
        total_loss = torch.tensor(0.0, device=logits.device)
        total_weight = 0.0

        for stage in range(min(n_q, self.n_q)):
            weight = self.decay_per_stage ** stage
            targets_stage = target_codes[:, stage, :]  # (B, T)
            loss_stage = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_stage.view(-1))
            total_loss = total_loss + weight * loss_stage
            total_weight += weight

        return total_loss / max(1e-6, total_weight)
