"""
nanomind/flow/sedd.py — Score Entropy Discrete Diffusion (SEDD, Lou et al., ICML 2024).
Computes concrete score matching objective over discrete token manifolds.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def score_entropy_loss(
    score_logits: torch.Tensor,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Score Entropy loss for discrete diffusion:
    L = E_{t, x_t, x_0} [ sum_{y != x_t} exp(s_theta(x_t, t)_y) - s_theta(x_t, t)_{x_0} ]
    score_logits: (B, T, vocab_size) model predicted unnormalized scores
    x_t: (B, T) corrupted tokens
    x_0: (B, T) clean ground-truth tokens
    """
    B, T, V = score_logits.shape

    # Cross-entropy surrogate formulation:
    # Only compute loss on tokens that were corrupted/masked in x_t
    mask = (x_t == mask_token_id)

    if not mask.any():
        return torch.tensor(0.0, device=score_logits.device, requires_grad=True)

    # Standard cross-entropy between score logits and true clean tokens at masked positions
    flat_logits = score_logits[mask]
    flat_targets = x_0[mask]

    ce_loss = F.cross_entropy(flat_logits, flat_targets)
    return ce_loss
