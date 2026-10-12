"""
nanomind/flow/metrics.py — Quality and unmasking dynamics metrics for discrete diffusion language models.
"""
from typing import List, Dict, Any
import torch


def compute_unmasking_trajectory(token_sequence_history: List[torch.Tensor], mask_token_id: int = 0) -> List[float]:
    """
    Measures the unmasking fraction (active unmasked tokens / total tokens) across diffusion steps.
    """
    fractions = []
    for step_tokens in token_sequence_history:
        total = step_tokens.numel()
        unmasked = (step_tokens != mask_token_id).sum().item()
        fractions.append(round(unmasked / max(1, total), 4))
    return fractions


def compute_token_diversity(tokens: torch.Tensor) -> float:
    """Computes distinct unigram ratio (unique tokens / total tokens)."""
    flat = tokens.flatten().tolist()
    if not flat:
        return 0.0
    return round(len(set(flat)) / len(flat), 4)
