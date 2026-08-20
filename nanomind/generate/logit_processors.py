"""
nanomind/generate/logit_processors.py — Logit filtering and processing.

All processors operate on a raw logit tensor of shape ``(vocab_size,)``
and return a modified logit tensor of the same shape.
Processors can be composed in sequence before sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature to sharpen or flatten the distribution.

    Lower temperature (< 1) makes distribution peakier (more confident).
    Higher temperature (> 1) makes distribution flatter (more random).

    Args:
        logits:      Raw logits ``(vocab_size,)``
        temperature: Positive scalar. Must be > 0.

    Returns:
        Scaled logit tensor ``(vocab_size,)``
    """
    return logits / max(temperature, 1e-8)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Set all logits below the top-K to -inf.

    Args:
        logits: Raw logits ``(vocab_size,)``
        top_k:  Number of top logits to keep. If 0, no filtering is applied.

    Returns:
        Filtered logit tensor with non-top-k entries set to -inf.
    """
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold, _ = torch.topk(logits, k)
    min_threshold = threshold[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability mass exceeds ``top_p``.

    Args:
        logits: Raw logits ``(vocab_size,)``
        top_p:  Cumulative probability threshold in (0, 1].
                If 0.0 or >= 1.0, no filtering is applied.

    Returns:
        Filtered logit tensor.
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative probability exceeds top_p
    remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[remove] = float("-inf")
    # Restore original order
    return sorted_logits.scatter(0, sorted_idx, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-P filtering: remove tokens with probability < min_p * max_prob.

    Args:
        logits: Raw logits ``(vocab_size,)``
        min_p:  Minimum probability ratio threshold [0, 1].

    Returns:
        Filtered logit tensor.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    past_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Reduce logits for tokens that already appeared in ``past_ids``.

    Divides positive logits by ``penalty`` and multiplies negative logits
    by ``penalty``, making repeated tokens less likely.

    Args:
        logits:   Raw logits ``(vocab_size,)``
        past_ids: Previously generated token IDs ``(T,)``
        penalty:  Penalty factor (>= 1.0; 1.0 = no effect).

    Returns:
        Modified logit tensor.
    """
    if penalty == 1.0 or past_ids.numel() == 0:
        return logits
    score = logits.clone()
    for token_id in past_ids.unique():
        if score[token_id] < 0:
            score[token_id] *= penalty
        else:
            score[token_id] /= penalty
    return score
