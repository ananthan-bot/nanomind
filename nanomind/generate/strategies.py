"""
nanomind/generate/strategies.py — Sampling strategy implementations.

Each strategy takes a processed logit tensor ``(vocab_size,)`` and
returns the next token as a scalar integer tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: always pick the highest-probability token.

    Deterministic — no randomness involved.

    Args:
        logits: Raw logits ``(vocab_size,)``

    Returns:
        Scalar token ID tensor.
    """
    return logits.argmax(dim=-1)


def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from a temperature-scaled distribution.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        temperature: Softmax temperature. < 1 = sharper, > 1 = flatter.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature
    scaled = apply_temperature(logits, temperature)
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
