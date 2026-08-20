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


def top_k_sample(
    logits: torch.Tensor,
    top_k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample after keeping only the top-K logits.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        top_k:       Number of top tokens to sample from.
        temperature: Temperature scaling applied before sampling.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature, apply_top_k
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds ``top_p``.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        top_p:       Nucleus probability threshold.
        temperature: Temperature scaling applied before sampling.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature, apply_top_p
    logits = apply_temperature(logits, temperature)
    logits = apply_top_p(logits, top_p)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
