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


def sample_next_token(
    logits: torch.Tensor,
    strategy: str = "temperature",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    past_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Unified next-token sampler: applies processors then dispatches to strategy.

    Processing order:
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k filtering
    4. Top-p (nucleus) filtering
    5. Min-p filtering
    6. Softmax + sample (or argmax for greedy)

    Args:
        logits:             Raw logit vector ``(vocab_size,)``
        strategy:           ``"greedy"``, ``"temperature"``, ``"top_k"``, or ``"top_p"``
        temperature:        Softmax temperature.
        top_k:              Top-K filter (0 = disabled).
        top_p:              Nucleus threshold (0.0 = disabled).
        min_p:              Min-p threshold (0.0 = disabled).
        repetition_penalty: Penalty for previously generated tokens.
        past_ids:           Previously generated IDs for repetition penalty.

    Returns:
        Next token as a scalar tensor.
    """
    from nanomind.generate.logit_processors import (
        apply_temperature, apply_top_k, apply_top_p,
        apply_min_p, apply_repetition_penalty,
    )

    if strategy == "greedy":
        return greedy_decode(logits)

    # Apply processors in order
    if repetition_penalty != 1.0 and past_ids is not None:
        logits = apply_repetition_penalty(logits, past_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    if top_k > 0:
        logits = apply_top_k(logits, top_k)
    if top_p > 0.0:
        logits = apply_top_p(logits, top_p)
    if min_p > 0.0:
        logits = apply_min_p(logits, min_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def beam_decode(
    model:        "nn.Module",
    input_ids:    "torch.Tensor",
    num_beams:    int = 4,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
    **kwargs,
) -> "torch.Tensor":
    """
    Beam search decode — convenience wrapper for use in the generate pipeline.
    Returns the best beam as a token tensor (1, T + generated).
    """
    from nanomind.generate.beam import BeamConfig, beam_search
    cfg = BeamConfig(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        return_n_best=1,
    )
    hyps = beam_search(model, input_ids, cfg)
    best = hyps[0].tokens
    import torch
    return torch.tensor([best], dtype=torch.long, device=input_ids.device)
