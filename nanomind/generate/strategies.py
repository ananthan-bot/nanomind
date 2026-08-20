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
