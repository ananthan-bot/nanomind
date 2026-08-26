"""
nanomind/speculative/config.py — Speculative decoding configuration.

Speculative decoding accelerates autoregressive generation by:
  1. Running a small *draft* model to generate K candidate tokens cheaply
  2. Running the large *target* model to verify all K tokens in one forward pass
  3. Accepting tokens where draft and target distributions agree; rejecting the rest
  4. Guaranteed to produce the SAME distribution as pure target-model sampling

Expected speedup: 2-4x on typical text (higher for repetitive/predictable text).

Reference: Leviathan et al. (2022) — https://arxiv.org/abs/2211.17192
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    """
    Configuration for speculative decoding.

    Attributes:
        n_draft:        Number of draft tokens generated per speculative step.
                        Higher = more parallelism but lower acceptance rate.
                        Typical range: 4–8.
        temperature:    Sampling temperature for the target model.
        top_k:          Top-K filter applied to target model logits (0 = off).
        top_p:          Nucleus filter applied to target logits (0.0 = off).
        max_new_tokens: Total number of new tokens to generate.
        seed:           Optional random seed for reproducibility.
    """

    n_draft:        int   = 5
    temperature:    float = 1.0
    top_k:          int   = 0
    top_p:          float = 0.0
    max_new_tokens: int   = 100
    seed:           int | None = None

    def __post_init__(self) -> None:
        assert self.n_draft >= 1,       "n_draft must be >= 1"
        assert self.temperature > 0.0,  "temperature must be positive"
        assert self.top_k >= 0
        assert 0.0 <= self.top_p <= 1.0
        assert self.max_new_tokens > 0
