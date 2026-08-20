"""
nanomind/generate/config.py — Generation configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """
    Configuration for NanoMind text generation.

    Attributes:
        max_new_tokens:    Maximum tokens to generate.
        strategy:          Sampling strategy — ``"greedy"``, ``"temperature"``,
                           ``"top_k"``, ``"top_p"``, ``"beam"``.
        temperature:       Softmax temperature (< 1 = sharper, > 1 = more random).
        top_k:             Keep top-K logits before sampling (0 = disabled).
        top_p:             Nucleus probability threshold (0.0 = disabled).
        min_p:             Minimum probability threshold (0.0 = disabled).
        repetition_penalty: Penalize repeated tokens (1.0 = disabled).
        num_beams:         Number of beams for beam search.
        eos_token_id:      Token ID that signals end of sequence (None = no EOS).
        seed:              Optional random seed for reproducible sampling.
    """

    max_new_tokens:     int   = 100
    strategy:           str   = "temperature"
    temperature:        float = 0.8
    top_k:              int   = 50
    top_p:              float = 0.0
    min_p:              float = 0.0
    repetition_penalty: float = 1.0
    num_beams:          int   = 1
    eos_token_id:       int | None = None
    seed:               int | None = None

    def __post_init__(self) -> None:
        assert self.max_new_tokens > 0
        assert self.temperature > 0.0
        assert self.top_k >= 0
        assert 0.0 <= self.top_p <= 1.0
        assert 0.0 <= self.min_p <= 1.0
        assert self.repetition_penalty >= 1.0
        assert self.num_beams >= 1
        assert self.strategy in ("greedy", "temperature", "top_k", "top_p", "beam"), (
            f"Unknown strategy '{self.strategy}'"
        )
