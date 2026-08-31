"""NanoMind text generation sub-package.

Primary exports:
    - :class:`Generator`          — high-level text generation interface
    - :class:`GenerationConfig`   — generation hyperparameter configuration
    - :func:`sample_next_token`   — unified next-token sampler
    - :func:`greedy_decode`       — argmax decoding
    - :func:`temperature_sample`  — temperature-scaled sampling
    - :func:`top_k_sample`        — top-K filtered sampling
    - :func:`top_p_sample`        — nucleus (top-p) sampling
    - :func:`beam_search`         — beam search decoder

Logit processors:
    - :func:`apply_temperature`       — scale logits by temperature
    - :func:`apply_top_k`             — zero out below top-K
    - :func:`apply_top_p`             — nucleus filtering
    - :func:`apply_min_p`             — min-p filtering
    - :func:`apply_repetition_penalty`— penalise repeated tokens
"""

from nanomind.generate.config import GenerationConfig
from nanomind.generate.generator import Generator
from nanomind.generate.strategies import (
    greedy_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    sample_next_token,
)
from nanomind.generate.beam import beam_search
from nanomind.generate.logit_processors import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_min_p,
    apply_repetition_penalty,
)

__all__ = [
    "GenerationConfig",
    "Generator",
    "greedy_decode",
    "temperature_sample",
    "top_k_sample",
    "top_p_sample",
    "sample_next_token",
    "beam_search",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_min_p",
    "apply_repetition_penalty",
]

from nanomind.generate.beam import (
    BeamConfig,
    BeamHypothesis,
    BeamHypotheses,
    beam_search,
    diverse_beam_search,
)
from nanomind.generate.beam_generator import BeamSearchGenerator
