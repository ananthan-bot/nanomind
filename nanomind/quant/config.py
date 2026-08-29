"""
nanomind/quant/config.py — Quantization configuration.

Post-training quantization (PTQ) compresses model weights from float32/float16
to int8 with minimal accuracy loss. Key techniques:

  Symmetric per-tensor:   one scale per weight matrix  (fastest, least accurate)
  Symmetric per-channel:  one scale per output channel  (better accuracy)
  Dynamic quantization:   activations quantized at runtime, weights offline

Benefits:
  - 4x smaller model size (float32 → int8)
  - 2-4x faster matrix multiplications on supported hardware
  - Lower memory bandwidth requirements

Reference: Dettmers et al. (2022) LLM.int8() — https://arxiv.org/abs/2208.07339
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class QuantConfig:
    """
    Configuration for INT8 post-training quantization.

    Attributes:
        mode:           ``"weight_only"`` (quantize weights, float activations)
                        ``"dynamic"``    (quantize weights + activations at runtime)
        granularity:    ``"per_tensor"``  — one scale per weight matrix
                        ``"per_channel"`` — one scale per output channel (more accurate)
        target_modules: Layer name patterns to quantize. Defaults to all Linear layers.
        skip_modules:   Layer name patterns to leave in float (e.g., LM head).
        bits:           Quantization bit width (currently only 8 supported).
    """

    mode:           str        = "weight_only"
    granularity:    str        = "per_channel"
    target_modules: list[str]  = field(default_factory=lambda: ["Linear"])
    skip_modules:   list[str]  = field(default_factory=lambda: ["lm_head"])
    bits:           int        = 8

    def __post_init__(self) -> None:
        assert self.mode in ("weight_only", "dynamic"),             f"Unsupported mode: {self.mode}"
        assert self.granularity in ("per_tensor", "per_channel"),             f"Unsupported granularity: {self.granularity}"
        assert self.bits == 8, "Only 8-bit quantization is currently supported"
