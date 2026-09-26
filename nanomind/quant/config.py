"""
nanomind/quant/config.py — Quantization configuration.

## Why Quantize?

FP32 LLM weights: 4 bytes per parameter.
  LLaMA-70B: 70B × 4 bytes = 280 GB — won't fit on consumer hardware!

INT8: 1 byte per parameter → 4× smaller (70 GB)
INT4: 0.5 bytes per parameter → 8× smaller (35 GB)
INT2: 0.25 bytes per parameter → 16× smaller (17.5 GB)

## Quantization Fundamentals

Uniform quantization:
  q(x) = round(x / s) + z
  x̂   = s × (q(x) - z)

where:
  s = scale = (max - min) / (2^b - 1)   [float]
  z = zero point                          [integer]
  b = bits (8, 4, 2)

Error: quantization noise = x̂ - x

## Quantization Schemes

Asymmetric: z ≠ 0, s = (xmax - xmin) / (2^b - 1)
  Better for activations (non-symmetric distributions)

Symmetric: z = 0, s = max(|x|) / (2^{b-1} - 1)
  Better for weights (often symmetric around 0)

## Granularity

Per-tensor:  one (s, z) per tensor — fastest, lowest quality
Per-channel: one (s, z) per output channel — standard for weights
Per-group:   one (s, z) per G-element group — used in GPTQ/AWQ (G=128)

## Post-Training Quantization (PTQ)

Quantize a trained model without further training.
Requires calibration data to estimate activation ranges.

Methods:
  Round-To-Nearest (RTN): direct quantization, fast but low quality
  GPTQ (Frantar et al., 2022): layer-wise optimal quantization
  AWQ (Lin et al., 2023):  activation-aware weight quantization
  SmoothQuant (Xiao et al., 2022): migrate quantization difficulty

## Quantization-Aware Training (QAT)

Simulate quantization during training with straight-through estimator:
  Forward:  use quantized weights
  Backward: treat quantize() as identity (STE)

References:
  Frantar et al. (2022) GPTQ: https://arxiv.org/abs/2210.17323
  Lin et al. (2023) AWQ: https://arxiv.org/abs/2306.00978
  Dettmers et al. (2022) LLM.int8(): https://arxiv.org/abs/2208.07339
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """
    Configuration for model quantization.

    Attributes:
        bits:         Number of bits (8, 4, 2).
        scheme:       ``"symmetric"`` or ``"asymmetric"``.
        granularity:  ``"per_tensor"``, ``"per_channel"``, or ``"per_group"``.
        group_size:   Group size for per-group quantization (default 128).
        method:       ``"rtn"``, ``"gptq"``, ``"awq"``, or ``"qat"``.
        quantize_weights:     Quantize weight tensors.
        quantize_activations: Quantize activation tensors.
        calibration_samples:  Number of samples for PTQ calibration.
        clip_ratio:   Clip outliers at clip_ratio × max_abs (AWQ).
    """
    bits:                 int   = 8
    scheme:               str   = "symmetric"
    granularity:          str   = "per_channel"
    group_size:           int   = 128
    method:               str   = "rtn"
    quantize_weights:     bool  = True
    quantize_activations: bool  = False
    calibration_samples:  int   = 128
    clip_ratio:           float = 1.0

    def __post_init__(self) -> None:
        assert self.bits        in (2, 4, 8, 16)
        assert self.scheme      in ("symmetric", "asymmetric")
        assert self.granularity in ("per_tensor", "per_channel", "per_group")
        assert self.method      in ("rtn", "gptq", "awq", "qat")
        assert self.group_size  >= 1
        assert 0.0 < self.clip_ratio <= 1.0

    @property
    def n_levels(self) -> int:
        """Number of quantization levels."""
        return 2 ** self.bits

    @property
    def q_min(self) -> int:
        if self.scheme == "symmetric":
            return -(2 ** (self.bits - 1))
        return 0

    @property
    def q_max(self) -> int:
        if self.scheme == "symmetric":
            return 2 ** (self.bits - 1) - 1
        return 2 ** self.bits - 1

    @property
    def bytes_per_param(self) -> float:
        return self.bits / 8.0

    def compression_ratio(self, orig_bits: int = 32) -> float:
        return orig_bits / self.bits

    def to_dict(self) -> dict:
        return {
            "bits":          self.bits,
            "scheme":        self.scheme,
            "granularity":   self.granularity,
            "method":        self.method,
            "n_levels":      self.n_levels,
            "compression":   f"{self.compression_ratio():.1f}x",
            "bytes_per_param": self.bytes_per_param,
        }
