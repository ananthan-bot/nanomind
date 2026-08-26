"""
nanomind/lora/config.py — LoRA configuration dataclass.

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices A and B
into frozen linear layers. The effective weight update is::

    W' = W + (alpha / r) * B @ A

where r is the rank and alpha is the scaling factor.

Reference: Hu et al. (2021) — https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.

    Attributes:
        r:            LoRA rank (dimensionality of low-rank matrices A and B).
                      Typical values: 4, 8, 16, 32. Higher rank = more capacity.
        alpha:        LoRA scaling factor. Effective scale = alpha / r.
                      Typically set equal to r (scale = 1.0) or 2r (scale = 2.0).
        dropout:      Dropout applied to the LoRA input before A projection.
        target_modules: Names of module types to inject LoRA into.
                      Typical choices: ``["q_proj", "v_proj"]`` (query + value only,
                      as in the original LoRA paper) or all attention projections.
        bias:         Whether to train biases alongside LoRA (``"none"``, ``"all"``,
                      or ``"lora_only"``).
    """

    r:               int        = 8
    alpha:           float      = 16.0
    dropout:         float      = 0.0
    target_modules:  list[str]  = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias:            str        = "none"   # "none", "all", "lora_only"

    def __post_init__(self) -> None:
        assert self.r > 0,              "LoRA rank r must be positive"
        assert self.alpha > 0,          "LoRA alpha must be positive"
        assert 0.0 <= self.dropout < 1.0
        assert self.bias in ("none", "all", "lora_only")

    @property
    def scaling(self) -> float:
        """Effective LoRA scaling factor: alpha / r."""
        return self.alpha / self.r
