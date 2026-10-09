"""
nanomind/safety/config.py — Configuration dataclasses for guardrails, watermarking, and representation engineering.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class WatermarkConfig:
    """Configuration for Kirchenbauer statistical text watermarking."""
    gamma: float = 0.5         # Fraction of vocabulary in green list
    delta: float = 2.0         # Additive logit bias for green tokens
    hash_key: int = 15485863   # Secret PRNG seed key
    z_threshold: float = 4.0   # Detection z-score threshold (p < 3.2e-5)


@dataclass
class RepEConfig:
    """Configuration for Representation Engineering (RepE) concept steering."""
    layer_idx: int = -1
    alpha: float = 1.0         # Steering intensity multiplier
    concept_name: str = "refusal"
    normalize: bool = True


@dataclass
class GuardrailConfig:
    """Configuration for input and output guardrails."""
    enable_pii_redaction: bool = True
    enable_prompt_injection_detection: bool = True
    max_toxicity_threshold: float = 0.5
    action_on_violation: str = "block"  # "block", "redact", "warn"
    refusal_message: str = "I cannot fulfill this request as it violates safety guidelines."


@dataclass
class SafetyConfig:
    """Master configuration for safety and alignment defense."""
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    repe: RepEConfig = field(default_factory=RepEConfig)
    enforce_constitutional_loop: bool = False
    max_critique_turns: int = 2
