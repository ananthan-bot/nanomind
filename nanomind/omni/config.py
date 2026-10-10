"""
nanomind/omni/config.py — Configuration dataclasses for unified omni-modal modeling.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict


class ModalityType(str, Enum):
    """Enumeration of supported modality token types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TOOL = "tool"


@dataclass
class DuplexConfig:
    """Configuration for real-time duplex streaming dialogue and interruption handling."""
    chunk_size_ms: int = 200
    vad_energy_threshold: float = 0.02
    barge_in_sensitivity: float = 0.6
    interruption_cooldown_ms: int = 400


@dataclass
class OmniConfig:
    """Master configuration for NanoMind-Omni unified foundation model."""
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 1024
    max_seq_len: int = 2048
    text_vocab_size: int = 50257
    audio_codebook_size: int = 1024
    image_patch_dim: int = 256
    audio_mel_dim: int = 80
    dropout: float = 0.1
    duplex: DuplexConfig = field(default_factory=DuplexConfig)
