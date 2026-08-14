"""
nanomind/model/config.py — NanoMind model configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Full configuration for a NanoMind transformer LLM.

    Attributes:
        vocab_size:     Size of the token vocabulary.
        block_size:     Maximum context window length (sequence length).
        d_model:        Embedding dimension.
        n_layers:       Number of stacked transformer blocks.
        n_heads:        Number of attention heads per block.
        d_ff:           FFN hidden dimension. None defaults to 4 * d_model.
        dropout:        Dropout probability (set to 0 for inference).
        norm_type:      Normalization — ``"layernorm"`` or ``"rmsnorm"``.
        activation:     FFN activation — ``"gelu"`` or ``"swiglu"``.
        norm_placement: ``"pre"`` (Pre-LN) or ``"post"`` (Post-LN).
        bias:           Whether to use bias in attention projections.
        weight_tying:   Tie token embedding weights to LM head.
    """

    vocab_size:     int       = 256
    block_size:     int       = 128
    d_model:        int       = 128
    n_layers:       int       = 4
    n_heads:        int       = 4
    d_ff:           int | None = None
    dropout:        float     = 0.1
    norm_type:      str       = "layernorm"
    activation:     str       = "gelu"
    norm_placement: str       = "pre"
    bias:           bool      = False
    weight_tying:   bool      = True
