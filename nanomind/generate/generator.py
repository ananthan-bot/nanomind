"""
nanomind/generate/generator.py — High-level text generator for NanoMind.

The Generator class wraps a model and tokenizer to provide a convenient
API for text generation with any supported strategy.
"""

from __future__ import annotations

from typing import Generator as PythonGenerator, Iterator

import torch
import torch.nn as nn

from nanomind.generate.config import GenerationConfig
from nanomind.generate.strategies import sample_next_token
from nanomind.generate.beam import beam_search
from nanomind.tokenizer.base import BaseTokenizer


class Generator:
    """
    High-level text generation interface.

    Args:
        model:     The :class:`~nanomind.model.NanoMind` model in eval mode.
        tokenizer: A fitted tokenizer for encoding prompts and decoding output.
        device:    Device to run generation on.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.model.eval()
        self.model.to(self.device)
