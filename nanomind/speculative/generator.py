"""
nanomind/speculative/generator.py — High-level speculative generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.speculative.config import SpeculativeConfig
from nanomind.speculative.decode import speculative_decode
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger


class SpeculativeGenerator:
    """
    High-level speculative text generator.

    Wraps a target model + draft model pair to provide a convenient
    generate() API that returns both text and performance statistics.

    Args:
        target_model: Large, accurate language model.
        draft_model:  Small, fast language model (same vocabulary).
        tokenizer:    Tokenizer for encoding prompts and decoding output.
        device:       Device to run inference on.

    Example::

        gen = SpeculativeGenerator(large_model, small_model, tokenizer)
        text, stats = gen.generate("Once upon a time")
        print(f"Generated: {text}")
        print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model:  nn.Module,
        tokenizer:    BaseTokenizer,
        device:       torch.device | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model  = draft_model
        self.tokenizer    = tokenizer
        self.device = device or next(target_model.parameters()).device
        self.log    = get_logger("speculative.generator")

        target_model.eval().to(self.device)
        draft_model.eval().to(self.device)

    def generate(
        self,
        prompt: str,
        cfg: SpeculativeConfig | None = None,
    ) -> tuple[str, dict]:
        """
        Generate text speculatively from a string prompt.

        Args:
            prompt: Input text to condition generation on.
            cfg:    Speculative decoding configuration.

        Returns:
            Tuple of ``(generated_text, stats_dict)``.
            Stats include ``acceptance_rate``, ``n_draft_calls``, etc.
        """
        cfg = cfg or SpeculativeConfig()

        ids  = self.tokenizer.encode(prompt)
        idx  = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        n_prompt = len(ids)

        output_ids, stats = speculative_decode(
            self.target_model,
            self.draft_model,
            idx,
            cfg=cfg,
        )

        new_ids = output_ids[0, n_prompt:].tolist()
        text    = self.tokenizer.decode(new_ids)
        return text, stats

    def __repr__(self) -> str:
        return (
            f"SpeculativeGenerator("
            f"target={type(self.target_model).__name__}, "
            f"draft={type(self.draft_model).__name__}, "
            f"device={self.device})"
        )
