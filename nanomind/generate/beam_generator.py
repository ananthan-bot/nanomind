"""
nanomind/generate/beam_generator.py — High-level beam search generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.generate.beam import BeamConfig, BeamHypothesis, beam_search, diverse_beam_search
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger


class BeamSearchGenerator:
    """
    High-level beam search text generator.

    Wraps a model + tokenizer pair to provide a clean generate() API
    that supports both standard beam search and diverse beam search.

    Args:
        model:     Language model.
        tokenizer: Tokenizer for encoding prompts and decoding output.
        device:    Inference device.

    Example::

        gen = BeamSearchGenerator(model, tokenizer)
        texts = gen.generate("Once upon a time", BeamConfig(num_beams=4))
        for t in texts:
            print(t)
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer: BaseTokenizer,
        device:    torch.device | None = None,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.log       = get_logger("generate.beam")
        model.eval().to(self.device)

    def generate(
        self,
        prompt:       str,
        cfg:          BeamConfig | None = None,
        eos_token_id: int | None = None,
    ) -> list[str]:
        """
        Generate text using beam search from a string prompt.

        Args:
            prompt:       Input text prompt.
            cfg:          Beam search configuration.
            eos_token_id: Stop token ID.

        Returns:
            List of generated text strings (length ``cfg.return_n_best``).
        """
        cfg = cfg or BeamConfig()
        ids = self.tokenizer.encode(prompt)
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        n_prompt = len(ids)

        search_fn = (
            diverse_beam_search
            if cfg.num_beam_groups > 1
            else beam_search
        )
        hypotheses = search_fn(self.model, idx, cfg, eos_token_id)

        results = []
        for hyp in hypotheses:
            new_ids = hyp.tokens[n_prompt:]
            results.append(self.tokenizer.decode(new_ids))
        return results

    def __repr__(self) -> str:
        return f"BeamSearchGenerator(model={type(self.model).__name__}, device={self.device})"
