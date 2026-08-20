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

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
    ) -> str:
        """
        Generate text from a string prompt.

        Args:
            prompt: Input text to condition generation on.
            cfg:    :class:`GenerationConfig` (uses defaults if None).

        Returns:
            Generated text string (excluding the prompt).
        """
        cfg = cfg or GenerationConfig()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # Encode prompt
        ids  = self.tokenizer.encode(prompt)
        idx  = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Beam search path
        if cfg.strategy == "beam" or cfg.num_beams > 1:
            result = beam_search(
                self.model, idx,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
                eos_token_id=cfg.eos_token_id,
            )
            new_ids = result[0, len(ids):].tolist()
            return self.tokenizer.decode(new_ids)

        # Autoregressive sampling
        block_size = getattr(self.model, "cfg", None) and self.model.cfg.block_size or 512
        generated: list[int] = []

        for _ in range(cfg.max_new_tokens):
            ctx     = idx[:, -block_size:]
            logits, _ = self.model(ctx)
            next_tok = sample_next_token(
                logits[0, -1, :],
                strategy=cfg.strategy,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                min_p=cfg.min_p,
                repetition_penalty=cfg.repetition_penalty,
                past_ids=idx[0] if cfg.repetition_penalty != 1.0 else None,
            )
            tok_id = next_tok.item()

            # EOS check
            if cfg.eos_token_id is not None and tok_id == cfg.eos_token_id:
                break

            generated.append(tok_id)
            idx = torch.cat([idx, next_tok.unsqueeze(0).unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated)
