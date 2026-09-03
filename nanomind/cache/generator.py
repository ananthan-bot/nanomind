"""
nanomind/cache/generator.py — Fast cached generation with temperature + top-K/top-P.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nanomind.cache.model import NanoMindCached
from nanomind.cache.manager import KVCacheManager
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("cache.generator")


def _sample_next(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
) -> torch.Tensor:
    """
    Sample next token from logits with temperature, top-K, and top-P filtering.

    Args:
        logits:      ``(B, vocab_size)`` raw logits.
        temperature: Logit temperature (< 1 = sharper, > 1 = flatter).
        top_k:       Keep only top-K tokens (0 = off).
        top_p:       Nucleus sampling threshold (1.0 = off).

    Returns:
        Sampled token IDs ``(B, 1)``.
    """
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k_vals = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < k_vals, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative               = sorted_probs.cumsum(dim=-1)
        remove                   = cumulative - sorted_probs > top_p
        sorted_probs[remove]     = 0.0
        sorted_probs            /= sorted_probs.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

    return torch.multinomial(probs, num_samples=1)


class CachedGenerator:
    """
    Fast autoregressive text generator using KV cache.

    Compared to naive generation, KV cache gives approximately
    ``T × n_layers`` fewer matrix multiplications per generated token.

    Args:
        model:     NanoMindCached model.
        tokenizer: Tokenizer for encoding/decoding.

    Example::

        gen = CachedGenerator(model, tokenizer)
        text = gen.generate("Once upon a time", max_new_tokens=100)
        print(text)
    """

    def __init__(
        self,
        model:     NanoMindCached,
        tokenizer: BaseTokenizer,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 1.0,
        eos_token_id:   int | None = None,
    ) -> str:
        """
        Generate text from a string prompt using KV cache.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature:    Sampling temperature.
            top_k:          Top-K filter (0 = off).
            top_p:          Nucleus sampling threshold (1.0 = off).
            eos_token_id:   Stop when this token is generated.

        Returns:
            Generated text (excluding the prompt).
        """
        ids    = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        idx    = torch.tensor([ids], dtype=torch.long, device=device)

        cache  = self.model.new_cache()

        # Prefill — process the full prompt
        logits = self.model.prefill(idx, cache)

        generated: list[int] = []

        for _ in range(max_new_tokens):
            next_tok = _sample_next(logits[:, -1, :], temperature, top_k, top_p)
            tok_id   = next_tok.item()

            if eos_token_id is not None and tok_id == eos_token_id:
                break

            generated.append(tok_id)

            # Decode step — one token at a time, O(1) thanks to cache
            logits = self.model.decode_step(next_tok, cache)

        return self.tokenizer.decode(generated)

    def __repr__(self) -> str:
        return f"CachedGenerator(model={type(self.model).__name__})"
