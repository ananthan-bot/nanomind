"""
nanomind/streaming/generator.py — Token-by-token streaming generation.

Wraps a NanoMind model to yield one token at a time, enabling:
  - Real-time UI updates (text appears as it's generated)
  - Early stopping (stop_sequences, timeout)
  - Per-token logprob tracking
  - Streaming to multiple consumers simultaneously
"""

from __future__ import annotations

import math
import time
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.streaming.config import StreamConfig
from nanomind.streaming.types import StreamToken, StreamEvent

if TYPE_CHECKING:
    pass


def _sample_next(
    logits:      torch.Tensor,
    temperature: float,
    top_k:       int,
    top_p:       float,
) -> tuple[int, float]:
    """
    Sample the next token from logits with temperature, top-K, and nucleus sampling.

    Args:
        logits:      ``(V,)`` unnormalised logits for the next token.
        temperature: Sampling temperature (0 = greedy argmax).
        top_k:       Keep only top-K tokens.
        top_p:       Nucleus: keep smallest set with cumulative prob >= top_p.

    Returns:
        ``(token_id, log_probability)``
    """
    if temperature == 0.0:
        return int(logits.argmax().item()), 0.0

    logits = logits / max(temperature, 1e-8)

    # Top-K
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits       = logits.masked_fill(logits < top_vals[..., -1:], float("-inf"))

    # Nucleus (top-p)
    if top_p < 1.0:
        probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
        cumprobs = probs_sorted.cumsum(dim=-1)
        remove   = cumprobs - probs_sorted > top_p
        remove[..., :1] = False          # always keep top token
        logits.scatter_(-1, sorted_idx, logits.masked_fill(remove, float("-inf")))

    probs    = F.softmax(logits, dim=-1)
    token_id = int(torch.multinomial(probs, 1).item())
    logprob  = math.log(max(probs[token_id].item(), 1e-9))
    return token_id, logprob


class StreamingGenerator:
    """
    Token-by-token streaming generator for NanoMind models.

    Wraps any NanoMind model and yields :class:`StreamToken` objects
    one at a time, compatible with SSE streaming.

    Args:
        model:     Language model (must have ``forward(x) → (logits, loss)``).
        tokenizer: Tokenizer with ``encode(str) → list[int]``
                   and ``decode(list[int]) → str``.
        cfg:       Streaming configuration.

    Example::

        gen = StreamingGenerator(model, tokenizer)
        for token in gen.stream("Once upon a time"):
            print(token.token, end="", flush=True)
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer,
        cfg:       StreamConfig | None = None,
    ) -> None:
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.cfg       = cfg or StreamConfig()

    @torch.no_grad()
    def stream(
        self,
        prompt:     str,
        cfg:        StreamConfig | None = None,
        request_id: str = "",
    ) -> Iterator[StreamToken]:
        """
        Stream tokens one-by-one for the given prompt.

        Args:
            prompt:     Input text prompt.
            cfg:        Override default config for this call.
            request_id: Optional ID for tracking.

        Yields:
            :class:`StreamToken` for each generated token.
        """
        cfg      = cfg or self.cfg
        ids      = self.tokenizer.encode(prompt)
        ctx      = torch.tensor([ids], dtype=torch.long)
        block_sz = getattr(self.model, "T", 512)

        generated  = []
        t_start    = time.time()

        for i in range(cfg.max_new_tokens):
            # Timeout guard
            if time.time() - t_start > cfg.timeout_seconds:
                break

            # Truncate context to block size
            ctx_crop  = ctx[:, -block_sz:]
            logits, _ = self.model(ctx_crop)
            next_logits = logits[0, -1, :]

            token_id, logprob = _sample_next(
                next_logits, cfg.temperature, cfg.top_k, cfg.top_p
            )

            # Decode single token
            token_str = self.tokenizer.decode([token_id])
            generated.append(token_id)
            ctx = torch.cat([ctx, torch.tensor([[token_id]])], dim=1)

            # Yield every stream_interval tokens
            if (i + 1) % cfg.stream_interval == 0:
                yield StreamToken(
                    token    = token_str,
                    token_id = token_id,
                    index    = i,
                    logprob  = logprob if cfg.include_logprobs else None,
                )

            # Check stop sequences
            so_far = self.tokenizer.decode(generated)
            if any(stop in so_far for stop in cfg.stop_sequences):
                break

    def stream_events(
        self,
        prompt:     str,
        cfg:        StreamConfig | None = None,
        request_id: str = "",
    ) -> Iterator[StreamEvent]:
        """
        Stream :class:`StreamEvent` objects (ready for SSE serialisation).

        Yields token events, then a finish event.
        """
        cfg   = cfg or self.cfg
        count = 0
        for token in self.stream(prompt, cfg, request_id):
            yield StreamEvent.token_event(token, request_id)
            count += 1
        yield StreamEvent.finish_event(count, request_id=request_id)

    def generate_full(self, prompt: str, cfg: StreamConfig | None = None) -> str:
        """
        Convenience: generate and return the full text (non-streaming).

        Args:
            prompt: Input text.
            cfg:    Optional config override.

        Returns:
            Generated text string.
        """
        tokens = list(self.stream(prompt, cfg))
        return "".join(t.token for t in tokens)
