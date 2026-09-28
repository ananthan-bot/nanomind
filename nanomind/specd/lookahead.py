"""
nanomind/specd/lookahead.py — Lookahead Decoding (Fu et al., 2023).

## Lookahead Decoding

Unlike speculative decoding (uses two models), lookahead decoding
uses ONE model but exploits the Jacobi iteration structure.

Key idea: standard autoregressive decoding solves a sequential system.
Lookahead treats it as a parallel Jacobi iteration:

  Standard:
    x_{n+1} = argmax p(x | x_1..x_n)   ← sequential

  Lookahead:
    Maintain a "lookahead" window W of speculative future tokens.
    Run the model on context + window in parallel.
    Update window based on model predictions.
    Commit tokens when they converge (Jacobi fixed point).

Speedup: 1.5-2.5× with a single model.
No draft model needed — pure single-model speculation.

Reference:
  Fu et al. (2023) "Break the Sequential Dependency of LLM Inference
  Using Lookahead Decoding"
  https://arxiv.org/abs/2402.02057
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LookaheadState:
    """Internal state of lookahead decoding."""
    window:      torch.Tensor    # (B, W) current lookahead window
    n_iters:     int = 0
    n_committed: int = 0


class LookaheadDecoder:
    """
    Lookahead decoding: single-model speculation via Jacobi iteration.

    Args:
        model:       LM model ``forward(input_ids) → (B, T, V)`` logits.
        window_size: Lookahead window width W.
        n_iters:     Jacobi iterations per step (more = higher acceptance).
        temperature: Sampling temperature.

    Example::

        decoder = LookaheadDecoder(model, window_size=5, n_iters=2)
        ids     = decoder.generate(input_ids, max_new_tokens=32)
    """

    def __init__(
        self,
        model:       nn.Module,
        window_size: int   = 5,
        n_iters:     int   = 2,
        temperature: float = 1.0,
    ) -> None:
        self.model       = model
        self.window_size = window_size
        self.n_iters     = n_iters
        self.temperature = max(temperature, 1e-8)

    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids)
            if isinstance(out, tuple):
                return out[0]
            return out

    def _sample_next(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits at last position."""
        l = logits[:, -1, :] / self.temperature
        return F.softmax(l, dim=-1).argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Generate with lookahead decoding.

        Args:
            input_ids:      ``(B, T)`` context.
            max_new_tokens: Tokens to generate.

        Returns:
            ``(B, T + new)`` generated token IDs.
        """
        B, T     = input_ids.shape
        current  = input_ids.clone()
        generated = 0

        # Initialise lookahead window with repeated last token
        window = input_ids[:, -1:].expand(B, self.window_size).clone()

        while generated < max_new_tokens:
            # Jacobi iteration: refine window
            for _ in range(self.n_iters):
                ctx    = torch.cat([current, window], dim=1)   # (B, T+W)
                logits = self._get_logits(ctx)                  # (B, T+W, V)
                # Update window from model predictions
                T_cur    = current.shape[1]
                new_win  = logits[:, T_cur-1 : T_cur+self.window_size-1, :]
                new_win  = (new_win / self.temperature).softmax(-1).argmax(-1)
                window   = new_win

            # Commit tokens that match between consecutive iterations
            # (simplified: commit first window token unconditionally)
            ctx    = torch.cat([current, window], dim=1)
            logits = self._get_logits(ctx)
            T_cur  = current.shape[1]

            # Accept first token
            next_t = logits[:, T_cur - 1, :].argmax(-1, keepdim=True)
            current = torch.cat([current, next_t], dim=1)
            generated += 1

            # Slide window
            window = torch.cat([window[:, 1:], next_t], dim=1)

        return current
