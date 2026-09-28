"""
nanomind/specd/engine.py — Speculative decoding generation engine.

Orchestrates the full speculative decoding loop:
  1. Draft K tokens with small model
  2. Run target model on context + K tokens (one pass)
  3. Accept/reject using SpeculativeSampler
  4. Update context with accepted tokens
  5. Repeat until max_new_tokens

Also tracks statistics: tokens/step, acceptance rates, speedup.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from nanomind.specd.draft import DraftModel
from nanomind.specd.sampler import SpeculativeSampler, SpeculativeResult
from nanomind.utils.logger import get_logger

log = get_logger("specd.engine")


@dataclass
class GenerationStats:
    """Statistics from a speculative decoding generation run."""
    n_tokens_generated: int   = 0
    n_target_calls:     int   = 0
    n_draft_calls:      int   = 0
    total_accepted:     int   = 0
    total_drafted:      int   = 0
    wall_time_s:        float = 0.0
    acceptance_rates:   list  = field(default_factory=list)

    @property
    def mean_accepted_per_step(self) -> float:
        return self.total_accepted / max(self.n_target_calls, 1)

    @property
    def acceptance_rate(self) -> float:
        if not self.acceptance_rates:
            return 0.0
        return sum(self.acceptance_rates) / len(self.acceptance_rates)

    @property
    def tokens_per_second(self) -> float:
        return self.n_tokens_generated / max(self.wall_time_s, 1e-6)

    @property
    def speedup(self) -> float:
        """Speedup vs naive (1 target call per token)."""
        return self.mean_accepted_per_step

    def to_dict(self) -> dict:
        return {
            "tokens_generated":       self.n_tokens_generated,
            "target_model_calls":     self.n_target_calls,
            "mean_accepted_per_step": round(self.mean_accepted_per_step, 2),
            "acceptance_rate":        round(self.acceptance_rate, 3),
            "speedup_factor":         round(self.speedup, 2),
            "tokens_per_second":      round(self.tokens_per_second, 1),
        }


class SpeculativeDecoder:
    """
    Speculative decoding generation engine.

    Uses a fast draft model + slow target model to generate text
    with target model quality at draft model speed.

    Args:
        target_model:  Large, accurate model. Must support
                       ``forward(input_ids) → (logits, ...)`` returning
                       ``(B, T, V)`` logits.
        draft_model:   Small, fast :class:`DraftModel`.
        k:             Number of draft tokens per step.
        temperature:   Sampling temperature.
        top_p:         Nucleus sampling p.

    Example::

        engine = SpeculativeDecoder(target, draft, k=4)
        ids    = engine.generate(input_ids, max_new_tokens=64)
        stats  = engine.stats
        print(f"Speedup: {stats.speedup:.1f}x")
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model:  DraftModel,
        k:            int   = 4,
        temperature:  float = 1.0,
        top_p:        float = 1.0,
    ) -> None:
        self.target     = target_model
        self.draft      = draft_model
        self.k          = k
        self.sampler    = SpeculativeSampler(temperature, top_p)
        self.stats      = GenerationStats()

    def _target_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get target model logits for all positions."""
        with torch.no_grad():
            out = self.target(input_ids)
            if isinstance(out, tuple):
                out = out[0]
            return out   # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids:      ``(B, T)`` context token IDs.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            ``(B, T + new)`` token IDs including generated tokens.
        """
        self.stats = GenerationStats()
        t0         = time.monotonic()

        current    = input_ids.clone()
        generated  = 0

        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            k         = min(self.k, remaining)

            # Step 1: Draft K tokens
            draft_ids, draft_logits = self.draft.draft(current, k)
            self.stats.n_draft_calls += 1
            self.stats.total_drafted += k

            # Step 2: Target model scores context + K draft tokens
            target_input  = torch.cat([current, draft_ids], dim=1)
            target_logits = self._target_logits(target_input)
            # Logits at positions T..T+K (for verifying draft tokens)
            T             = current.shape[1]
            verify_logits = target_logits[:, T-1: T+k, :]   # (B, K+1, V)
            self.stats.n_target_calls += 1

            # Step 3: Accept/reject
            result = self.sampler.verify(draft_ids, draft_logits, verify_logits)
            self.stats.acceptance_rates.append(result.acceptance_rate)
            self.stats.total_accepted += result.n_accepted

            # Step 4: Append accepted tokens
            current   = torch.cat([current, result.accepted_ids], dim=1)
            generated += result.accepted_ids.shape[1]
            self.stats.n_tokens_generated += result.accepted_ids.shape[1]

        self.stats.wall_time_s = time.monotonic() - t0
        return current

    @torch.no_grad()
    def generate_naive(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Naive autoregressive generation (baseline, no speculation).

        Used for comparison to measure speedup.
        """
        current = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self._target_logits(current)[:, -1, :]   # (B, V)
            probs  = F.softmax(logits / self.sampler.temperature, dim=-1)
            next_t = torch.multinomial(probs + 1e-10, 1)
            current = torch.cat([current, next_t], dim=1)
        return current
