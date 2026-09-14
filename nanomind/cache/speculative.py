"""
nanomind/cache/speculative.py — Speculative decoding (draft-then-verify).

Speculative decoding (Leviathan et al., 2023) uses a small DRAFT model
to propose K tokens, then a large TARGET model verifies them in one pass:

  1. Draft model generates K tokens greedily (cheap, fast)
  2. Target model runs ONE forward pass on all K tokens in parallel
  3. Accept tokens where target agrees with draft; reject the first mismatch
  4. Always guaranteed to match target model's distribution

Speedup: up to K× faster if draft model has high acceptance rate.
Typical: 2-3× speedup with a draft model 10-100× smaller than target.

Used by: Hugging Face TGI, DeepMind's SpS, Google's Medusa.

References:
  Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
  https://arxiv.org/abs/2211.17192

  Chen et al. (2023) "Accelerating Large Language Model Decoding with
  Speculative Sampling" https://arxiv.org/abs/2302.01318
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("cache.speculative")


class SpeculativeDecoder:
    """
    Speculative decoding: draft → verify → accept/reject.

    Args:
        draft_model:  Small, fast language model for token proposals.
        target_model: Large, accurate language model for verification.
        tokenizer:    Shared tokenizer.
        k:            Number of draft tokens per speculative step.

    Example::

        decoder = SpeculativeDecoder(small_lm, large_lm, tokenizer, k=4)
        text    = decoder.generate("Once upon a time", max_new_tokens=50)
    """

    def __init__(
        self,
        draft_model:  nn.Module,
        target_model: nn.Module,
        tokenizer,
        k:            int = 4,
    ) -> None:
        self.draft  = draft_model.eval()
        self.target = target_model.eval()
        self.tok    = tokenizer
        self.k      = k
        self._accepted = 0
        self._total    = 0

    @torch.no_grad()
    def _draft_k(
        self,
        ctx:         torch.Tensor,
        temperature: float = 0.8,
    ) -> list[int]:
        """Generate K draft token IDs greedily."""
        draft_ids   = []
        cur_ctx     = ctx.clone()
        block_sz    = getattr(self.draft, "T", 512)
        for _ in range(self.k):
            crop          = cur_ctx[:, -block_sz:]
            logits, _     = self.draft(crop)
            token_id      = int(logits[0, -1, :].argmax().item())
            draft_ids.append(token_id)
            cur_ctx = torch.cat([cur_ctx, torch.tensor([[token_id]])], dim=1)
        return draft_ids

    @torch.no_grad()
    def _verify(
        self,
        ctx:       torch.Tensor,
        draft_ids: list[int],
    ) -> tuple[list[int], int]:
        """
        Verify draft tokens with the target model.

        Returns:
            ``(accepted_ids, n_accepted)``
        """
        # Run target on ctx + all draft tokens at once
        draft_tensor  = torch.tensor([draft_ids], dtype=torch.long)
        full_ctx      = torch.cat([ctx, draft_tensor], dim=1)
        block_sz      = getattr(self.target, "T", 512)
        crop          = full_ctx[:, -block_sz:]
        logits, _     = self.target(crop)

        # logits[:, -(k+1):-1, :] → target predictions at draft positions
        target_start  = -(len(draft_ids) + 1)
        target_logits = logits[0, target_start:-1, :]  # (K, V)

        accepted = []
        for i, (draft_id, t_logit) in enumerate(zip(draft_ids, target_logits)):
            target_id = int(t_logit.argmax().item())
            if target_id == draft_id:
                accepted.append(draft_id)
            else:
                # Accept up to but not including mismatch, then use target's token
                accepted.append(target_id)
                break

        return accepted, len(accepted)

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 0.8,
    ) -> str:
        """
        Generate text with speculative decoding.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature.

        Returns:
            Generated text string.
        """
        ids       = self.tok.encode(prompt)
        ctx       = torch.tensor([ids], dtype=torch.long)
        generated = []

        while len(generated) < max_new_tokens:
            draft_ids         = self._draft_k(ctx, temperature)
            accepted, n_accept = self._verify(ctx, draft_ids)
            self._accepted    += n_accept
            self._total       += self.k

            for tid in accepted:
                generated.append(tid)
                ctx = torch.cat([ctx, torch.tensor([[tid]])], dim=1)
                if len(generated) >= max_new_tokens:
                    break

        log.info(f"Acceptance rate: {self.acceptance_rate:.1%} "
                 f"({self._accepted}/{self._total})")
        return self.tok.decode(generated[:max_new_tokens])

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the target model."""
        return self._accepted / max(self._total, 1)

    def reset_stats(self) -> None:
        """Reset acceptance rate counters."""
        self._accepted = 0
        self._total    = 0
