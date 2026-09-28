"""
nanomind/specd/draft.py — Draft model interface for speculative decoding.

## Speculative Decoding (Chen et al., 2023; Leviathan et al., 2023)

LLM inference bottleneck: each token requires a full forward pass.
For a 70B model: ~10ms per token → 100 tok/s max.

Speculative decoding uses two models:
  - Draft model (small, fast): generates K candidate tokens quickly
  - Target model (large, accurate): verifies all K tokens in ONE pass

Key insight: target model can verify K tokens in the same time it generates 1!
(Because attention is parallelisable over existing tokens)

Algorithm:
  1. Draft model generates K tokens: x̃_{n+1}, ..., x̃_{n+K}
  2. Target model scores ALL K tokens in one forward pass
  3. Accept/reject each token based on probability ratio:
     - If p_target(x) >= p_draft(x): always accept
     - Else: accept with probability p_target(x) / p_draft(x)
  4. If token i is rejected: sample correction token from target, discard i+1..K
  5. Result: ~2-3× speedup with IDENTICAL output distribution!

This is rejection sampling — mathematically equivalent to target model sampling.

Speedup depends on acceptance rate α:
  Mean accepted tokens per step = K × α + 1

For code generation (α ≈ 0.9): ~9× speedup theoretically!
For chat (α ≈ 0.7): ~4× speedup.

## Draft Model Options

1. Smaller version of same family (LLaMA-7B drafts for LLaMA-70B)
2. Distilled draft model (trained specifically to match target)
3. N-gram / retrieval model (even faster, no neural network)
4. Self-speculation: target model drafts its own early exits

References:
  Chen et al. (2023) "Accelerating Large Language Model Decoding with
  Speculative Sampling" https://arxiv.org/abs/2302.01318

  Leviathan et al. (2023) "Fast Inference from Transformers via
  Speculative Decoding" https://arxiv.org/abs/2211.17192

  Cai et al. (2024) Medusa: https://arxiv.org/abs/2401.10774
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DraftModel(ABC):
    """
    Abstract draft model interface.

    Draft models propose K candidate tokens per step.
    Any model implementing `draft()` can be used.
    """

    @abstractmethod
    def draft(
        self,
        input_ids: torch.Tensor,
        n_tokens:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate K draft tokens autoregressively.

        Args:
            input_ids: ``(B, T)`` context token IDs.
            n_tokens:  K — number of tokens to draft.

        Returns:
            ``(draft_ids, draft_logits)``
            draft_ids:    ``(B, K)`` draft token IDs
            draft_logits: ``(B, K, V)`` per-token probability distributions
        """
        ...

    @abstractmethod
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get next-token logits for a context.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B, V)`` logits for next token.
        """
        ...


class NgramDraftModel(DraftModel):
    """
    N-gram based draft model: purely retrieval, no parameters.

    Looks up the most common completion of the last N tokens
    from a reference corpus (stored as a dict).

    For demonstration purposes, uses a simple pattern:
      repeat the last token (works surprisingly well for code!)

    Args:
        vocab_size: Vocabulary size.
        n:          N-gram order (context window for lookup).

    Example::

        draft = NgramDraftModel(vocab_size=1000, n=3)
        ids, logits = draft.draft(input_ids, n_tokens=4)
    """

    def __init__(self, vocab_size: int, n: int = 3) -> None:
        self.vocab_size = vocab_size
        self.n          = n
        self._ngrams:   dict = {}   # (ctx_tuple) → most_common_next

    def train_ngrams(self, corpus: list[list[int]]) -> None:
        """Build N-gram table from a list of token sequences."""
        from collections import Counter
        counts: dict = {}
        for seq in corpus:
            for i in range(len(seq) - self.n):
                ctx  = tuple(seq[i: i + self.n])
                next_t = seq[i + self.n]
                if ctx not in counts:
                    counts[ctx] = Counter()
                counts[ctx][next_t] += 1
        self._ngrams = {ctx: ctr.most_common(1)[0][0] for ctx, ctr in counts.items()}

    def draft(
        self,
        input_ids: torch.Tensor,
        n_tokens:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        draft_ids    = torch.zeros(B, n_tokens, dtype=torch.long)
        draft_logits = torch.zeros(B, n_tokens, self.vocab_size)

        for b in range(B):
            current = input_ids[b].tolist()
            for k in range(n_tokens):
                ctx    = tuple(current[-self.n:]) if len(current) >= self.n else tuple(current)
                next_t = self._ngrams.get(ctx, current[-1])   # default: repeat last
                draft_ids[b, k] = next_t
                # Uniform-ish logit with predicted token boosted
                logit = torch.zeros(self.vocab_size)
                logit[next_t] = 3.0   # boosted confidence
                draft_logits[b, k] = logit
                current.append(next_t)

        return draft_ids, draft_logits

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.zeros(B, self.vocab_size)
        for b in range(B):
            ctx    = tuple(input_ids[b, -self.n:].tolist())
            next_t = self._ngrams.get(ctx, input_ids[b, -1].item())
            logits[b, next_t] = 3.0
        return logits


class SmallModelDraft(DraftModel):
    """
    Small transformer draft model.

    Wraps a small LM (e.g. NanoMind core model) as a draft model.

    Args:
        model:  A model with `forward(input_ids) → logits` method.

    Example::

        small   = TinyTransformer(vocab_size=1000, d_model=64)
        draft   = SmallModelDraft(small)
        ids, lp = draft.draft(input_ids, n_tokens=4)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids)
            if isinstance(out, tuple):
                out = out[0]
            return out[:, -1, :]   # (B, V) last token logits

    def draft(
        self,
        input_ids: torch.Tensor,
        n_tokens:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        V    = None
        current      = input_ids.clone()
        draft_ids    = []
        draft_logits = []

        with torch.no_grad():
            for _ in range(n_tokens):
                l    = self.logits(current)        # (B, V)
                if V is None:
                    V = l.shape[-1]
                probs = F.softmax(l, dim=-1)
                next_t = probs.argmax(dim=-1, keepdim=True)   # greedy
                draft_ids.append(next_t)
                draft_logits.append(l.unsqueeze(1))
                current = torch.cat([current, next_t], dim=1)

        return (
            torch.cat(draft_ids, dim=1),           # (B, K)
            torch.cat(draft_logits, dim=1),        # (B, K, V)
        )
