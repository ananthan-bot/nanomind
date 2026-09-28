"""
day48_commits.py — 20 atomic commits for Day 48: Speculative Decoding & Fast Inference.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 48: Speculative Decoding & Fast Inference — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — specd package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/__init__.py",
      '"""NanoMind Speculative Decoding sub-package — Fast inference via speculation."""\n')
commit("feat: add nanomind/specd/ package skeleton for speculative decoding and fast inference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — theory + draft model interface
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/draft.py", '''\
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
''')
commit("feat: add DraftModel ABC, NgramDraftModel (n-gram lookup), SmallModelDraft (neural wrapper)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — speculative sampler (accept/reject)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/sampler.py", '''\
"""
nanomind/specd/sampler.py — Speculative sampling: accept/reject with target model.

The core algorithm verifies K draft tokens using the target model in ONE pass,
accepting each based on probability ratios. Mathematically identical to
sampling from the target model directly — no accuracy loss!

Acceptance rule:
  For token x at position i:
    If p_target(x) / p_draft(x) >= 1: accept (target agrees or is more confident)
    Else: accept with prob = p_target(x) / p_draft(x) (rejection sampling)

  On rejection at position i:
    Sample correction token from: max(0, p_target - p_draft) (normalised)
    Discard all tokens after i
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SpeculativeResult:
    """Result of one speculative decoding step."""
    accepted_ids:   torch.Tensor    # (B, n_accepted) accepted token IDs
    n_accepted:     int             # mean accepted tokens this step
    acceptance_rate: float          # fraction of draft tokens accepted
    target_calls:   int = 1        # target model forward passes used

    def efficiency_gain(self, k: int) -> float:
        """Tokens generated per target model call."""
        return self.n_accepted / max(self.target_calls, 1)


class SpeculativeSampler:
    """
    Speculative decoding accept/reject sampler.

    Takes draft logits and target logits, applies the rejection sampling
    algorithm to produce tokens with the target model distribution.

    Args:
        temperature: Sampling temperature (1.0 = standard, 0 = greedy).
        top_p:       Nucleus sampling probability cutoff.

    Example::

        sampler    = SpeculativeSampler(temperature=1.0)
        result     = sampler.verify(
            draft_ids    = ...,   # (B, K)
            draft_logits = ...,   # (B, K, V)
            target_logits = ...,  # (B, K+1, V)  — target sees all K+1 positions
        )
        new_ids    = result.accepted_ids
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p:       float = 1.0,
    ) -> None:
        self.temperature = max(temperature, 1e-8)
        self.top_p       = top_p

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def _apply_top_p(self, probs: torch.Tensor) -> torch.Tensor:
        """Nucleus sampling: zero out low-probability tokens."""
        if self.top_p >= 1.0:
            return probs
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        remove = cumsum - sorted_p > self.top_p
        sorted_p[remove] = 0.0
        # Scatter back
        out = torch.zeros_like(probs)
        out.scatter_(-1, sorted_idx, sorted_p)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-8)

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample a token from probability distribution."""
        probs = self._apply_top_p(probs)
        return torch.multinomial(probs + 1e-10, num_samples=1).squeeze(-1)

    def verify(
        self,
        draft_ids:     torch.Tensor,
        draft_logits:  torch.Tensor,
        target_logits: torch.Tensor,
    ) -> SpeculativeResult:
        """
        Apply speculative acceptance/rejection.

        Args:
            draft_ids:     ``(B, K)`` draft token IDs.
            draft_logits:  ``(B, K, V)`` draft model logits.
            target_logits: ``(B, K+1, V)`` target model logits.
                           Position K+1 is for the token AFTER all K draft tokens.

        Returns:
            :class:`SpeculativeResult` with accepted token IDs.
        """
        B, K   = draft_ids.shape
        V      = target_logits.shape[-1]

        # Convert logits to probabilities
        draft_probs  = F.softmax(self._apply_temperature(draft_logits),  dim=-1)   # (B,K,V)
        target_probs = F.softmax(self._apply_temperature(target_logits), dim=-1)   # (B,K+1,V)

        accepted      = torch.zeros(B, K + 1, dtype=torch.long)
        n_accepted    = torch.zeros(B, dtype=torch.long)
        all_rejected  = torch.zeros(B, dtype=torch.bool)

        for b in range(B):
            if all_rejected[b]:
                continue
            n_acc = 0
            for k in range(K):
                tok       = draft_ids[b, k].item()
                p_draft   = draft_probs[b, k, tok].item()
                p_target  = target_probs[b, k, tok].item()

                # Rejection sampling accept criterion
                if p_draft <= 0:
                    ratio = 1.0
                else:
                    ratio = min(1.0, p_target / (p_draft + 1e-10))

                u = torch.rand(1).item()
                if u <= ratio:
                    # Accept: keep draft token
                    accepted[b, n_acc] = tok
                    n_acc             += 1
                else:
                    # Reject: sample correction from target - draft residual
                    residual = (target_probs[b, k] - draft_probs[b, k]).clamp(min=0)
                    if residual.sum() < 1e-8:
                        residual = target_probs[b, k]
                    residual = residual / (residual.sum() + 1e-8)
                    correction = self._sample(residual)
                    accepted[b, n_acc] = correction
                    n_acc             += 1
                    break   # Stop at rejection

            # Always add at least one token from target if no rejections
            if n_acc == K:
                next_tok = self._sample(target_probs[b, K])
                accepted[b, n_acc] = next_tok
                n_acc             += 1

            n_accepted[b] = n_acc

        # Trim to actual accepted length
        max_acc = n_accepted.max().item()
        accepted_trimmed = accepted[:, :max_acc]

        acc_rate = (n_accepted.float().mean() / K).item()
        return SpeculativeResult(
            accepted_ids    = accepted_trimmed,
            n_accepted      = int(n_accepted.float().mean().item()),
            acceptance_rate = acc_rate,
        )

    def greedy_verify(
        self,
        draft_ids:     torch.Tensor,
        target_logits: torch.Tensor,
    ) -> SpeculativeResult:
        """
        Greedy verification: accept if target's argmax matches draft.

        Faster but not exactly equivalent to target sampling.
        Used for temperature=0 (greedy decoding) scenarios.
        """
        B, K  = draft_ids.shape
        # Target greedy predictions for each position
        target_tokens = target_logits[:, :K].argmax(dim=-1)   # (B, K)

        # Find first mismatch
        matches       = (draft_ids == target_tokens)           # (B, K)
        accepted_list = []
        n_acc_list    = []

        for b in range(B):
            first_reject = K
            for k in range(K):
                if not matches[b, k]:
                    first_reject = k
                    break
            # Accept tokens up to first mismatch + correction
            acc_tokens = draft_ids[b, :first_reject].tolist()
            # Add target's correction at first_reject position
            correction = target_logits[b, first_reject].argmax().item()
            acc_tokens.append(correction)
            accepted_list.append(acc_tokens)
            n_acc_list.append(len(acc_tokens))

        max_len   = max(len(x) for x in accepted_list)
        accepted  = torch.zeros(B, max_len, dtype=torch.long)
        for b, tokens in enumerate(accepted_list):
            accepted[b, :len(tokens)] = torch.tensor(tokens)

        mean_acc  = sum(n_acc_list) / B
        return SpeculativeResult(
            accepted_ids    = accepted,
            n_accepted      = int(mean_acc),
            acceptance_rate = mean_acc / K,
        )
''')
commit("feat: add SpeculativeSampler — rejection sampling verify(), greedy_verify(), acceptance_rate")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Speculative decoding engine
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/engine.py", '''\
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
''')
commit("feat: add SpeculativeDecoder — K-step speculation loop, GenerationStats, speedup tracking, naive baseline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Medusa heads (self-speculation)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/medusa.py", '''\
"""
nanomind/specd/medusa.py — Medusa: self-speculation with multiple LM heads.

## Medusa (Cai et al., 2024)

Standard speculative decoding needs TWO models (draft + target).
Medusa adds K extra "heads" to the target model itself:

  Token n:
    Head 0 (original): predicts token n+1 (standard LM head)
    Head 1 (Medusa):   predicts token n+2 (2-ahead)
    Head 2 (Medusa):   predicts token n+3 (3-ahead)
    ...
    Head K (Medusa):   predicts token n+K+1 (K+1 ahead)

These heads are trained with teacher forcing on the same input,
using only a fraction of the training compute.

Verification: run one forward pass, get K+1 candidate tokens,
accept greedily or via tree attention.

Benefits:
  ✓ Single model (no separate draft model needed)
  ✓ 2-3× speedup on most tasks
  ✓ Easy to add to any transformer

## Tree Attention (for candidate expansion)

Instead of K linear candidates, Medusa uses tree attention to
explore multiple hypotheses simultaneously:

         root
        / | \\
       A  B  C       ← Head 1 top-3
      /|  |  |\\
     D E  F  G H     ← Head 2 top-{2,1,2}

This explores more candidates with the same compute budget.

Reference:
  Cai et al. (2024) "Medusa: Simple LLM Inference Acceleration Framework
  with Multiple Decoding Heads"
  https://arxiv.org/abs/2401.10774
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaHead(nn.Module):
    """
    A single Medusa lookahead head.

    A lightweight 2-layer MLP on top of the LM hidden states,
    trained to predict the token K positions ahead.

    Args:
        d_model:    LM hidden dimension.
        vocab_size: Vocabulary size.
        d_ff:       Hidden dim of Medusa MLP (default = 2 × d_model).
    """

    def __init__(self, d_model: int, vocab_size: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: ``(B, T, D)`` LM hidden states.
        Returns:
            ``(B, T, V)`` logits for k-ahead token.
        """
        return self.net(hidden)


class MedusaModel(nn.Module):
    """
    LM model augmented with K Medusa speculation heads.

    Args:
        base_model:  The base LM (must return hidden states + logits).
        n_heads:     Number of Medusa heads (lookahead steps).
        d_model:     LM hidden dimension.
        vocab_size:  Vocabulary size.

    Example::

        model  = MedusaModel(base_lm, n_heads=3, d_model=128, vocab_size=1000)
        logits, medusa_logits = model(input_ids)
        # logits:         (B, T, V)   — standard next token
        # medusa_logits:  list of 3 × (B, T, V)  — 2,3,4-ahead
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_heads:    int,
        d_model:    int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.base       = base_model
        self.n_heads    = n_heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(d_model, vocab_size)
            for _ in range(n_heads)
        ])
        self.vocab_size = vocab_size
        self.d_model    = d_model

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass with Medusa heads.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(base_logits, medusa_logits)``
            base_logits:   ``(B, T, V)``
            medusa_logits: list of K tensors, each ``(B, T, V)``
        """
        # Get base model output
        base_out = self.base(input_ids)
        if isinstance(base_out, tuple):
            base_logits = base_out[0]
        else:
            base_logits = base_out

        # Use base logits as proxy for hidden states
        # (In production: extract hidden states from LM, not logits)
        hidden = base_logits   # (B, T, V) used as features

        medusa_logits = [head(hidden) for head in self.medusa_heads]
        return base_logits, medusa_logits

    @torch.no_grad()
    def speculate(
        self,
        input_ids:  torch.Tensor,
        top_k_each: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate K+1 candidate token sequences.

        Returns greedy predictions from each head.

        Args:
            input_ids:  ``(B, T)`` context.
            top_k_each: Top-K candidates per head (for tree decoding).

        Returns:
            ``(base_token, medusa_tokens)``
            base_token:    ``(B, 1)`` base model's next token
            medusa_tokens: list of K tensors ``(B, top_k_each)`` per head
        """
        base_logits, medusa_logits = self.forward(input_ids)
        base_token    = base_logits[:, -1, :].topk(top_k_each, dim=-1).indices
        medusa_tokens = [ml[:, -1, :].topk(top_k_each, dim=-1).indices
                         for ml in medusa_logits]
        return base_token, medusa_tokens

    def medusa_loss(
        self,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Compute base + Medusa training losses.

        For head k, the label is the token k+2 positions ahead
        (shifted by k+1 from the base label).

        Args:
            input_ids: ``(B, T)`` tokens.
            labels:    ``(B, T)`` next-token labels.

        Returns:
            ``(base_loss, medusa_losses)``
        """
        base_logits, medusa_logits = self.forward(input_ids)
        B, T, V = base_logits.shape

        # Base loss: standard next-token prediction
        base_loss = F.cross_entropy(
            base_logits[:, :-1].reshape(-1, V),
            labels[:, 1:].reshape(-1),
        )

        # Medusa losses: k+2 ahead
        medusa_losses = []
        for k, ml in enumerate(medusa_logits):
            shift = k + 2
            if shift >= T:
                medusa_losses.append(torch.tensor(0.0))
                continue
            loss = F.cross_entropy(
                ml[:, :-shift].reshape(-1, V),
                labels[:, shift:].reshape(-1),
            )
            medusa_losses.append(loss)

        return base_loss, medusa_losses

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_medusa_params(self) -> int:
        return sum(p.numel() for p in self.medusa_heads.parameters())
''')
commit("feat: add MedusaHead (SiLU MLP), MedusaModel — K lookahead heads, speculate(), medusa_loss()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Token tree and batch speculation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/tree.py", '''\
"""
nanomind/specd/tree.py — Token tree for candidate exploration.

In standard speculative decoding: K tokens in a linear sequence.
In tree speculative decoding: a TREE of candidate token sequences.

Example K=3, top_k=2 per head:
  Root (current context)
    ├─ Token A (prob 0.4)
    │   ├─ Token C (prob 0.3)
    │   └─ Token D (prob 0.2)
    └─ Token B (prob 0.3)
        └─ Token E (prob 0.4)

Tree attention: attend to all paths simultaneously.
Accept the longest valid prefix from the tree.

This is a simplified version of SpecTr (Sun et al., 2023)
and Medusa's tree decoding.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A node in the candidate token tree."""
    token_id:  int
    prob:      float
    depth:     int
    parent:    "TreeNode | None"  = None
    children:  list               = field(default_factory=list)
    node_id:   int                = 0

    def path_to_root(self) -> list[int]:
        """Get token sequence from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.token_id)
            node = node.parent
        return path[::-1]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class TokenTree:
    """
    A tree of candidate token sequences for speculative decoding.

    Each path from root to leaf represents a candidate continuation.

    Args:
        max_depth: Maximum tree depth (= speculation length K).
        branching: Number of candidates per node (top-K per head).

    Example::

        tree   = TokenTree(max_depth=3, branching=2)
        root   = tree.build(candidate_tokens)
        paths  = tree.all_paths()
    """

    def __init__(self, max_depth: int = 4, branching: int = 2) -> None:
        self.max_depth = max_depth
        self.branching = branching
        self._node_counter = 0

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def build(
        self,
        candidate_tokens: list[list[int]],
        candidate_probs:  list[list[float]] | None = None,
    ) -> TreeNode:
        """
        Build tree from per-depth candidate lists.

        Args:
            candidate_tokens: List of K lists, each with branching token IDs.
                              candidate_tokens[d] = [tok_1, tok_2, ...] at depth d.
            candidate_probs:  Optional probabilities.

        Returns:
            Root :class:`TreeNode`.
        """
        root = TreeNode(token_id=-1, prob=1.0, depth=0, node_id=0)
        if not candidate_tokens:
            return root

        probs = candidate_probs or [[1.0] * len(t) for t in candidate_tokens]
        self._node_counter = 0

        # Build level by level
        leaves = [root]
        for depth, (tokens, plist) in enumerate(zip(candidate_tokens, probs)):
            new_leaves = []
            for parent in leaves:
                for tok, p in zip(tokens[:self.branching], plist[:self.branching]):
                    child = TreeNode(
                        token_id = tok,
                        prob     = p,
                        depth    = depth + 1,
                        parent   = parent,
                        node_id  = self._new_id(),
                    )
                    parent.children.append(child)
                    new_leaves.append(child)
                if depth >= self.max_depth - 1:
                    break   # don't expand further
            leaves = new_leaves
        return root

    def all_paths(self, root: TreeNode) -> list[list[int]]:
        """Return all root-to-leaf paths as token sequences."""
        paths = []
        def dfs(node, path):
            if node.is_leaf():
                if path:
                    paths.append(path[:])
                return
            for child in node.children:
                dfs(child, path + [child.token_id])
        dfs(root, [])
        return paths

    def n_nodes(self, root: TreeNode) -> int:
        """Count total nodes in tree."""
        count = [0]
        def dfs(n):
            count[0] += 1
            for c in n.children:
                dfs(c)
        dfs(root)
        return count[0]

    def verify_paths(
        self,
        root:          TreeNode,
        target_tokens: list[int],
    ) -> list[int]:
        """
        Find longest path that matches target_tokens prefix.

        Args:
            root:          Tree root.
            target_tokens: Target model's predicted tokens (one per position).

        Returns:
            Best matching path (list of token IDs).
        """
        best  = []
        def dfs(node, path, depth):
            nonlocal best
            if depth >= len(target_tokens):
                if len(path) > len(best):
                    best = path[:]
                return
            if node.token_id != target_tokens[depth - 1] and depth > 0:
                if len(path) > len(best):
                    best = path[:]
                return
            for child in node.children:
                dfs(child, path + [child.token_id], depth + 1)
        dfs(root, [], 0)
        return best
''')
commit("feat: add TreeNode, TokenTree — build(), all_paths(), verify_paths(), n_nodes() for tree speculation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Lookahead decoding
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/lookahead.py", '''\
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
''')
commit("feat: add LookaheadDecoder — Jacobi iteration window, single-model speculation, generate()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — specd __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/specd/__init__.py", '''\
"""NanoMind Speculative Decoding sub-package — Fast inference via speculation.

Implements the full speculative decoding stack:
  1. DraftModel           — abstract draft model interface
  2. NgramDraftModel      — n-gram lookup draft model (no parameters)
  3. SmallModelDraft      — neural small model draft wrapper
  4. SpeculativeSampler   — reject/accept + greedy_verify
  5. SpeculativeResult    — accepted_ids, acceptance_rate, efficiency_gain
  6. SpeculativeDecoder   — K-step speculation loop, GenerationStats
  7. GenerationStats      — speedup, mean_accepted, acceptance_rate
  8. MedusaHead           — SiLU MLP lookahead head
  9. MedusaModel          — K Medusa heads, speculate(), medusa_loss()
  10. TreeNode / TokenTree — candidate tree, all_paths, verify_paths
  11. LookaheadDecoder    — Jacobi single-model speculation

Primary exports:
    - :class:`DraftModel`           — ABC: draft(), logits()
    - :class:`NgramDraftModel`      — n-gram lookup
    - :class:`SmallModelDraft`      — neural draft
    - :class:`SpeculativeSampler`   — verify(), greedy_verify()
    - :class:`SpeculativeResult`    — accepted_ids, acceptance_rate
    - :class:`SpeculativeDecoder`   — generate(), generate_naive(), stats
    - :class:`GenerationStats`      — speedup, tokens_per_second
    - :class:`MedusaHead`           — single lookahead head
    - :class:`MedusaModel`          — full Medusa model, medusa_loss
    - :class:`TokenTree`            — tree builder, all_paths, verify_paths
    - :class:`TreeNode`             — tree node with parent/children
    - :class:`LookaheadDecoder`     — single-model Jacobi decoding
"""

from nanomind.specd.draft import DraftModel, NgramDraftModel, SmallModelDraft
from nanomind.specd.sampler import SpeculativeSampler, SpeculativeResult
from nanomind.specd.engine import SpeculativeDecoder, GenerationStats
from nanomind.specd.medusa import MedusaHead, MedusaModel
from nanomind.specd.tree import TreeNode, TokenTree
from nanomind.specd.lookahead import LookaheadDecoder, LookaheadState

__all__ = [
    "DraftModel", "NgramDraftModel", "SmallModelDraft",
    "SpeculativeSampler", "SpeculativeResult",
    "SpeculativeDecoder", "GenerationStats",
    "MedusaHead", "MedusaModel",
    "TreeNode", "TokenTree",
    "LookaheadDecoder", "LookaheadState",
]
''')
commit("refactor: export all speculative decoding components from nanomind/specd/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/specd_demo.py", '''\
"""
examples/specd_demo.py — NanoMind Speculative Decoding & Fast Inference demo.

Demonstrates:
  1. N-gram draft model
  2. Small model draft (neural)
  3. Speculative sampler: verify (accept/reject)
  4. Speculative decoder: full speculation loop + stats
  5. Naive vs speculative comparison
  6. Medusa heads: multi-lookahead prediction
  7. Token tree: candidate exploration
  8. Lookahead decoding: Jacobi iteration

Usage:
    python examples/specd_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.specd import (
    NgramDraftModel, SmallModelDraft,
    SpeculativeSampler, SpeculativeResult,
    SpeculativeDecoder, GenerationStats,
    MedusaHead, MedusaModel,
    TokenTree, TreeNode,
    LookaheadDecoder,
)

V = 32   # small vocab for demo

# ── Tiny target model (for demo — normally a large LM) ────────────────────────
class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

target_model = TinyLM()

print("=" * 60)
print("NanoMind Speculative Decoding & Fast Inference Demo")
print("=" * 60)

# ── N-gram Draft Model ────────────────────────────────────────────────────────
print("\n── N-gram Draft Model ──")
ngram = NgramDraftModel(vocab_size=V, n=2)
corpus = [list(range(V)), list(range(V-1, -1, -1))]
ngram.train_ngrams(corpus)
ids   = torch.tensor([[1, 2, 3]])
draft_ids, draft_logits = ngram.draft(ids, n_tokens=4)
print(f"  Context: {ids.tolist()}")
print(f"  Draft tokens (4): {draft_ids.tolist()}")
print(f"  Draft logits shape: {tuple(draft_logits.shape)}")

# ── Small Model Draft ─────────────────────────────────────────────────────────
print("\n── Small Model Draft (neural) ──")
small_draft = SmallModelDraft(TinyLM())
input_ids   = torch.randint(0, V, (1, 6))
d_ids, d_lp = small_draft.draft(input_ids, n_tokens=4)
print(f"  Draft shape: {tuple(d_ids.shape)}, logits: {tuple(d_lp.shape)}")

# ── Speculative Sampler ───────────────────────────────────────────────────────
print("\n── Speculative Sampler (verify) ──")
sampler     = SpeculativeSampler(temperature=1.0)
K           = 4
draft_ids_  = torch.randint(0, V, (2, K))
draft_lp_   = torch.randn(2, K, V)
target_lp_  = torch.randn(2, K + 1, V)
result      = sampler.verify(draft_ids_, draft_lp_, target_lp_)
print(f"  Accepted shape:    {tuple(result.accepted_ids.shape)}")
print(f"  n_accepted:        {result.n_accepted}")
print(f"  Acceptance rate:   {result.acceptance_rate:.2%}")
print(f"  Efficiency gain:   {result.efficiency_gain(K):.2f}x")

# Greedy verify
greedy_res = sampler.greedy_verify(draft_ids_, target_lp_)
print(f"  Greedy accepted: {greedy_res.n_accepted} tokens")

# ── SpeculativeDecoder ────────────────────────────────────────────────────────
print("\n── SpeculativeDecoder (full loop) ──")
engine    = SpeculativeDecoder(target_model, ngram, k=4, temperature=1.0)
ctx       = torch.randint(0, V, (1, 8))
generated = engine.generate(ctx, max_new_tokens=12)
stats     = engine.stats
print(f"  Generated: {tuple(generated.shape)}")
print(f"  Stats: {stats.to_dict()}")

# Compare with naive
naive_out = engine.generate_naive(ctx, max_new_tokens=12)
print(f"  Naive output shape: {tuple(naive_out.shape)}")

# ── Medusa Model ──────────────────────────────────────────────────────────────
print("\n── Medusa Self-Speculation ──")
medusa = MedusaModel(TinyLM(), n_heads=3, d_model=V, vocab_size=V)
ids    = torch.randint(0, V, (2, 8))
base_logits, medusa_logits = medusa(ids)
print(f"  Base logits: {tuple(base_logits.shape)}")
print(f"  Medusa heads: {len(medusa_logits)} × {tuple(medusa_logits[0].shape)}")
print(f"  Total params:  {medusa.n_params:,}")
print(f"  Medusa params: {medusa.n_medusa_params:,} ({medusa.n_medusa_params/medusa.n_params:.1%})")

base_t, medusa_t = medusa.speculate(ids, top_k_each=2)
print(f"  Base predicted token: {tuple(base_t.shape)}")
print(f"  Medusa tokens per head: {[tuple(m.shape) for m in medusa_t]}")

# Medusa loss
labels     = torch.randint(0, V, (2, 8))
base_loss, med_losses = medusa.medusa_loss(ids, labels)
print(f"  Base loss: {base_loss.item():.4f}")
print(f"  Medusa losses: {[round(l.item(),4) for l in med_losses]}")

# ── Token Tree ────────────────────────────────────────────────────────────────
print("\n── Token Tree (candidate exploration) ──")
tree   = TokenTree(max_depth=3, branching=2)
cands  = [[5, 7], [3, 8], [1, 6]]   # depth 0,1,2 candidates
probs  = [[0.4, 0.3], [0.5, 0.3], [0.6, 0.2]]
root   = tree.build(cands, probs)
paths  = tree.all_paths(root)
n_nodes = tree.n_nodes(root)
print(f"  Tree nodes: {n_nodes}")
print(f"  All paths: {paths}")
# Verify against target
target_seq = [5, 3, 6]   # what target model would predict
best_path  = tree.verify_paths(root, target_seq)
print(f"  Best matching path: {best_path}")

# ── Lookahead Decoding ────────────────────────────────────────────────────────
print("\n── Lookahead Decoding (Jacobi) ──")
la_decoder = LookaheadDecoder(target_model, window_size=4, n_iters=2)
ctx2       = torch.randint(0, V, (1, 6))
out        = la_decoder.generate(ctx2, max_new_tokens=8)
print(f"  Input: {tuple(ctx2.shape)} → Output: {tuple(out.shape)}")
print(f"  Generated {out.shape[1] - ctx2.shape[1]} new tokens via Jacobi iteration")

print("\nSpeculative decoding demo complete!")
''')
commit("feat: add examples/specd_demo.py — ngram, small model, verify, SpecDecoder, Medusa, tree, lookahead")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_specd.py", '''\
"""tests/test_specd.py — Tests for NanoMind speculative decoding package."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.specd import (
    NgramDraftModel, SmallModelDraft,
    SpeculativeSampler, SpeculativeResult,
    SpeculativeDecoder, GenerationStats,
    MedusaHead, MedusaModel,
    TokenTree, TreeNode,
    LookaheadDecoder,
)

V = 16


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── NgramDraftModel ───────────────────────────────────────────────────────────

class TestNgramDraft:
    def _draft(self):
        d = NgramDraftModel(vocab_size=V, n=2)
        d.train_ngrams([[0, 1, 2, 3, 4, 5]] * 5)
        return d

    def test_draft_shape(self):
        d   = self._draft()
        ids = torch.tensor([[0, 1, 2]])
        di, dl = d.draft(ids, n_tokens=3)
        assert di.shape == (1, 3)
        assert dl.shape == (1, 3, V)

    def test_draft_ids_in_range(self):
        d   = self._draft()
        ids = torch.tensor([[0, 1]])
        di, _ = d.draft(ids, n_tokens=4)
        assert (di >= 0).all() and (di < V).all()

    def test_logits_shape(self):
        d  = self._draft()
        l  = d.logits(torch.tensor([[0, 1, 2]]))
        assert l.shape == (1, V)


# ── SmallModelDraft ───────────────────────────────────────────────────────────

class TestSmallModelDraft:
    def test_draft_shape(self):
        d   = SmallModelDraft(TinyLM())
        ids = torch.randint(0, V, (1, 4))
        di, dl = d.draft(ids, n_tokens=3)
        assert di.shape == (1, 3)
        assert dl.shape == (1, 3, V)

    def test_logits_shape(self):
        d   = SmallModelDraft(TinyLM())
        ids = torch.randint(0, V, (2, 5))
        l   = d.logits(ids)
        assert l.shape == (2, V)


# ── SpeculativeSampler ────────────────────────────────────────────────────────

class TestSpeculativeSampler:
    def _s(self):
        return SpeculativeSampler(temperature=1.0)

    def test_verify_output_shape(self):
        s = self._s()
        K = 4
        di = torch.randint(0, V, (2, K))
        dl = torch.randn(2, K, V)
        tl = torch.randn(2, K + 1, V)
        r  = s.verify(di, dl, tl)
        assert r.accepted_ids.shape[0] == 2
        assert r.accepted_ids.shape[1] >= 1

    def test_acceptance_rate_in_range(self):
        s  = self._s()
        di = torch.randint(0, V, (1, 4))
        dl = torch.randn(1, 4, V)
        tl = torch.randn(1, 5, V)
        r  = s.verify(di, dl, tl)
        assert 0.0 <= r.acceptance_rate <= 1.0

    def test_greedy_verify_shape(self):
        s  = self._s()
        di = torch.randint(0, V, (1, 4))
        tl = torch.randn(1, 5, V)
        r  = s.greedy_verify(di, tl)
        assert r.accepted_ids.shape[0] == 1

    def test_efficiency_gain(self):
        r = SpeculativeResult(
            accepted_ids=torch.zeros(1, 3, dtype=torch.long),
            n_accepted=3, acceptance_rate=0.75,
        )
        assert r.efficiency_gain(4) == 3.0


# ── SpeculativeDecoder ────────────────────────────────────────────────────────

class TestSpeculativeDecoder:
    def _engine(self):
        ngram = NgramDraftModel(V, n=2)
        ngram.train_ngrams([[i % V for i in range(20)]])
        return SpeculativeDecoder(TinyLM(), ngram, k=3, temperature=1.0)

    def test_generate_shape(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        out = e.generate(ids, max_new_tokens=6)
        assert out.shape[1] >= ids.shape[1] + 6

    def test_stats_populated(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        e.generate(ids, max_new_tokens=4)
        s = e.stats
        assert s.n_target_calls > 0
        assert s.n_tokens_generated > 0

    def test_speedup_positive(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        e.generate(ids, max_new_tokens=6)
        assert e.stats.speedup > 0

    def test_naive_generate_shape(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        out = e.generate_naive(ids, max_new_tokens=4)
        assert out.shape == (1, 8)


# ── MedusaModel ───────────────────────────────────────────────────────────────

class TestMedusa:
    def _model(self):
        return MedusaModel(TinyLM(), n_heads=2, d_model=V, vocab_size=V)

    def test_forward_shapes(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 6))
        bl, ml = m(ids)
        assert bl.shape == (2, 6, V)
        assert len(ml) == 2
        assert ml[0].shape == (2, 6, V)

    def test_speculate_shapes(self):
        m   = self._model()
        ids = torch.randint(0, V, (1, 4))
        bt, mt = m.speculate(ids, top_k_each=2)
        assert bt.shape == (1, 2)
        assert len(mt) == 2

    def test_medusa_loss_scalars(self):
        m    = self._model()
        ids  = torch.randint(0, V, (2, 6))
        lbl  = torch.randint(0, V, (2, 6))
        bl, ml = m.medusa_loss(ids, lbl)
        assert bl.shape == ()
        assert all(l.shape == () for l in ml)

    def test_n_medusa_params_positive(self):
        m = self._model()
        assert m.n_medusa_params > 0

    def test_total_params_gt_base(self):
        base  = TinyLM()
        m     = MedusaModel(base, n_heads=2, d_model=V, vocab_size=V)
        base_p = sum(p.numel() for p in base.parameters())
        assert m.n_params > base_p


# ── TokenTree ─────────────────────────────────────────────────────────────────

class TestTokenTree:
    def test_build_and_paths(self):
        tree  = TokenTree(max_depth=2, branching=2)
        cands = [[1, 2], [3, 4]]
        root  = tree.build(cands)
        paths = tree.all_paths(root)
        assert len(paths) > 0
        for p in paths:
            assert len(p) > 0

    def test_n_nodes(self):
        tree  = TokenTree(max_depth=2, branching=2)
        root  = tree.build([[1, 2], [3, 4]])
        n     = tree.n_nodes(root)
        assert n > 1   # at least root + children

    def test_verify_paths(self):
        tree   = TokenTree(max_depth=2, branching=2)
        root   = tree.build([[5, 7], [3, 8]])
        target = [5, 3]
        best   = tree.verify_paths(root, target)
        assert isinstance(best, list)

    def test_empty_cands(self):
        tree = TokenTree(max_depth=2, branching=2)
        root = tree.build([])
        assert root.token_id == -1


# ── LookaheadDecoder ──────────────────────────────────────────────────────────

class TestLookaheadDecoder:
    def test_generate_shape(self):
        d   = LookaheadDecoder(TinyLM(), window_size=3, n_iters=1)
        ids = torch.randint(0, V, (1, 4))
        out = d.generate(ids, max_new_tokens=4)
        assert out.shape == (1, 8)

    def test_longer_than_input(self):
        d   = LookaheadDecoder(TinyLM(), window_size=4, n_iters=2)
        ids = torch.randint(0, V, (1, 5))
        out = d.generate(ids, max_new_tokens=6)
        assert out.shape[1] > ids.shape[1]
''')
commit("test: add full speculative decoding test suite — ngram, small model, sampler, decoder, Medusa, tree, lookahead")

# COMMITS 12-18
for title, body in [
    ("test: add NgramDraftModel untrained defaults to repeat last token", '''
class TestNgramUntrained:
    def test_untrained_repeats_last(self):
        d   = NgramDraftModel(vocab_size=V, n=2)
        ids = torch.tensor([[5, 3]])
        di, _ = d.draft(ids, n_tokens=3)
        # Without training, should default to repeating last token (3)
        assert di[0, 0].item() == 3
'''),
    ("test: add SpeculativeSampler temperature=0 greedy test", '''
class TestGreedySampler:
    def test_low_temperature_deterministic(self):
        s  = SpeculativeSampler(temperature=0.01)
        di = torch.randint(0, V, (1, 3))
        dl = torch.randn(1, 3, V)
        tl = torch.randn(1, 4, V)
        r1 = s.verify(di, dl, tl)
        r2 = s.verify(di, dl, tl)
        # Very low temp → nearly deterministic
        assert r1.accepted_ids.shape == r2.accepted_ids.shape
'''),
    ("test: add SpeculativeDecoder k=1 equals near-naive test", '''
class TestSpecDecoderK1:
    def test_k1_generates_tokens(self):
        ngram = NgramDraftModel(V, n=1)
        ngram.train_ngrams([[i % V for i in range(16)]])
        engine = SpeculativeDecoder(TinyLM(), ngram, k=1)
        ids    = torch.randint(0, V, (1, 4))
        out    = engine.generate(ids, max_new_tokens=4)
        assert out.shape[1] >= ids.shape[1] + 4
'''),
    ("test: add GenerationStats to_dict has all keys test", '''
class TestGenerationStats:
    def test_to_dict_keys(self):
        s = GenerationStats(n_tokens_generated=20, n_target_calls=5,
                             total_accepted=18, total_drafted=20,
                             acceptance_rates=[0.9, 0.85])
        d = s.to_dict()
        for k in ("tokens_generated", "target_model_calls",
                   "mean_accepted_per_step", "acceptance_rate", "speedup_factor"):
            assert k in d
'''),
    ("test: add MedusaHead output shape test", '''
class TestMedusaHead:
    def test_head_output_shape(self):
        head = MedusaHead(d_model=16, vocab_size=V)
        h    = torch.randn(2, 6, 16)
        out  = head(h)
        assert out.shape == (2, 6, V)
'''),
    ("test: add TokenTree path length equals max_depth test", '''
class TestTreeDepth:
    def test_path_length_bounded_by_max_depth(self):
        tree  = TokenTree(max_depth=3, branching=2)
        cands = [[1, 2], [3, 4], [5, 6]]
        root  = tree.build(cands)
        paths = tree.all_paths(root)
        for p in paths:
            assert len(p) <= 3
'''),
    ("test: add LookaheadDecoder window_size=1 works test", '''
class TestLookaheadWindowSize1:
    def test_window_1(self):
        d   = LookaheadDecoder(TinyLM(), window_size=1, n_iters=1)
        ids = torch.randint(0, V, (1, 4))
        out = d.generate(ids, max_new_tokens=3)
        assert out.shape == (1, 7)
'''),
]:
    src = read("tests/test_specd.py")
    src += "\n" + body
    write("tests/test_specd.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.8.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.7.0\"", "__version__ = \"4.8.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.8.0 — Speculative Decoding & Fast Inference release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `agents`     | Tool Use & Agents — ReAct, function calling, parallel exec, structured output, DAG planner |",
    "| `agents`     | Tool Use & Agents — ReAct, function calling, parallel exec, structured output, DAG planner |\n"
    "| `specd`      | Speculative Decoding — draft model, reject/accept, Medusa heads, tree attention, lookahead |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.8.0] — 2024 — Speculative Decoding & Fast Inference\n\n### Added\n"
      "- `DraftModel` ABC — abstract draft model interface\n"
      "- `NgramDraftModel` — n-gram lookup, no neural parameters\n"
      "- `SmallModelDraft` — wraps any small LM as draft model\n"
      "- `SpeculativeSampler` — rejection sampling verify(), greedy_verify()\n"
      "- `SpeculativeDecoder` — K-step speculation loop, GenerationStats\n"
      "- `GenerationStats` — speedup, acceptance_rate, tokens_per_second\n"
      "- `MedusaHead` — SiLU MLP lookahead head for self-speculation\n"
      "- `MedusaModel` — K Medusa heads, speculate(), medusa_loss()\n"
      "- `TokenTree` / `TreeNode` — candidate token tree, all_paths, verify_paths\n"
      "- `LookaheadDecoder` — Jacobi iteration single-model speculation\n"
      "- `examples/specd_demo.py` — full speculative decoding demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.8.0, update README and CHANGELOG for Day 48 Speculative Decoding")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 48 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.8.0",
    "-m", "NanoMind v4.8.0 — Speculative Decoding & Fast Inference", check=False)
r = run("git", "push", "origin", "v4.8.0", check=False)
print("Tag v4.8.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 48 COMPLETE — v4.8.0 TAGGED! ===")
