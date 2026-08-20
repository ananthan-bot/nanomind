"""
day11_commits.py — 20 atomic commits for Day 11: Text Generation Strategies.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

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

print("\n=== DAY 11: Text Generation Strategies — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — generate package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/__init__.py", '"""NanoMind text generation sub-package."""\n')
commit("feat: add nanomind/generate/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — GenerationConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/config.py", '''\
"""
nanomind/generate/config.py — Generation configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """
    Configuration for NanoMind text generation.

    Attributes:
        max_new_tokens:    Maximum tokens to generate.
        strategy:          Sampling strategy — ``"greedy"``, ``"temperature"``,
                           ``"top_k"``, ``"top_p"``, ``"beam"``.
        temperature:       Softmax temperature (< 1 = sharper, > 1 = more random).
        top_k:             Keep top-K logits before sampling (0 = disabled).
        top_p:             Nucleus probability threshold (0.0 = disabled).
        min_p:             Minimum probability threshold (0.0 = disabled).
        repetition_penalty: Penalize repeated tokens (1.0 = disabled).
        num_beams:         Number of beams for beam search.
        eos_token_id:      Token ID that signals end of sequence (None = no EOS).
        seed:              Optional random seed for reproducible sampling.
    """

    max_new_tokens:     int   = 100
    strategy:           str   = "temperature"
    temperature:        float = 0.8
    top_k:              int   = 50
    top_p:              float = 0.0
    min_p:              float = 0.0
    repetition_penalty: float = 1.0
    num_beams:          int   = 1
    eos_token_id:       int | None = None
    seed:               int | None = None

    def __post_init__(self) -> None:
        assert self.max_new_tokens > 0
        assert self.temperature > 0.0
        assert self.top_k >= 0
        assert 0.0 <= self.top_p <= 1.0
        assert 0.0 <= self.min_p <= 1.0
        assert self.repetition_penalty >= 1.0
        assert self.num_beams >= 1
        assert self.strategy in ("greedy", "temperature", "top_k", "top_p", "beam"), (
            f"Unknown strategy '{self.strategy}'"
        )
''')
commit("feat: add GenerationConfig dataclass with all sampling parameters")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — logit processing utilities
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/logit_processors.py", '''\
"""
nanomind/generate/logit_processors.py — Logit filtering and processing.

All processors operate on a raw logit tensor of shape ``(vocab_size,)``
and return a modified logit tensor of the same shape.
Processors can be composed in sequence before sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature to sharpen or flatten the distribution.

    Lower temperature (< 1) makes distribution peakier (more confident).
    Higher temperature (> 1) makes distribution flatter (more random).

    Args:
        logits:      Raw logits ``(vocab_size,)``
        temperature: Positive scalar. Must be > 0.

    Returns:
        Scaled logit tensor ``(vocab_size,)``
    """
    return logits / max(temperature, 1e-8)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Set all logits below the top-K to -inf.

    Args:
        logits: Raw logits ``(vocab_size,)``
        top_k:  Number of top logits to keep. If 0, no filtering is applied.

    Returns:
        Filtered logit tensor with non-top-k entries set to -inf.
    """
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold, _ = torch.topk(logits, k)
    min_threshold = threshold[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability mass exceeds ``top_p``.

    Args:
        logits: Raw logits ``(vocab_size,)``
        top_p:  Cumulative probability threshold in (0, 1].
                If 0.0 or >= 1.0, no filtering is applied.

    Returns:
        Filtered logit tensor.
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative probability exceeds top_p
    remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[remove] = float("-inf")
    # Restore original order
    return sorted_logits.scatter(0, sorted_idx, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-P filtering: remove tokens with probability < min_p * max_prob.

    Args:
        logits: Raw logits ``(vocab_size,)``
        min_p:  Minimum probability ratio threshold [0, 1].

    Returns:
        Filtered logit tensor.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    past_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Reduce logits for tokens that already appeared in ``past_ids``.

    Divides positive logits by ``penalty`` and multiplies negative logits
    by ``penalty``, making repeated tokens less likely.

    Args:
        logits:   Raw logits ``(vocab_size,)``
        past_ids: Previously generated token IDs ``(T,)``
        penalty:  Penalty factor (>= 1.0; 1.0 = no effect).

    Returns:
        Modified logit tensor.
    """
    if penalty == 1.0 or past_ids.numel() == 0:
        return logits
    score = logits.clone()
    for token_id in past_ids.unique():
        if score[token_id] < 0:
            score[token_id] *= penalty
        else:
            score[token_id] /= penalty
    return score
''')
commit("feat: add logit processors — temperature, top-k, top-p, min-p, repetition penalty")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — greedy_decode()
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/strategies.py", '''\
"""
nanomind/generate/strategies.py — Sampling strategy implementations.

Each strategy takes a processed logit tensor ``(vocab_size,)`` and
returns the next token as a scalar integer tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: always pick the highest-probability token.

    Deterministic — no randomness involved.

    Args:
        logits: Raw logits ``(vocab_size,)``

    Returns:
        Scalar token ID tensor.
    """
    return logits.argmax(dim=-1)
''')
commit("feat: add greedy_decode() — deterministic argmax token selection")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — temperature_sample()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/strategies.py")
src += '''

def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from a temperature-scaled distribution.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        temperature: Softmax temperature. < 1 = sharper, > 1 = flatter.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature
    scaled = apply_temperature(logits, temperature)
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
'''
write("nanomind/generate/strategies.py", src)
commit("feat: add temperature_sample() — sample from temperature-scaled softmax distribution")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — top_k_sample()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/strategies.py")
src += '''

def top_k_sample(
    logits: torch.Tensor,
    top_k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample after keeping only the top-K logits.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        top_k:       Number of top tokens to sample from.
        temperature: Temperature scaling applied before sampling.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature, apply_top_k
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
'''
write("nanomind/generate/strategies.py", src)
commit("feat: add top_k_sample() — sample from top-K filtered logit distribution")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — top_p_sample()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/strategies.py")
src += '''

def top_p_sample(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds ``top_p``.

    Args:
        logits:      Raw logits ``(vocab_size,)``
        top_p:       Nucleus probability threshold.
        temperature: Temperature scaling applied before sampling.

    Returns:
        Sampled token ID scalar tensor.
    """
    from nanomind.generate.logit_processors import apply_temperature, apply_top_p
    logits = apply_temperature(logits, temperature)
    logits = apply_top_p(logits, top_p)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
'''
write("nanomind/generate/strategies.py", src)
commit("feat: add top_p_sample() — nucleus sampling over cumulative probability threshold")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — combined sample() dispatcher
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/strategies.py")
src += '''

def sample_next_token(
    logits: torch.Tensor,
    strategy: str = "temperature",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    past_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Unified next-token sampler: applies processors then dispatches to strategy.

    Processing order:
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k filtering
    4. Top-p (nucleus) filtering
    5. Min-p filtering
    6. Softmax + sample (or argmax for greedy)

    Args:
        logits:             Raw logit vector ``(vocab_size,)``
        strategy:           ``"greedy"``, ``"temperature"``, ``"top_k"``, or ``"top_p"``
        temperature:        Softmax temperature.
        top_k:              Top-K filter (0 = disabled).
        top_p:              Nucleus threshold (0.0 = disabled).
        min_p:              Min-p threshold (0.0 = disabled).
        repetition_penalty: Penalty for previously generated tokens.
        past_ids:           Previously generated IDs for repetition penalty.

    Returns:
        Next token as a scalar tensor.
    """
    from nanomind.generate.logit_processors import (
        apply_temperature, apply_top_k, apply_top_p,
        apply_min_p, apply_repetition_penalty,
    )

    if strategy == "greedy":
        return greedy_decode(logits)

    # Apply processors in order
    if repetition_penalty != 1.0 and past_ids is not None:
        logits = apply_repetition_penalty(logits, past_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    if top_k > 0:
        logits = apply_top_k(logits, top_k)
    if top_p > 0.0:
        logits = apply_top_p(logits, top_p)
    if min_p > 0.0:
        logits = apply_min_p(logits, min_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
'''
write("nanomind/generate/strategies.py", src)
commit("feat: add sample_next_token() — unified dispatcher with all logit processors")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — beam_search()
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/beam.py", '''\
"""
nanomind/generate/beam.py — Basic beam search decoder for NanoMind.

Beam search maintains ``num_beams`` candidate sequences in parallel and
selects the globally most probable sequence, unlike greedy which only
keeps one at each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def beam_search(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    num_beams: int = 4,
    eos_token_id: int | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """
    Simple beam search decoder.

    Args:
        model:          The NanoMind model.
        idx:            Seed token IDs ``(1, T)`` — single sequence only.
        max_new_tokens: Maximum tokens to generate.
        num_beams:      Number of beams to maintain.
        eos_token_id:   If provided, stop beam when all beams hit EOS.
        block_size:     Maximum context length (defaults to model.cfg.block_size).

    Returns:
        Best token sequence ``(1, T + generated)``
    """
    model.eval()
    device     = idx.device
    block_size = block_size or getattr(model, "cfg", None) and model.cfg.block_size or 512

    # Initialise beams: (score, sequence)
    beams: list[tuple[float, torch.Tensor]] = [(0.0, idx.clone())]

    for _ in range(max_new_tokens):
        all_candidates: list[tuple[float, torch.Tensor]] = []

        for score, seq in beams:
            # Check EOS
            if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                all_candidates.append((score, seq))
                continue

            ctx     = seq[:, -block_size:]
            logits, _ = model(ctx)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            # Expand each beam by num_beams candidates
            topk_log_probs, topk_ids = log_probs.topk(num_beams)
            for lp, tok_id in zip(topk_log_probs, topk_ids):
                new_seq = torch.cat([seq, tok_id.unsqueeze(0).unsqueeze(0)], dim=1)
                all_candidates.append((score + lp.item(), new_seq))

        # Keep top num_beams candidates by score
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:num_beams]

        # Early stop if all beams end with EOS
        if eos_token_id is not None:
            if all(b[1][0, -1].item() == eos_token_id for b in beams):
                break

    # Return the highest-scoring sequence
    best_score, best_seq = beams[0]
    return best_seq
''')
commit("feat: add beam_search() — maintain N candidate sequences, return best scoring")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — Generator class skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/generator.py", '''\
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
''')
commit("feat: add Generator class skeleton with model, tokenizer, and device setup")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — Generator.generate()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/generator.py")
src += '''
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
'''
write("nanomind/generate/generator.py", src)
commit("feat: implement Generator.generate() with strategy dispatch and EOS stopping")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — Generator.stream()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/generator.py")
src += '''
    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens one at a time.

        Yields each decoded character/token as it is generated, enabling
        real-time display without waiting for the full output.

        Args:
            prompt: Input text prompt.
            cfg:    Generation configuration.

        Yields:
            One decoded string token at a time.
        """
        cfg = cfg or GenerationConfig()
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        ids  = self.tokenizer.encode(prompt)
        idx  = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        block_size = getattr(self.model, "cfg", None) and self.model.cfg.block_size or 512

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

            if cfg.eos_token_id is not None and tok_id == cfg.eos_token_id:
                return

            idx = torch.cat([idx, next_tok.unsqueeze(0).unsqueeze(0)], dim=1)
            yield self.tokenizer.decode([tok_id])
'''
write("nanomind/generate/generator.py", src)
commit("feat: add Generator.stream() — yield tokens one at a time for real-time display")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — Generator.batch_generate()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/generator.py")
src += '''
    @torch.no_grad()
    def batch_generate(
        self,
        prompts: list[str],
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """
        Generate text for a list of prompts.

        Note: Prompts are padded to the same length. For best results,
        use prompts of similar length.

        Args:
            prompts: List of prompt strings.
            cfg:     Generation configuration.

        Returns:
            List of generated strings (one per prompt).
        """
        return [self.generate(p, cfg) for p in prompts]

    def __repr__(self) -> str:
        return (
            f"Generator("
            f"model={type(self.model).__name__}, "
            f"tokenizer={self.tokenizer}, "
            f"device={self.device})"
        )
'''
write("nanomind/generate/generator.py", src)
commit("feat: add Generator.batch_generate() and __repr__")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update generate __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/__init__.py", '''\
"""NanoMind text generation sub-package.

Primary exports:
    - :class:`Generator`          — high-level text generation interface
    - :class:`GenerationConfig`   — generation hyperparameter configuration
    - :func:`sample_next_token`   — unified next-token sampler
    - :func:`greedy_decode`       — argmax decoding
    - :func:`temperature_sample`  — temperature-scaled sampling
    - :func:`top_k_sample`        — top-K filtered sampling
    - :func:`top_p_sample`        — nucleus (top-p) sampling
    - :func:`beam_search`         — beam search decoder

Logit processors:
    - :func:`apply_temperature`       — scale logits by temperature
    - :func:`apply_top_k`             — zero out below top-K
    - :func:`apply_top_p`             — nucleus filtering
    - :func:`apply_min_p`             — min-p filtering
    - :func:`apply_repetition_penalty`— penalise repeated tokens
"""

from nanomind.generate.config import GenerationConfig
from nanomind.generate.generator import Generator
from nanomind.generate.strategies import (
    greedy_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    sample_next_token,
)
from nanomind.generate.beam import beam_search
from nanomind.generate.logit_processors import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_min_p,
    apply_repetition_penalty,
)

__all__ = [
    "GenerationConfig",
    "Generator",
    "greedy_decode",
    "temperature_sample",
    "top_k_sample",
    "top_p_sample",
    "sample_next_token",
    "beam_search",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_min_p",
    "apply_repetition_penalty",
]
''')
commit("refactor: export all generation components from nanomind/generate/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: greedy decode
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_generate.py", '''\
"""
tests/test_generate.py — Tests for NanoMind text generation.
"""

import pytest
import torch
import torch.nn.functional as F

from nanomind.model import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.generate import (
    GenerationConfig,
    Generator,
    greedy_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    sample_next_token,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
)

VOCAB = 32
CFG   = ModelConfig(
    vocab_size=VOCAB, block_size=16,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)
CORPUS = "abcdefghijklmnopqrstuvwxyz " * 10


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NanoMind(CFG)


@pytest.fixture
def tokenizer():
    return CharTokenizer().build(CORPUS)


@pytest.fixture
def generator(model, tokenizer):
    return Generator(model, tokenizer, device=torch.device("cpu"))


# ── greedy_decode ─────────────────────────────────────────────────────────────

class TestGreedyDecode:
    def test_returns_argmax(self):
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        assert greedy_decode(logits).item() == 1   # index of max = 1

    def test_deterministic(self):
        logits = torch.randn(VOCAB)
        assert greedy_decode(logits).item() == greedy_decode(logits).item()

    def test_returns_scalar(self):
        logits = torch.randn(VOCAB)
        result = greedy_decode(logits)
        assert result.shape == ()
''')
commit("test: add greedy_decode() tests — argmax correctness and determinism")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: logit processors
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_generate.py")
src += '''

# ── Logit processors ──────────────────────────────────────────────────────────

class TestLogitProcessors:
    def test_temperature_sharpens(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        low_t  = F.softmax(apply_temperature(logits, 0.1), dim=-1)
        high_t = F.softmax(apply_temperature(logits, 5.0), dim=-1)
        # Low temperature -> higher max prob
        assert low_t.max() > high_t.max()

    def test_top_k_keeps_k_tokens(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_k(logits, top_k=5)
        n_finite = (filtered != float("-inf")).sum().item()
        assert n_finite == 5

    def test_top_k_zero_no_filter(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_k(logits, top_k=0)
        assert torch.equal(filtered, logits)

    def test_top_p_filters_low_prob(self):
        logits   = torch.randn(VOCAB)
        filtered = apply_top_p(logits, top_p=0.5)
        n_inf    = (filtered == float("-inf")).sum().item()
        assert n_inf > 0   # some tokens should be removed

    def test_repetition_penalty_reduces_seen_token(self):
        logits  = torch.zeros(VOCAB)
        logits[5] = 2.0
        past    = torch.tensor([5])
        penalized = apply_repetition_penalty(logits, past, penalty=2.0)
        assert penalized[5] < logits[5]   # penalized = original / 2

    def test_rep_penalty_one_is_no_op(self):
        logits  = torch.randn(VOCAB)
        past    = torch.arange(VOCAB)
        result  = apply_repetition_penalty(logits, past, penalty=1.0)
        assert torch.equal(result, logits)
'''
write("tests/test_generate.py", src)
commit("test: add logit processor tests — temperature, top-k, top-p, repetition penalty")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: top-k and top-p sampling
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_generate.py")
src += '''

# ── Sampling strategies ───────────────────────────────────────────────────────

class TestSamplingStrategies:
    def test_temperature_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = temperature_sample(logits)
        assert 0 <= tok.item() < VOCAB

    def test_top_k_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = top_k_sample(logits, top_k=5)
        assert 0 <= tok.item() < VOCAB

    def test_top_p_sample_in_vocab(self):
        logits = torch.randn(VOCAB)
        tok    = top_p_sample(logits, top_p=0.9)
        assert 0 <= tok.item() < VOCAB

    def test_greedy_vs_temperature_zero(self):
        logits = torch.randn(VOCAB)
        greedy = greedy_decode(logits).item()
        sampled = temperature_sample(logits, temperature=1e-8).item()
        assert greedy == sampled   # Very low temp -> greedy-like

    def test_sample_next_token_dispatcher_greedy(self):
        logits = torch.randn(VOCAB)
        g      = greedy_decode(logits).item()
        s      = sample_next_token(logits, strategy="greedy").item()
        assert g == s
'''
write("tests/test_generate.py", src)
commit("test: add top-k, top-p, temperature sampling and dispatcher tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: Generator.generate() shape and content
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_generate.py")
src += '''

# ── Generator ─────────────────────────────────────────────────────────────────

class TestGenerator:
    def test_generate_returns_string(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="greedy")
        out = generator.generate("abc", cfg)
        assert isinstance(out, str)

    def test_generate_length_bounded(self, generator):
        cfg = GenerationConfig(max_new_tokens=10, strategy="greedy")
        out = generator.generate("abc", cfg)
        # Decoded output length depends on tokenizer but tokens <= max_new_tokens
        assert len(generator.tokenizer.encode(out)) <= 10

    def test_greedy_is_deterministic(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="greedy")
        out1 = generator.generate("abc", cfg)
        out2 = generator.generate("abc", cfg)
        assert out1 == out2

    def test_seeded_generation_is_deterministic(self, generator):
        cfg = GenerationConfig(max_new_tokens=5, strategy="temperature", seed=42)
        out1 = generator.generate("abc", cfg)
        out2 = generator.generate("abc", cfg)
        assert out1 == out2
'''
write("tests/test_generate.py", src)
commit("test: add Generator.generate() string output, length bound, and determinism tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: Generator.stream() + GenerationConfig validation
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_generate.py")
src += '''

# ── Generator.stream() ────────────────────────────────────────────────────────

class TestGeneratorStream:
    def test_stream_yields_strings(self, generator):
        cfg    = GenerationConfig(max_new_tokens=5, strategy="greedy")
        tokens = list(generator.stream("abc", cfg))
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_n_tokens(self, generator):
        cfg    = GenerationConfig(max_new_tokens=5, strategy="greedy")
        tokens = list(generator.stream("abc", cfg))
        assert len(tokens) == 5

    def test_stream_concat_matches_generate(self, generator):
        cfg  = GenerationConfig(max_new_tokens=5, strategy="greedy")
        gen  = generator.generate("abc", cfg)
        strm = "".join(generator.stream("abc", cfg))
        assert gen == strm


# ── GenerationConfig ──────────────────────────────────────────────────────────

class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.strategy == "temperature"
        assert cfg.max_new_tokens == 100

    def test_invalid_strategy(self):
        with pytest.raises(AssertionError):
            GenerationConfig(strategy="random_walk")

    def test_invalid_top_p(self):
        with pytest.raises(AssertionError):
            GenerationConfig(top_p=1.5)

    def test_invalid_rep_penalty(self):
        with pytest.raises(AssertionError):
            GenerationConfig(repetition_penalty=0.5)   # must be >= 1.0
'''
write("tests/test_generate.py", src)
commit("test: add Generator.stream() and GenerationConfig validation tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 11 | Text generation strategies | 🔜 |",
    "| 11 | Text generation strategies | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Checkpointing: atomic save/load, CheckpointManager, best tracking, auto_resume, inference ckpt (Day 10)",
    "- Checkpointing: atomic save/load, CheckpointManager, best tracking, auto_resume, inference ckpt (Day 10)\n- Generation: greedy, temperature, top-k, top-p, min-p, beam search, Generator, stream() (Day 11)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 11 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 11 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 11 COMPLETE ===")
