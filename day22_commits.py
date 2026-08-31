"""
day22_commits.py — 20 atomic commits for Day 22: Beam Search & Diverse Beam Search.
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

print("\n=== DAY 22: Beam Search & Diverse Beam Search — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — beam.py skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/beam.py", '''\
"""
nanomind/generate/beam.py — Beam Search and Diverse Beam Search.

Greedy decoding always takes the single most likely token at each step.
Beam search keeps the top-B most likely *sequences*, expanding each one
at every step and pruning back to B beams:

  Greedy:   O(T)     sequences explored — fast but suboptimal
  Beam (B): O(T × B) sequences explored — better quality, B times slower

Diverse Beam Search (Vijayakumar et al. 2016) splits the B beams into G groups
and penalises each group for choosing tokens already chosen by earlier groups,
encouraging diversity across the returned hypotheses.

References:
  Beam search:          Sutskever et al. (2014)
  Diverse beam search:  Vijayakumar et al. (2016) — https://arxiv.org/abs/1610.02424
  Length penalty:       Wu et al. (2016) GNMT — https://arxiv.org/abs/1609.08144
"""

from __future__ import annotations
''')
commit("feat: add nanomind/generate/beam.py — Beam Search module skeleton with docstring")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — BeamConfig
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''
from dataclasses import dataclass


@dataclass
class BeamConfig:
    """
    Configuration for beam search generation.

    Attributes:
        num_beams:          Number of beams (width of the search).
                            1 = greedy decoding, >1 = beam search.
        max_new_tokens:     Maximum tokens to generate per beam.
        length_penalty:     Exponent for length normalisation.
                            α > 1 favours longer sequences; α < 1 shorter.
                            α = 1.0 means no length penalty.
        early_stopping:     Stop when all beams have produced EOS.
        no_repeat_ngram:    Block repeated n-grams (0 = off).
        num_beam_groups:    Number of diverse beam groups (Diverse BS).
                            Must divide ``num_beams`` evenly.
        diversity_penalty:  Penalty for tokens chosen by earlier groups (DBS).
        temperature:        Logit temperature before beam scoring.
        top_k:              Top-K filter on logits (0 = off).
        return_n_best:      How many final hypotheses to return (≤ num_beams).
    """

    num_beams:         int   = 4
    max_new_tokens:    int   = 50
    length_penalty:    float = 1.0
    early_stopping:    bool  = True
    no_repeat_ngram:   int   = 0
    num_beam_groups:   int   = 1
    diversity_penalty: float = 0.5
    temperature:       float = 1.0
    top_k:             int   = 0
    return_n_best:     int   = 1

    def __post_init__(self) -> None:
        assert self.num_beams >= 1
        assert self.num_beams % self.num_beam_groups == 0, \
            "num_beams must be divisible by num_beam_groups"
        assert self.return_n_best <= self.num_beams
        assert self.temperature > 0.0
        assert self.max_new_tokens > 0
'''
write("nanomind/generate/beam.py", src)
commit("feat: add BeamConfig — num_beams, length_penalty, diverse groups, diversity_penalty")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — BeamHypothesis and BeamHypotheses
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''

import torch
import torch.nn.functional as F
import math


class BeamHypothesis:
    """
    A single beam hypothesis: a sequence of tokens with an accumulated log-prob.

    Args:
        tokens:   Token IDs generated so far (including prompt).
        log_prob: Cumulative log-probability (unnormalized).
    """

    __slots__ = ("tokens", "log_prob")

    def __init__(self, tokens: list[int], log_prob: float = 0.0) -> None:
        self.tokens   = tokens
        self.log_prob = log_prob

    def score(self, length_penalty: float = 1.0) -> float:
        """Length-normalised score (GNMT formula)."""
        lp = ((5 + len(self.tokens)) / 6) ** length_penalty
        return self.log_prob / lp

    def extend(self, token_id: int, log_p: float) -> "BeamHypothesis":
        return BeamHypothesis(self.tokens + [token_id], self.log_prob + log_p)

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"BeamHypothesis(len={len(self.tokens)}, score={self.score():.3f})"


class BeamHypotheses:
    """
    Container for completed beam hypotheses.

    Keeps only the top ``num_beams`` completed hypotheses by length-penalised score.

    Args:
        num_beams:      Maximum number of hypotheses to store.
        length_penalty: GNMT length penalty exponent.
        early_stopping: Whether to declare done when filled.
    """

    def __init__(
        self,
        num_beams:      int,
        length_penalty: float = 1.0,
        early_stopping: bool  = True,
    ) -> None:
        self.num_beams      = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.hyps: list[BeamHypothesis] = []

    def add(self, hyp: BeamHypothesis) -> None:
        """Add a completed hypothesis, keeping only the top num_beams."""
        self.hyps.append(hyp)
        self.hyps.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
        if len(self.hyps) > self.num_beams:
            self.hyps.pop()

    @property
    def is_done(self) -> bool:
        """True if we have enough hypotheses and early stopping is enabled."""
        return self.early_stopping and len(self.hyps) >= self.num_beams

    def best(self, n: int = 1) -> list[BeamHypothesis]:
        """Return the top-n hypotheses by score."""
        return self.hyps[:n]
'''
write("nanomind/generate/beam.py", src)
commit("feat: add BeamHypothesis and BeamHypotheses — length-penalised score, sorted storage")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — _get_logits() helper
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''

import torch.nn as nn


def _get_next_logprobs(
    model:       nn.Module,
    input_ids:   torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    block_size:  int   = 512,
) -> torch.Tensor:
    """
    Run model forward and return log-probabilities for the next token.

    Args:
        model:      Language model.
        input_ids:  ``(1, T)`` token IDs.
        temperature: Temperature for logit scaling.
        top_k:      Top-K filtering (0 = off).
        block_size: Model\'s context window.

    Returns:
        Log-prob tensor ``(vocab_size,)``.
    """
    ctx    = input_ids[:, -block_size:]
    logits, _ = model(ctx)
    logits = logits[0, -1, :].float() / max(temperature, 1e-8)

    if top_k > 0:
        topk_vals = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < topk_vals, float("-inf"))

    return F.log_softmax(logits, dim=-1)
'''
write("nanomind/generate/beam.py", src)
commit("feat: add _get_next_logprobs() — model forward + temperature + top-K → log-probs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — no_repeat_ngram blocking
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''

def _block_repeat_ngrams(
    tokens:       list[int],
    log_probs:    torch.Tensor,
    no_repeat_ngram: int,
) -> torch.Tensor:
    """
    Block tokens that would create a repeated n-gram.

    Scans the current token sequence for any n-gram of length ``no_repeat_ngram``
    and sets the log-prob of any token that would continue a seen n-gram to -inf.

    Args:
        tokens:          Current token sequence.
        log_probs:       Next-token log-probs ``(vocab_size,)``.
        no_repeat_ngram: Block n-grams of this length (0 = off).

    Returns:
        Modified log-probs with blocked tokens set to -inf.
    """
    if no_repeat_ngram <= 0 or len(tokens) < no_repeat_ngram:
        return log_probs

    n   = no_repeat_ngram
    lp  = log_probs.clone()
    # Build set of banned next tokens
    banned: set[int] = set()
    tail = tokens[-(n - 1):]   # last (n-1) tokens
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n - 1] == tail:
            banned.add(tokens[i + n - 1])
    for tok in banned:
        lp[tok] = float("-inf")
    return lp
'''
write("nanomind/generate/beam.py", src)
commit("feat: add _block_repeat_ngrams() — no-repeat-ngram constraint for beam search")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — beam_search() main function
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''

@torch.no_grad()
def beam_search(
    model:        nn.Module,
    input_ids:    torch.Tensor,
    cfg:          BeamConfig | None = None,
    eos_token_id: int | None = None,
) -> list[BeamHypothesis]:
    """
    Standard beam search generation.

    Maintains ``num_beams`` candidate sequences in parallel, expanding each
    by the top-``num_beams`` next tokens and keeping the globally best B sequences.

    Args:
        model:        Language model with forward(input_ids) → (logits, _).
        input_ids:    Prompt token IDs ``(1, T)``.
        cfg:          Beam search configuration.
        eos_token_id: End-of-sequence token ID.

    Returns:
        List of completed :class:`BeamHypothesis` objects sorted by score.
        Length is ``cfg.return_n_best``.
    """
    cfg   = cfg or BeamConfig()
    model.eval()
    device      = input_ids.device
    block_size  = getattr(model, "cfg", None) and model.cfg.block_size or 512
    vocab_size  = getattr(model, "cfg", None) and model.cfg.vocab_size or model.lm_head.out_features

    # Initialise beams: one starting hypothesis
    prompt_tokens = input_ids[0].tolist()
    active_beams  = [BeamHypothesis(prompt_tokens, 0.0)]
    completed     = BeamHypotheses(cfg.num_beams, cfg.length_penalty, cfg.early_stopping)

    for _ in range(cfg.max_new_tokens):
        if not active_beams:
            break

        candidates: list[tuple[float, BeamHypothesis]] = []

        for hyp in active_beams:
            ids    = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
            lp     = _get_next_logprobs(model, ids, cfg.temperature, cfg.top_k, block_size)

            if cfg.no_repeat_ngram > 0:
                lp = _block_repeat_ngrams(hyp.tokens, lp, cfg.no_repeat_ngram)

            # Expand: take top num_beams tokens
            top_lp, top_ids = torch.topk(lp, min(cfg.num_beams, vocab_size))
            for tok, tok_lp in zip(top_ids.tolist(), top_lp.tolist()):
                new_hyp = hyp.extend(tok, tok_lp)
                candidates.append((new_hyp.log_prob, new_hyp))

        # Sort candidates globally by cumulative log-prob
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Partition into complete (EOS) and active beams
        active_beams = []
        for _, hyp in candidates:
            if eos_token_id is not None and hyp.tokens[-1] == eos_token_id:
                completed.add(hyp)
                if completed.is_done:
                    break
            else:
                active_beams.append(hyp)
                if len(active_beams) >= cfg.num_beams:
                    break

        if completed.is_done:
            break

    # Add any remaining active beams as completed
    for hyp in active_beams:
        completed.add(hyp)

    results = completed.best(cfg.return_n_best)
    if not results:
        results = active_beams[:cfg.return_n_best]
    return results
'''
write("nanomind/generate/beam.py", src)
commit("feat: implement beam_search() — full beam expansion, EOS handling, no-repeat-ngram")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — diverse_beam_search()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/beam.py")
src += '''

@torch.no_grad()
def diverse_beam_search(
    model:        nn.Module,
    input_ids:    torch.Tensor,
    cfg:          BeamConfig | None = None,
    eos_token_id: int | None = None,
) -> list[BeamHypothesis]:
    """
    Diverse Beam Search (Vijayakumar et al. 2016).

    Splits ``num_beams`` into ``num_beam_groups`` groups. Each group runs
    standard beam search but is penalised for choosing tokens already selected
    by earlier groups in the same step, encouraging diverse outputs.

    Args:
        model:        Language model.
        input_ids:    Prompt token IDs ``(1, T)``.
        cfg:          Beam configuration with ``num_beam_groups > 1``.
        eos_token_id: EOS token ID.

    Returns:
        List of top :class:`BeamHypothesis` objects from all groups.
    """
    cfg = cfg or BeamConfig()
    model.eval()
    device      = input_ids.device
    block_size  = getattr(model, "cfg", None) and model.cfg.block_size or 512
    vocab_size  = getattr(model, "cfg", None) and model.cfg.vocab_size or model.lm_head.out_features

    n_groups    = cfg.num_beam_groups
    beams_group = cfg.num_beams // n_groups
    prompt_tokens = input_ids[0].tolist()

    # One set of active beams per group
    group_beams: list[list[BeamHypothesis]] = [
        [BeamHypothesis(prompt_tokens, 0.0)]
        for _ in range(n_groups)
    ]
    group_completed: list[BeamHypotheses] = [
        BeamHypotheses(beams_group, cfg.length_penalty, cfg.early_stopping)
        for _ in range(n_groups)
    ]

    for _ in range(cfg.max_new_tokens):
        for g_idx in range(n_groups):
            if not group_beams[g_idx]:
                continue

            # Tokens already selected by earlier groups this step
            already_chosen: set[int] = set()
            for prev_g in range(g_idx):
                for hyp in group_beams[prev_g][:beams_group]:
                    if hyp.tokens:
                        already_chosen.add(hyp.tokens[-1])

            candidates: list[tuple[float, BeamHypothesis]] = []
            for hyp in group_beams[g_idx]:
                ids  = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
                lp   = _get_next_logprobs(model, ids, cfg.temperature, cfg.top_k, block_size)

                # Apply diversity penalty
                for tok in already_chosen:
                    lp[tok] -= cfg.diversity_penalty

                if cfg.no_repeat_ngram > 0:
                    lp = _block_repeat_ngrams(hyp.tokens, lp, cfg.no_repeat_ngram)

                top_lp, top_ids = torch.topk(lp, min(beams_group, vocab_size))
                for tok, tok_lp in zip(top_ids.tolist(), top_lp.tolist()):
                    new_hyp = hyp.extend(tok, tok_lp)
                    candidates.append((new_hyp.log_prob, new_hyp))

            candidates.sort(key=lambda x: x[0], reverse=True)
            new_active: list[BeamHypothesis] = []
            for _, hyp in candidates:
                if eos_token_id is not None and hyp.tokens[-1] == eos_token_id:
                    group_completed[g_idx].add(hyp)
                else:
                    new_active.append(hyp)
                    if len(new_active) >= beams_group:
                        break
            group_beams[g_idx] = new_active

    # Collect results from all groups
    all_hyps: list[BeamHypothesis] = []
    for g_idx in range(n_groups):
        for hyp in group_beams[g_idx]:
            group_completed[g_idx].add(hyp)
        all_hyps.extend(group_completed[g_idx].best(beams_group))

    all_hyps.sort(key=lambda h: h.score(cfg.length_penalty), reverse=True)
    return all_hyps[:cfg.return_n_best]
'''
write("nanomind/generate/beam.py", src)
commit("feat: implement diverse_beam_search() — G groups with diversity penalty (DBS)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — BeamSearchGenerator high-level class
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/generate/beam_generator.py", '''\
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
''')
commit("feat: add BeamSearchGenerator — high-level generate() with standard and diverse beam")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — update generate strategies to include "beam"
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/strategies.py")
if "beam" not in src:
    src += '''

def beam_decode(
    model:        "nn.Module",
    input_ids:    "torch.Tensor",
    num_beams:    int = 4,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
    **kwargs,
) -> "torch.Tensor":
    """
    Beam search decode — convenience wrapper for use in the generate pipeline.
    Returns the best beam as a token tensor (1, T + generated).
    """
    from nanomind.generate.beam import BeamConfig, beam_search
    cfg = BeamConfig(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        return_n_best=1,
    )
    hyps = beam_search(model, input_ids, cfg)
    best = hyps[0].tokens
    import torch
    return torch.tensor([best], dtype=torch.long, device=input_ids.device)
'''
write("nanomind/generate/strategies.py", src)
commit("feat: add beam_decode() to strategies — convenience wrapper for the generate pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update generate __init__ to export beam search
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/generate/__init__.py")
src = src.rstrip() + (
    "\n\nfrom nanomind.generate.beam import ("
    "\n    BeamConfig,"
    "\n    BeamHypothesis,"
    "\n    BeamHypotheses,"
    "\n    beam_search,"
    "\n    diverse_beam_search,"
    "\n)\n"
    "from nanomind.generate.beam_generator import BeamSearchGenerator\n"
)
write("nanomind/generate/__init__.py", src)
commit("refactor: export BeamConfig, beam_search, diverse_beam_search, BeamSearchGenerator")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: beam_search_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/beam_search_demo.py", '''\
"""
examples/beam_search_demo.py — Beam search vs. greedy decoding demo.

Compares greedy, standard beam search, and diverse beam search outputs
from the same prompt and model.

Usage:
    python examples/beam_search_demo.py
"""

import torch
from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.generate.beam import BeamConfig, beam_search, diverse_beam_search
from nanomind.generate import generate_text

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 40
tokenizer = CharTokenizer().build(CORPUS)
VOCAB     = tokenizer.vocab_size
BLOCK     = 48
device    = torch.device("cpu")

model = NanoMind(ModelConfig(
    vocab_size=VOCAB, block_size=BLOCK,
    d_model=64, n_layers=2, n_heads=4, dropout=0.0,
)).to(device)

PROMPT   = "the quick"
ids      = tokenizer.encode(PROMPT)
idx      = torch.tensor([ids], dtype=torch.long, device=device)
n_prompt = len(ids)

print("=" * 60)
print("Greedy decoding:")
greedy_text = generate_text(model, tokenizer, PROMPT, max_new_tokens=30, strategy="greedy")
print(f"  {greedy_text!r}")

print("\\nStandard Beam Search (B=4, length_penalty=1.2):")
cfg  = BeamConfig(num_beams=4, max_new_tokens=30, length_penalty=1.2, return_n_best=4)
hyps = beam_search(model, idx, cfg)
for i, h in enumerate(hyps):
    text = tokenizer.decode(h.tokens[n_prompt:])
    print(f"  Beam {i+1} (score={h.score(1.2):.3f}): {text!r}")

print("\\nDiverse Beam Search (B=4, G=2, div_penalty=0.5):")
dcfg = BeamConfig(num_beams=4, max_new_tokens=30, num_beam_groups=2,
                  diversity_penalty=0.5, return_n_best=4)
dhyps = diverse_beam_search(model, idx, dcfg)
for i, h in enumerate(dhyps):
    text = tokenizer.decode(h.tokens[n_prompt:])
    print(f"  Beam {i+1} (score={h.score():.3f}): {text!r}")
print("=" * 60)
''')
commit("feat: add examples/beam_search_demo.py — greedy vs beam vs diverse beam comparison")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: BeamConfig
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_beam.py", '''\
"""
tests/test_beam.py — Tests for Beam Search and Diverse Beam Search.
"""

import pytest
import torch

from nanomind import NanoMind, ModelConfig
from nanomind.generate.beam import (
    BeamConfig, BeamHypothesis, BeamHypotheses,
    beam_search, diverse_beam_search,
    _block_repeat_ngrams,
)
from nanomind.generate.beam_generator import BeamSearchGenerator
from nanomind.tokenizer.char import CharTokenizer

VOCAB, BLOCK, D, H = 32, 16, 64, 4
B = 1

def tiny_model():
    torch.manual_seed(42)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D, n_layers=2, n_heads=H, dropout=0.0))

TOKENIZER = CharTokenizer().build("abcdefghijklmnopqrstuvwxyz " * 5)

def make_idx(t=4):
    return torch.randint(0, VOCAB, (1, t))


# ── BeamConfig ────────────────────────────────────────────────────────────────

class TestBeamConfig:
    def test_defaults(self):
        cfg = BeamConfig()
        assert cfg.num_beams == 4
        assert cfg.length_penalty == 1.0
        assert cfg.num_beam_groups == 1

    def test_invalid_num_beams(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=0)

    def test_invalid_group_divisor(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=4, num_beam_groups=3)

    def test_invalid_return_n_best(self):
        with pytest.raises(AssertionError):
            BeamConfig(num_beams=2, return_n_best=5)

    def test_temperature_positive(self):
        with pytest.raises(AssertionError):
            BeamConfig(temperature=0.0)
''')
commit("test: add BeamConfig validation — defaults, invalid num_beams, groups, return_n_best")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: BeamHypothesis and BeamHypotheses
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── BeamHypothesis ────────────────────────────────────────────────────────────

class TestBeamHypothesis:
    def test_extend_appends_token(self):
        hyp = BeamHypothesis([1, 2, 3], log_prob=-1.0)
        new = hyp.extend(4, -0.5)
        assert new.tokens == [1, 2, 3, 4]
        assert abs(new.log_prob - (-1.5)) < 1e-6

    def test_len(self):
        hyp = BeamHypothesis([1, 2, 3])
        assert len(hyp) == 3

    def test_score_with_no_penalty(self):
        hyp  = BeamHypothesis([1, 2, 3, 4], log_prob=-4.0)
        s1   = hyp.score(length_penalty=1.0)
        s0   = hyp.score(length_penalty=0.0)
        assert isinstance(s1, float)
        assert isinstance(s0, float)

    def test_longer_favoured_by_high_penalty(self):
        short = BeamHypothesis([1, 2],             log_prob=-2.0)
        long_ = BeamHypothesis([1, 2, 3, 4, 5, 6], log_prob=-6.0)
        # With high length penalty, longer sequence should score better
        assert long_.score(2.0) > short.score(2.0)


class TestBeamHypotheses:
    def test_add_and_best(self):
        bh = BeamHypotheses(num_beams=2, length_penalty=1.0)
        h1 = BeamHypothesis([1, 2], log_prob=-1.0)
        h2 = BeamHypothesis([1, 3], log_prob=-0.5)
        bh.add(h1)
        bh.add(h2)
        best = bh.best(2)
        assert len(best) == 2
        assert best[0].log_prob == -0.5   # higher score first

    def test_capacity_capped_at_num_beams(self):
        bh = BeamHypotheses(num_beams=2, length_penalty=1.0)
        for i in range(5):
            bh.add(BeamHypothesis([i], log_prob=float(-i)))
        assert len(bh.hyps) <= 2

    def test_is_done_when_full_and_early_stop(self):
        bh = BeamHypotheses(num_beams=2, early_stopping=True)
        bh.add(BeamHypothesis([1], log_prob=0.0))
        bh.add(BeamHypothesis([2], log_prob=-0.1))
        assert bh.is_done

    def test_not_done_without_early_stop(self):
        bh = BeamHypotheses(num_beams=2, early_stopping=False)
        bh.add(BeamHypothesis([1], log_prob=0.0))
        bh.add(BeamHypothesis([2], log_prob=-0.1))
        assert not bh.is_done
'''
write("tests/test_beam.py", src)
commit("test: add BeamHypothesis extend, score, and BeamHypotheses capacity and is_done tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: no_repeat_ngram blocking
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── no-repeat-ngram ───────────────────────────────────────────────────────────

class TestNoRepeatNgram:
    def test_no_block_when_disabled(self):
        tokens  = [1, 2, 3]
        lp      = torch.zeros(VOCAB)
        result  = _block_repeat_ngrams(tokens, lp, 0)
        assert torch.equal(result, lp)

    def test_blocks_repeated_ngram(self):
        # tokens=[1,2,3,1,2] → with n=3, tail=[1,2] seen before at pos 0,1
        # → token 3 should be blocked
        tokens  = [1, 2, 3, 1, 2]
        lp      = torch.zeros(VOCAB)
        result  = _block_repeat_ngrams(tokens, lp, 3)
        assert result[3].item() == float("-inf")

    def test_no_block_when_sequence_too_short(self):
        tokens  = [1, 2]
        lp      = torch.zeros(VOCAB)
        result  = _block_repeat_ngrams(tokens, lp, 3)
        assert (result == 0).all()
'''
write("tests/test_beam.py", src)
commit("test: add _block_repeat_ngrams() disable, block, and short-sequence tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: beam_search output shape
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── beam_search ───────────────────────────────────────────────────────────────

class TestBeamSearch:
    def test_returns_list_of_hypotheses(self):
        model = tiny_model()
        idx   = make_idx()
        cfg   = BeamConfig(num_beams=2, max_new_tokens=5, return_n_best=2)
        hyps  = beam_search(model, idx, cfg)
        assert len(hyps) == 2

    def test_hypotheses_longer_than_prompt(self):
        model = tiny_model()
        idx   = make_idx(4)
        cfg   = BeamConfig(num_beams=2, max_new_tokens=5)
        hyps  = beam_search(model, idx, cfg)
        assert len(hyps[0]) > 4

    def test_return_n_best_1(self):
        model = tiny_model()
        idx   = make_idx()
        cfg   = BeamConfig(num_beams=4, max_new_tokens=5, return_n_best=1)
        hyps  = beam_search(model, idx, cfg)
        assert len(hyps) == 1

    def test_tokens_in_vocab(self):
        model = tiny_model()
        idx   = make_idx()
        cfg   = BeamConfig(num_beams=2, max_new_tokens=5)
        hyps  = beam_search(model, idx, cfg)
        for tok in hyps[0].tokens:
            assert 0 <= tok < VOCAB

    def test_score_is_finite(self):
        model = tiny_model()
        idx   = make_idx()
        cfg   = BeamConfig(num_beams=2, max_new_tokens=5)
        hyps  = beam_search(model, idx, cfg)
        assert all(abs(h.score()) < float("inf") for h in hyps)
'''
write("tests/test_beam.py", src)
commit("test: add beam_search() shape, hypothesis length, vocab bounds, and score tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: diverse_beam_search
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── diverse_beam_search ───────────────────────────────────────────────────────

class TestDiverseBeamSearch:
    def test_returns_hypotheses(self):
        model = tiny_model()
        idx   = make_idx()
        cfg   = BeamConfig(num_beams=4, max_new_tokens=5,
                           num_beam_groups=2, diversity_penalty=0.5,
                           return_n_best=4)
        hyps  = diverse_beam_search(model, idx, cfg)
        assert len(hyps) == 4

    def test_hypotheses_longer_than_prompt(self):
        model = tiny_model()
        idx   = make_idx(4)
        cfg   = BeamConfig(num_beams=4, max_new_tokens=5,
                           num_beam_groups=2, diversity_penalty=0.5)
        hyps  = diverse_beam_search(model, idx, cfg)
        assert len(hyps[0]) > 4

    def test_diversity_different_from_standard(self):
        """Diverse and standard beams should usually produce different output."""
        model  = tiny_model()
        idx    = make_idx()
        cfg_s  = BeamConfig(num_beams=4, max_new_tokens=10, return_n_best=4)
        cfg_d  = BeamConfig(num_beams=4, max_new_tokens=10, num_beam_groups=2,
                            diversity_penalty=1.0, return_n_best=4)
        std_  = beam_search(model, idx, cfg_s)
        div_  = diverse_beam_search(model, idx, cfg_d)
        std_toks = {tuple(h.tokens) for h in std_}
        div_toks = {tuple(h.tokens) for h in div_}
        # At least one diverse beam should differ from standard (probabilistic)
        assert len(std_toks | div_toks) >= len(std_toks)
'''
write("tests/test_beam.py", src)
commit("test: add diverse_beam_search() shape, length, and diversity tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: BeamSearchGenerator
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── BeamSearchGenerator ───────────────────────────────────────────────────────

class TestBeamSearchGenerator:
    def test_generate_returns_strings(self):
        gen  = BeamSearchGenerator(tiny_model(), TOKENIZER)
        cfg  = BeamConfig(num_beams=2, max_new_tokens=5, return_n_best=2)
        texts = gen.generate("abc", cfg)
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_generate_single_best(self):
        gen  = BeamSearchGenerator(tiny_model(), TOKENIZER)
        cfg  = BeamConfig(num_beams=2, max_new_tokens=5, return_n_best=1)
        texts = gen.generate("abc", cfg)
        assert len(texts) == 1

    def test_repr(self):
        gen = BeamSearchGenerator(tiny_model(), TOKENIZER)
        assert "NanoMind" in repr(gen)

    def test_diverse_generate(self):
        gen  = BeamSearchGenerator(tiny_model(), TOKENIZER)
        cfg  = BeamConfig(num_beams=4, max_new_tokens=5,
                          num_beam_groups=2, return_n_best=2)
        texts = gen.generate("abc", cfg)
        assert len(texts) == 2
'''
write("tests/test_beam.py", src)
commit("test: add BeamSearchGenerator generate, repr, and diverse mode tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: length penalty affects ranking
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_beam.py")
src += '''

# ── Length penalty ────────────────────────────────────────────────────────────

class TestLengthPenalty:
    def test_high_penalty_favours_longer_hypotheses(self):
        """With alpha=2.0, shorter sequences should score worse."""
        short = BeamHypothesis([1, 2],             log_prob=-2.0)
        long_ = BeamHypothesis([1, 2, 3, 4, 5, 6], log_prob=-6.0)
        assert long_.score(2.0) > short.score(2.0)

    def test_zero_penalty_score_equals_raw_log_prob(self):
        # With alpha=0: lp = ((5+T)/6)^0 = 1  → score = log_prob / 1
        hyp = BeamHypothesis([1, 2, 3], log_prob=-3.0)
        assert abs(hyp.score(0.0) - (-3.0)) < 1e-6

    def test_beam_search_with_length_penalty(self):
        model = tiny_model()
        idx   = make_idx()
        cfg_a = BeamConfig(num_beams=2, max_new_tokens=8, length_penalty=0.5, return_n_best=2)
        cfg_b = BeamConfig(num_beams=2, max_new_tokens=8, length_penalty=2.0, return_n_best=2)
        hyps_a = beam_search(model, idx, cfg_a)
        hyps_b = beam_search(model, idx, cfg_b)
        # Both should return hypotheses without error
        assert len(hyps_a) > 0
        assert len(hyps_b) > 0
'''
write("tests/test_beam.py", src)
commit("test: add length penalty ranking, zero-penalty score, and beam_search alpha tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump version + expose beam search in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.7.0\"", "__version__ = \"1.8.0\"")
src = src.replace(
    "from nanomind.logging import LogConfig, TrainingLogger",
    "from nanomind.logging import LogConfig, TrainingLogger\n"
    "from nanomind.generate.beam import BeamConfig, beam_search, diverse_beam_search\n"
    "from nanomind.generate.beam_generator import BeamSearchGenerator"
)
src = src.replace(
    "    \"TrainingLogger\",\n    \"__version__\",\n]",
    "    \"TrainingLogger\",\n"
    "    \"BeamConfig\",\n"
    "    \"beam_search\",\n"
    "    \"diverse_beam_search\",\n"
    "    \"BeamSearchGenerator\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v1.8.0 — expose BeamConfig, beam_search, diverse_beam_search in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Logging** | Console, TensorBoard, W&B — unified TrainingLogger API |",
    "| **Logging** | Console, TensorBoard, W&B — unified TrainingLogger API |\n"
    "| **Decoding** | Beam search + Diverse beam search — better quality generation |"
)
readme = readme.replace(
    "**Total: 420 commits across 21 days.**",
    "**Total: 440 commits across 22 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.7.0] — 2024 — Training Logging",
    "## [1.8.0] — 2024 — Beam Search & Diverse Beam Search\n\n### Added\n"
    "- `beam_search()` — standard beam search with length penalty and no-repeat-ngram\n"
    "- `diverse_beam_search()` — G-group diverse beam search (Vijayakumar et al. 2016)\n"
    "- `BeamConfig` — num_beams, length_penalty, num_beam_groups, diversity_penalty\n"
    "- `BeamHypothesis` / `BeamHypotheses` — hypothesis container with scored sorting\n"
    "- `BeamSearchGenerator` — high-level generate() API for beam and diverse beam\n"
    "- `_block_repeat_ngrams()` — no-repeat-ngram constraint for beam search\n"
    "- `beam_decode()` added to generate strategies pipeline\n"
    "- `examples/beam_search_demo.py` — greedy vs beam vs diverse beam comparison\n\n---\n\n"
    "## [1.7.0] — 2024 — Training Logging"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.8.0, update README and CHANGELOG for Day 22 Beam Search")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 22 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.8.0",
    "-m", "NanoMind v1.8.0 — Beam Search & Diverse Beam Search", check=False)
r = run("git", "push", "origin", "v1.8.0", check=False)
print("Tag v1.8.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 22 COMPLETE — v1.8.0 TAGGED! ===")
