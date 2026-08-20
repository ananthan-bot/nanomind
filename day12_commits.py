"""
day12_commits.py — 20 atomic commits for Day 12: Evaluation & Metrics.
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

print("\n=== DAY 12: Evaluation & Metrics — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — eval package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py", '"""NanoMind evaluation and metrics sub-package."""\n')
commit("feat: add nanomind/eval/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — perplexity() core metric
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/metrics.py", '''\
"""
nanomind/eval/metrics.py — Language model evaluation metrics.

Implements the standard LM evaluation metrics:
- Perplexity (PPL): exp(cross-entropy loss) — lower is better
- Bits per character (BPC): cross-entropy in bits — lower is better
- Token accuracy: fraction of correct next-token predictions
- Top-k accuracy: correct token in top-k predictions
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def perplexity(loss: float) -> float:
    """
    Compute perplexity from a cross-entropy loss value.

    PPL = exp(H)  where H is the average cross-entropy in nats.

    Lower perplexity = better language model.
    A uniform model over V tokens has PPL = V.

    Args:
        loss: Average cross-entropy loss (in nats).

    Returns:
        Perplexity as a float.
    """
    return math.exp(min(loss, 20.0))   # cap to avoid inf for very bad models


def bits_per_character(loss: float) -> float:
    """
    Compute bits-per-character (BPC) from a cross-entropy loss.

    BPC = H / log(2)   (converts nats to bits)

    Commonly used for character-level language models.
    A good model achieves BPC ≈ 1.0-1.5 on English text.

    Args:
        loss: Average cross-entropy loss (in nats).

    Returns:
        BPC as a float.
    """
    return loss / math.log(2)


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute the fraction of tokens where the argmax prediction is correct.

    Args:
        logits:  Model output ``(B, T, vocab_size)`` or ``(N, vocab_size)``
        targets: Ground-truth token IDs ``(B, T)`` or ``(N,)``

    Returns:
        Accuracy as a float in [0, 1].
    """
    if logits.dim() == 3:
        logits  = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    preds   = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute the fraction of tokens where the correct token is in the top-K predictions.

    Args:
        logits:  Model output ``(B, T, vocab_size)`` or ``(N, vocab_size)``
        targets: Ground-truth token IDs ``(B, T)`` or ``(N,)``
        k:       Number of top predictions to consider.

    Returns:
        Top-K accuracy as a float in [0, 1].
    """
    if logits.dim() == 3:
        logits  = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    _, top_preds = logits.topk(k, dim=-1)
    correct = top_preds.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean().item()
''')
commit("feat: add perplexity(), bits_per_character(), token_accuracy(), top_k_accuracy()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — cross_entropy_on_batch() helper
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/eval/metrics.py")
src += '''

def cross_entropy_on_batch(
    model: "torch.nn.Module",
    x: torch.Tensor,
    y: torch.Tensor,
    device: "torch.device | None" = None,
) -> float:
    """
    Compute cross-entropy loss for a single batch without gradients.

    Args:
        model:  The NanoMind model.
        x:      Input token IDs ``(B, T)``
        y:      Target token IDs ``(B, T)``
        device: Device to move tensors to.

    Returns:
        Scalar loss as a float.
    """
    import torch
    model.eval()
    with torch.no_grad():
        if device is not None:
            x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
    return loss.item()
'''
write("nanomind/eval/metrics.py", src)
commit("feat: add cross_entropy_on_batch() — no-grad loss computation for a single batch")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — EvalConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/config.py", '''\
"""
nanomind/eval/config.py — Evaluation configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Configuration for :class:`~nanomind.eval.Evaluator`.

    Attributes:
        max_batches:    Maximum number of batches to evaluate (0 = all).
        compute_acc:    Whether to compute token accuracy.
        compute_top_k:  Whether to compute top-K accuracy.
        top_k:          K value for top-K accuracy.
        compute_bpc:    Whether to compute bits-per-character.
        device:         Device for evaluation (``"auto"`` = use model device).
    """

    max_batches:   int   = 0
    compute_acc:   bool  = True
    compute_top_k: bool  = True
    top_k:         int   = 5
    compute_bpc:   bool  = True
    device:        str   = "auto"

    def __post_init__(self) -> None:
        assert self.max_batches >= 0
        assert self.top_k >= 1
''')
commit("feat: add EvalConfig dataclass for controlling evaluation behaviour")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — EvalResult dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/result.py", '''\
"""
nanomind/eval/result.py — Structured evaluation result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from nanomind.eval.metrics import perplexity, bits_per_character


@dataclass
class EvalResult:
    """
    Container for all evaluation metrics.

    Attributes:
        loss:         Mean cross-entropy loss.
        perplexity:   exp(loss) — lower is better.
        bpc:          Bits-per-character.
        accuracy:     Top-1 token prediction accuracy.
        top_k_acc:    Top-K token prediction accuracy.
        n_batches:    Number of batches evaluated.
        n_tokens:     Total tokens evaluated.
    """

    loss:      float = float("nan")
    ppl:       float = float("nan")
    bpc:       float = float("nan")
    accuracy:  float = float("nan")
    top_k_acc: float = float("nan")
    n_batches: int   = 0
    n_tokens:  int   = 0

    @classmethod
    def from_loss(
        cls,
        loss: float,
        accuracy: float = float("nan"),
        top_k_acc: float = float("nan"),
        n_batches: int = 0,
        n_tokens: int = 0,
    ) -> "EvalResult":
        """Build an EvalResult from a loss value."""
        return cls(
            loss=loss,
            ppl=perplexity(loss),
            bpc=bits_per_character(loss),
            accuracy=accuracy,
            top_k_acc=top_k_acc,
            n_batches=n_batches,
            n_tokens=n_tokens,
        )

    def __str__(self) -> str:
        lines = [
            f"loss={self.loss:.4f}",
            f"ppl={self.ppl:.2f}",
            f"bpc={self.bpc:.4f}",
        ]
        if not (self.accuracy != self.accuracy):   # not nan
            lines.append(f"acc={self.accuracy:.4f}")
        if not (self.top_k_acc != self.top_k_acc):
            lines.append(f"top_k_acc={self.top_k_acc:.4f}")
        return "EvalResult(" + " | ".join(lines) + ")"
''')
commit("feat: add EvalResult dataclass with loss, ppl, bpc, accuracy, and from_loss()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Evaluator class skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/evaluator.py", '''\
"""
nanomind/eval/evaluator.py — High-level evaluation runner for NanoMind.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult
from nanomind.eval.metrics import token_accuracy, top_k_accuracy
from nanomind.utils.logger import get_logger


class Evaluator:
    """
    Runs evaluation over a DataLoader and collects metrics.

    Args:
        model:  The NanoMind model to evaluate.
        cfg:    :class:`~nanomind.eval.EvalConfig`.
        device: Target device (None = auto-detect from model).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: EvalConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg or EvalConfig()
        self.device = device or next(model.parameters()).device
        self.log    = get_logger("evaluator")
''')
commit("feat: add Evaluator class skeleton with model, config, and device")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Evaluator.evaluate_perplexity()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/eval/evaluator.py")
src += '''
    @torch.no_grad()
    def evaluate_perplexity(self, loader: DataLoader) -> EvalResult:
        """
        Evaluate cross-entropy loss and perplexity over a DataLoader.

        Args:
            loader: DataLoader yielding ``(x, y)`` batches.

        Returns:
            :class:`~nanomind.eval.EvalResult` with loss, ppl, and bpc.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for i, (x, y) in enumerate(loader):
            if self.cfg.max_batches > 0 and i >= self.cfg.max_batches:
                break
            x, y    = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches  += 1

        mean_loss = total_loss / max(n_batches, 1)
        return EvalResult.from_loss(mean_loss, n_batches=n_batches)
'''
write("nanomind/eval/evaluator.py", src)
commit("feat: implement Evaluator.evaluate_perplexity() over a DataLoader")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Evaluator.evaluate_accuracy()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/eval/evaluator.py")
src += '''
    @torch.no_grad()
    def evaluate_accuracy(self, loader: DataLoader) -> dict[str, float]:
        """
        Compute token accuracy and top-K accuracy over a DataLoader.

        Args:
            loader: DataLoader yielding ``(x, y)`` batches.

        Returns:
            Dict with ``"accuracy"`` and ``"top_k_acc"`` keys.
        """
        self.model.eval()
        all_acc: list[float]   = []
        all_topk: list[float]  = []
        n_tokens = 0

        for i, (x, y) in enumerate(loader):
            if self.cfg.max_batches > 0 and i >= self.cfg.max_batches:
                break
            x, y    = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)

            if self.cfg.compute_acc:
                all_acc.append(token_accuracy(logits, y))
            if self.cfg.compute_top_k:
                all_topk.append(top_k_accuracy(logits, y, k=self.cfg.top_k))
            n_tokens += x.numel()

        return {
            "accuracy":  sum(all_acc) / max(len(all_acc), 1) if all_acc else float("nan"),
            "top_k_acc": sum(all_topk) / max(len(all_topk), 1) if all_topk else float("nan"),
            "n_tokens":  n_tokens,
        }
'''
write("nanomind/eval/evaluator.py", src)
commit("feat: implement Evaluator.evaluate_accuracy() — token and top-K accuracy")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — Evaluator.full_eval()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/eval/evaluator.py")
src += '''
    @torch.no_grad()
    def full_eval(self, loader: DataLoader) -> EvalResult:
        """
        Run all configured metrics in a single pass over the DataLoader.

        More efficient than calling evaluate_perplexity and evaluate_accuracy
        separately since only one forward pass per batch is needed.

        Args:
            loader: DataLoader yielding ``(x, y)`` batches.

        Returns:
            :class:`~nanomind.eval.EvalResult` with all metrics populated.
        """
        self.model.eval()
        total_loss = 0.0
        all_acc:   list[float] = []
        all_topk:  list[float] = []
        n_batches  = 0
        n_tokens   = 0

        for i, (x, y) in enumerate(loader):
            if self.cfg.max_batches > 0 and i >= self.cfg.max_batches:
                break
            x, y       = x.to(self.device), y.to(self.device)
            logits, loss = self.model(x, y)

            total_loss += loss.item()
            n_batches  += 1
            n_tokens   += x.numel()

            if self.cfg.compute_acc:
                all_acc.append(token_accuracy(logits, y))
            if self.cfg.compute_top_k:
                all_topk.append(top_k_accuracy(logits, y, k=self.cfg.top_k))

        mean_loss = total_loss / max(n_batches, 1)
        acc       = sum(all_acc) / max(len(all_acc), 1) if all_acc else float("nan")
        topk      = sum(all_topk) / max(len(all_topk), 1) if all_topk else float("nan")

        result = EvalResult.from_loss(
            mean_loss,
            accuracy=acc,
            top_k_acc=topk,
            n_batches=n_batches,
            n_tokens=n_tokens,
        )
        self.log.info(str(result))
        return result
'''
write("nanomind/eval/evaluator.py", src)
commit("feat: implement Evaluator.full_eval() — single-pass collection of all metrics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — benchmark() — time + memory profiling
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/benchmark.py", '''\
"""
nanomind/eval/benchmark.py — Inference benchmarking utilities.

Measures wall-clock time and (optionally) GPU memory usage
during a forward pass for a given batch size and sequence length.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn


def benchmark_inference(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 128,
    n_warmup: int = 3,
    n_runs: int = 20,
    device: torch.device | None = None,
) -> dict:
    """
    Benchmark model inference speed and memory usage.

    Args:
        model:      Model to benchmark.
        batch_size: Batch size for the synthetic input.
        seq_len:    Sequence length for the synthetic input.
        n_warmup:   Number of warm-up runs (not timed).
        n_runs:     Number of timed runs.
        device:     Device to benchmark on.

    Returns:
        Dict with:
        - ``mean_ms``       : mean wall-clock ms per forward pass
        - ``std_ms``        : standard deviation in ms
        - ``tokens_per_s``  : throughput in tokens/second
        - ``peak_mem_mb``   : peak GPU memory in MiB (0 on CPU)
    """
    if device is None:
        device = next(model.parameters()).device

    vocab_size = getattr(model, "cfg", None) and model.cfg.vocab_size or 256
    model.eval()
    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    std_ms  = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    tokens_per_s = batch_size * seq_len / (mean_ms / 1000)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "mean_ms":      mean_ms,
        "std_ms":       std_ms,
        "tokens_per_s": tokens_per_s,
        "peak_mem_mb":  peak_mem_mb,
    }
''')
commit("feat: add benchmark_inference() — time and memory profiling for model forward pass")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — compare_models() utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/compare.py", '''\
"""
nanomind/eval/compare.py — Side-by-side model comparison utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.evaluator import Evaluator
from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult


def compare_models(
    models: dict[str, nn.Module],
    loader: DataLoader,
    cfg: EvalConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, EvalResult]:
    """
    Evaluate multiple models on the same DataLoader and return results.

    Args:
        models: Dict mapping model name -> model.
        loader: DataLoader for evaluation.
        cfg:    EvalConfig (shared across all models).
        device: Evaluation device.

    Returns:
        Dict mapping model name -> :class:`EvalResult`.
    """
    results: dict[str, EvalResult] = {}
    for name, model in models.items():
        evaluator = Evaluator(model, cfg or EvalConfig(), device)
        results[name] = evaluator.full_eval(loader)
    return results


def print_comparison(results: dict[str, "EvalResult"]) -> None:
    """Pretty-print a model comparison table."""
    header = f"{'Model':<20} {'Loss':>8} {'PPL':>8} {'BPC':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        acc_str = f"{r.accuracy:.4f}" if r.accuracy == r.accuracy else "  N/A"
        print(
            f"{name:<20} {r.loss:>8.4f} {r.ppl:>8.2f} "
            f"{r.bpc:>8.4f} {acc_str:>8}"
        )
''')
commit("feat: add compare_models() and print_comparison() for side-by-side model comparison")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — text_generation_quality() heuristic
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/generation_quality.py", '''\
"""
nanomind/eval/generation_quality.py — Heuristic generation quality metrics.

These are lightweight, reference-free metrics that give a quick sense
of text quality without needing ground-truth references.
"""

from __future__ import annotations

import math
from collections import Counter


def type_token_ratio(text: str) -> float:
    """
    Compute the Type-Token Ratio (TTR) — a simple lexical diversity metric.

    TTR = unique_words / total_words

    A higher TTR indicates greater vocabulary diversity.
    Repetitive models tend to have low TTR.

    Args:
        text: Generated text string.

    Returns:
        TTR as a float in [0, 1].
    """
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def distinct_n(text: str, n: int = 2) -> float:
    """
    Compute distinct-N: fraction of unique N-grams over all N-grams.

    A higher distinct-N means less repetition in generated text.

    Args:
        text: Generated text string.
        n:    N-gram order.

    Returns:
        Distinct-N as a float in [0, 1].
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_fraction(text: str, n: int = 4) -> float:
    """
    Estimate the fraction of N-grams that are repeated (appear > once).

    Higher values = more repetition (worse generation quality).

    Args:
        text: Generated text string.
        n:    N-gram order.

    Returns:
        Repetition fraction in [0, 1].
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / max(len(ngrams), 1)


def generation_report(text: str) -> dict:
    """
    Compute a full generation quality report for a text string.

    Returns:
        Dict with ``ttr``, ``distinct_1``, ``distinct_2``, ``repetition_4``,
        ``n_words``, ``n_chars`` fields.
    """
    return {
        "ttr":           type_token_ratio(text),
        "distinct_1":    distinct_n(text, n=1),
        "distinct_2":    distinct_n(text, n=2),
        "repetition_4":  repetition_fraction(text, n=4),
        "n_words":       len(text.split()),
        "n_chars":       len(text),
    }
''')
commit("feat: add generation quality metrics — TTR, distinct-N, repetition fraction")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — update eval __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py", '''\
"""NanoMind evaluation and metrics sub-package.

Primary exports:
    - :class:`Evaluator`             — full evaluation runner
    - :class:`EvalConfig`            — evaluation configuration
    - :class:`EvalResult`            — structured results container

Metrics:
    - :func:`perplexity`             — exp(cross-entropy loss)
    - :func:`bits_per_character`     — loss in bits
    - :func:`token_accuracy`         — top-1 prediction accuracy
    - :func:`top_k_accuracy`         — top-K prediction accuracy
    - :func:`cross_entropy_on_batch` — per-batch loss helper

Benchmarking:
    - :func:`benchmark_inference`    — time and memory profiling

Comparison:
    - :func:`compare_models`         — evaluate multiple models side-by-side
    - :func:`print_comparison`       — pretty-print comparison table

Generation quality:
    - :func:`type_token_ratio`       — lexical diversity
    - :func:`distinct_n`             — N-gram diversity
    - :func:`repetition_fraction`    — N-gram repetition
    - :func:`generation_report`      — full quality report dict
"""

from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult
from nanomind.eval.evaluator import Evaluator
from nanomind.eval.metrics import (
    perplexity,
    bits_per_character,
    token_accuracy,
    top_k_accuracy,
    cross_entropy_on_batch,
)
from nanomind.eval.benchmark import benchmark_inference
from nanomind.eval.compare import compare_models, print_comparison
from nanomind.eval.generation_quality import (
    type_token_ratio,
    distinct_n,
    repetition_fraction,
    generation_report,
)

__all__ = [
    "EvalConfig",
    "EvalResult",
    "Evaluator",
    "perplexity",
    "bits_per_character",
    "token_accuracy",
    "top_k_accuracy",
    "cross_entropy_on_batch",
    "benchmark_inference",
    "compare_models",
    "print_comparison",
    "type_token_ratio",
    "distinct_n",
    "repetition_fraction",
    "generation_report",
]
''')
commit("refactor: export all eval components from nanomind/eval/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: perplexity and BPC
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_eval.py", '''\
"""
tests/test_eval.py — Tests for NanoMind evaluation metrics.
"""

import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model import NanoMind, ModelConfig
from nanomind.eval import (
    EvalConfig,
    EvalResult,
    Evaluator,
    perplexity,
    bits_per_character,
    token_accuracy,
    top_k_accuracy,
    type_token_ratio,
    distinct_n,
    repetition_fraction,
    generation_report,
)

VOCAB = 32
CFG   = ModelConfig(
    vocab_size=VOCAB, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NanoMind(CFG)


def make_loader(n: int = 32, b: int = 4, t: int = 8):
    tokens = torch.randint(0, VOCAB, (n + t,))
    xs = torch.stack([tokens[i:i+t]   for i in range(n)])
    ys = torch.stack([tokens[i+1:i+t+1] for i in range(n)])
    return DataLoader(TensorDataset(xs, ys), batch_size=b, drop_last=True)


# ── perplexity / BPC ──────────────────────────────────────────────────────────

class TestPerplexity:
    def test_ppl_of_zero_loss(self):
        assert perplexity(0.0) == 1.0

    def test_ppl_of_log_vocab(self):
        # Uniform model: loss = log(vocab_size), PPL = vocab_size
        loss = math.log(VOCAB)
        assert abs(perplexity(loss) - VOCAB) < 0.01

    def test_ppl_monotone_in_loss(self):
        assert perplexity(1.0) < perplexity(2.0) < perplexity(3.0)

    def test_bpc_zero_loss(self):
        assert bits_per_character(0.0) == 0.0

    def test_bpc_log2(self):
        # loss = log(2) -> BPC = 1.0
        assert abs(bits_per_character(math.log(2)) - 1.0) < 1e-9

    def test_bpc_larger_than_loss_in_nats(self):
        # BPC = loss / log(2) > loss for loss > 0
        loss = 2.5
        assert bits_per_character(loss) > loss
''')
commit("test: add perplexity() and bits_per_character() correctness tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: accuracy metrics
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Accuracy metrics ──────────────────────────────────────────────────────────

class TestAccuracyMetrics:
    def test_perfect_accuracy(self):
        vocab = 10
        targets = torch.tensor([0, 1, 2, 3])
        logits  = torch.zeros(4, vocab)
        for i, t in enumerate(targets):
            logits[i, t] = 100.0   # argmax = target
        assert token_accuracy(logits, targets) == 1.0

    def test_zero_accuracy(self):
        vocab = 10
        targets = torch.tensor([0, 0, 0, 0])
        logits  = torch.zeros(4, vocab)
        logits[:, 1] = 100.0   # argmax = 1, targets = 0
        assert token_accuracy(logits, targets) == 0.0

    def test_accuracy_in_01(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc = token_accuracy(logits, y)
        assert 0.0 <= acc <= 1.0

    def test_top_k_acc_geq_top1(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc1 = token_accuracy(logits, y)
        acc5 = top_k_accuracy(logits, y, k=5)
        assert acc5 >= acc1

    def test_top_k_perfect_if_k_equals_vocab(self, model):
        x = torch.randint(0, VOCAB, (2, 8))
        y = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits, _ = model(x)
        acc = top_k_accuracy(logits, y, k=VOCAB)
        assert acc == 1.0   # every token is in top-VOCAB
'''
write("tests/test_eval.py", src)
commit("test: add token_accuracy() and top_k_accuracy() correctness tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: EvalResult
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── EvalResult ────────────────────────────────────────────────────────────────

class TestEvalResult:
    def test_from_loss_computes_ppl(self):
        r = EvalResult.from_loss(math.log(VOCAB))
        assert abs(r.ppl - VOCAB) < 0.01

    def test_from_loss_computes_bpc(self):
        loss = math.log(2)
        r    = EvalResult.from_loss(loss)
        assert abs(r.bpc - 1.0) < 1e-6

    def test_str_contains_ppl(self):
        r = EvalResult.from_loss(1.0)
        assert "ppl" in str(r)

    def test_str_contains_loss(self):
        r = EvalResult.from_loss(1.0)
        assert "loss" in str(r)
'''
write("tests/test_eval.py", src)
commit("test: add EvalResult.from_loss(), ppl, bpc, and str() tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: Evaluator.evaluate_perplexity()
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_perplexity_returns_result(self, model):
        loader = make_loader()
        ev     = Evaluator(model, EvalConfig(), torch.device("cpu"))
        result = ev.evaluate_perplexity(loader)
        assert isinstance(result, EvalResult)
        assert result.ppl > 1.0

    def test_full_eval_all_metrics(self, model):
        loader = make_loader()
        ev     = Evaluator(model, EvalConfig(compute_acc=True, compute_top_k=True))
        result = ev.full_eval(loader)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.top_k_acc <= 1.0
        assert result.n_batches > 0

    def test_max_batches_limits_evaluation(self, model):
        loader = make_loader(n=64)
        ev1 = Evaluator(model, EvalConfig(max_batches=1))
        ev2 = Evaluator(model, EvalConfig(max_batches=0))
        r1  = ev1.full_eval(loader)
        r2  = ev2.full_eval(loader)
        assert r1.n_batches == 1
        assert r2.n_batches > 1
'''
write("tests/test_eval.py", src)
commit("test: add Evaluator.evaluate_perplexity() and full_eval() tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: generation quality metrics
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Generation quality ────────────────────────────────────────────────────────

class TestGenerationQuality:
    def test_ttr_perfect_diversity(self):
        text = "the cat sat on the mat"
        assert 0.0 < type_token_ratio(text) <= 1.0

    def test_ttr_all_unique(self):
        text = "a b c d e f"
        assert type_token_ratio(text) == 1.0

    def test_ttr_all_same(self):
        text = "a a a a a a"
        assert type_token_ratio(text) == 1 / 6

    def test_distinct_1_all_unique(self):
        text = "a b c d e f"
        assert distinct_n(text, n=1) == 1.0

    def test_distinct_2_range(self):
        text = "the cat sat on the mat"
        assert 0.0 <= distinct_n(text, n=2) <= 1.0

    def test_repetition_zero_for_unique_text(self):
        text = "a b c d e f g h"
        assert repetition_fraction(text, n=2) == 0.0

    def test_generation_report_keys(self):
        report = generation_report("hello world")
        assert "ttr" in report
        assert "distinct_2" in report
        assert "n_words" in report
'''
write("tests/test_eval.py", src)
commit("test: add generation quality metric tests (TTR, distinct-N, repetition)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: EvalConfig validation
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── EvalConfig ────────────────────────────────────────────────────────────────

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.compute_acc is True
        assert cfg.top_k == 5

    def test_invalid_max_batches(self):
        with pytest.raises(AssertionError):
            EvalConfig(max_batches=-1)

    def test_invalid_top_k(self):
        with pytest.raises(AssertionError):
            EvalConfig(top_k=0)
'''
write("tests/test_eval.py", src)
commit("test: add EvalConfig validation tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 12 | Evaluation & metrics | 🔜 |",
    "| 12 | Evaluation & metrics | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Generation: greedy, temperature, top-k, top-p, min-p, beam search, Generator, stream() (Day 11)",
    "- Generation: greedy, temperature, top-k, top-p, min-p, beam search, Generator, stream() (Day 11)\n- Evaluation: PPL, BPC, accuracy, top-K, Evaluator, benchmark, generation quality metrics (Day 12)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 12 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 12 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 12 COMPLETE ===")
