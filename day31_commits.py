"""
day31_commits.py — 15 atomic commits for Day 31: Benchmarking & Evaluation Suite.
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

print("\n=== DAY 31: Benchmarking & Evaluation Suite — 15 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — eval package init
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py",
      '"""NanoMind Eval sub-package — Benchmarking and evaluation utilities."""\n')
commit("feat: add nanomind/eval/ package skeleton for benchmarking and evaluation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — BenchmarkConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/config.py", '''\
"""
nanomind/eval/config.py — Benchmark and evaluation configuration.

Benchmarking a language model covers three axes:

  1. Quality   — How good are the model predictions?
                 Perplexity: exp(-1/N Σ log p(w_i)) — lower is better
                 Top-K accuracy: fraction of correct tokens in top K predictions

  2. Speed     — How fast does the model generate?
                 Throughput: tokens per second (prefill + decode)
                 Latency: milliseconds per token (time to first token)

  3. Memory    — How much memory does the model use?
                 Parameter memory: model weights in MB
                 Peak activation memory: max GPU MB during forward pass

Reference baselines:
  GPT-2 small (117M): ~30-50 tokens/sec on a single A100
  LLaMA 7B:           ~80-120 tokens/sec on 8×A100 with Flash Attention
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """
    Configuration for NanoMind evaluation benchmarks.

    Attributes:
        batch_size:     Batch size for perplexity / throughput evaluation.
        seq_len:        Sequence length for benchmarks.
        n_warmup:       Warmup iterations before timing (avoids JIT overhead).
        n_trials:       Number of timed iterations for throughput benchmarks.
        top_k_values:   List of K values for top-K accuracy computation.
        device:         Evaluation device (``"cpu"`` or ``"cuda"``).
        dtype:          Evaluation dtype (``"float32"`` or ``"float16"``).
    """
    batch_size:   int        = 4
    seq_len:      int        = 128
    n_warmup:     int        = 3
    n_trials:     int        = 10
    top_k_values: list[int]  = field(default_factory=lambda: [1, 5, 10])
    device:       str        = "cpu"
    dtype:        str        = "float32"

    def __post_init__(self) -> None:
        assert self.batch_size  >= 1
        assert self.seq_len     >= 1
        assert self.n_warmup    >= 0
        assert self.n_trials    >= 1
        assert self.dtype in ("float32", "float16", "bfloat16")
''')
commit("feat: add BenchmarkConfig — batch_size, seq_len, n_warmup, n_trials, top_k_values")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — perplexity evaluation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/perplexity.py", '''\
"""
nanomind/eval/perplexity.py — Perplexity evaluation for language models.

Perplexity (PPL) is the standard intrinsic metric for language model quality:

  PPL = exp(-1/N Σ_{i=1}^{N} log p_θ(w_i | w_1 ... w_{i-1}))

Interpretation:
  PPL = 1    → perfect model (assigns probability 1 to every correct token)
  PPL = V    → random model (assigns uniform probability, V = vocab size)
  PPL ↓      → better model

Typical values:
  Character-level models: PPL 2-5 (small alphabet)
  Word-level models:      PPL 20-100 on PTB, WikiText
  GPT-2 (1.5B):          PPL ~17.48 on WikiText-103
  LLaMA 2 (70B):         PPL ~3.3 on WikiText-2

Lower perplexity = better language understanding.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(
    model:      nn.Module,
    loader:     DataLoader,
    device:     str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict:
    """
    Compute perplexity of a language model on a dataset.

    Args:
        model:       Language model with ``forward(x, y) -> (logits, loss)``.
        loader:      DataLoader of ``(x, y)`` token batches.
        device:      Evaluation device.
        max_batches: Optionally limit to this many batches.

    Returns:
        Dict with ``perplexity``, ``nll`` (mean negative log-likelihood),
        ``n_tokens``, ``n_batches``.
    """
    model.eval()
    device   = torch.device(device)
    total_nll, n_tokens, n_batches = 0.0, 0, 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y  = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            total_nll += loss.item() * y.numel()
            n_tokens  += y.numel()
        n_batches += 1

    if n_tokens == 0:
        return {"perplexity": float("inf"), "nll": float("inf"),
                "n_tokens": 0, "n_batches": 0}

    mean_nll   = total_nll / n_tokens
    perplexity = math.exp(min(mean_nll, 100))   # cap at exp(100) to avoid overflow
    return {
        "perplexity": perplexity,
        "nll":        mean_nll,
        "n_tokens":   n_tokens,
        "n_batches":  n_batches,
        "bits_per_char": mean_nll / math.log(2),
    }


def perplexity_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute perplexity directly from logits and targets.

    Args:
        logits:  ``(N, V)`` model output logits.
        targets: ``(N,)`` target token IDs.

    Returns:
        Perplexity (scalar float).
    """
    nll = F.cross_entropy(logits, targets).item()
    return math.exp(min(nll, 100))
''')
commit("feat: add compute_perplexity() — NLL-based PPL with bits_per_char, dataset evaluation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — throughput benchmark
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/throughput.py", '''\
"""
nanomind/eval/throughput.py — Throughput and latency benchmarking.

Measures:
  - Prefill throughput: tokens/sec for the full forward pass
  - Decode latency:     ms/token for autoregressive generation
  - Memory footprint:   model parameter + activation memory
"""

from __future__ import annotations

import time
import math
import torch
import torch.nn as nn

from nanomind.eval.config import BenchmarkConfig


def benchmark_prefill(
    model:  nn.Module,
    cfg:    BenchmarkConfig,
) -> dict:
    """
    Benchmark forward pass (prefill) throughput.

    Args:
        model: Language model.
        cfg:   Benchmark configuration.

    Returns:
        Dict with ``tokens_per_sec``, ``ms_per_batch``, ``n_tokens``.
    """
    model.eval()
    device = torch.device(cfg.device)
    model  = model.to(device)
    V      = next(m for m in model.modules()
                  if isinstance(m, nn.Embedding)).num_embeddings

    x = torch.randint(0, V, (cfg.batch_size, cfg.seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(cfg.n_warmup):
            model(x)

    # Timed trials
    times = []
    with torch.no_grad():
        for _ in range(cfg.n_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    n_tokens     = cfg.batch_size * cfg.seq_len
    mean_ms      = sum(times) / len(times) * 1000
    tokens_per_s = n_tokens / (sum(times) / len(times))

    return {
        "tokens_per_sec":    tokens_per_s,
        "ms_per_batch":      mean_ms,
        "n_tokens_per_step": n_tokens,
        "mean_latency_ms":   mean_ms,
        "min_latency_ms":    min(times) * 1000,
        "max_latency_ms":    max(times) * 1000,
    }


def benchmark_memory(model: nn.Module) -> dict:
    """
    Compute model memory usage (parameters + buffers).

    Args:
        model: PyTorch model.

    Returns:
        Dict with ``params_mb``, ``buffers_mb``, ``total_mb``, ``n_params``.
    """
    params_bytes  = sum(p.nbytes for p in model.parameters())
    buffer_bytes  = sum(b.nbytes for b in model.buffers())
    n_params      = sum(p.numel() for p in model.parameters())
    return {
        "params_mb":  params_bytes  / (1024 ** 2),
        "buffers_mb": buffer_bytes  / (1024 ** 2),
        "total_mb":   (params_bytes + buffer_bytes) / (1024 ** 2),
        "n_params":   n_params,
    }


def full_benchmark_report(
    model: nn.Module,
    cfg:   BenchmarkConfig,
    name:  str = "Model",
) -> str:
    """
    Run prefill + memory benchmarks and return a formatted report.

    Args:
        model: Language model to benchmark.
        cfg:   Benchmark configuration.
        name:  Model display name.

    Returns:
        Multi-line formatted report string.
    """
    mem    = benchmark_memory(model)
    speed  = benchmark_prefill(model, cfg)

    return (
        f"Benchmark: {name}\n"
        f"{'─'*50}\n"
        f"  Parameters : {mem['n_params']:>12,}\n"
        f"  Memory     : {mem['total_mb']:>10.2f} MB\n"
        f"  Throughput : {speed['tokens_per_sec']:>10,.0f} tokens/sec\n"
        f"  Latency    : {speed['ms_per_batch']:>10.2f} ms/batch\n"
        f"  Batch      : {cfg.batch_size} × {cfg.seq_len} tokens\n"
    )
''')
commit("feat: add benchmark_prefill(), benchmark_memory(), full_benchmark_report()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — accuracy metrics
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/accuracy.py", '''\
"""
nanomind/eval/accuracy.py — Token prediction accuracy metrics.

Top-K accuracy: the fraction of time the correct next token appears
in the model\'s top-K predictions. Standard metrics in LLM evaluation.

  Top-1 accuracy = argmax match (greedy accuracy)
  Top-5 accuracy = correct token in the 5 most probable predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def top_k_accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    k:       int = 1,
) -> float:
    """
    Compute top-K token prediction accuracy.

    Args:
        logits:  ``(N, V)`` model output logits.
        targets: ``(N,)`` ground-truth token IDs.
        k:       Number of top predictions to consider.

    Returns:
        Accuracy in [0, 1].
    """
    topk = logits.topk(k, dim=-1).indices   # (N, k)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def multi_k_accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    k_values: list[int] = (1, 5, 10),
) -> dict:
    """
    Compute top-K accuracy for multiple values of K.

    Args:
        logits:   ``(N, V)`` model output logits.
        targets:  ``(N,)`` ground-truth token IDs.
        k_values: List of K values to compute.

    Returns:
        Dict mapping ``top_{k}`` → accuracy float.
    """
    return {f"top_{k}": top_k_accuracy(logits, targets, k) for k in k_values}


@torch.no_grad()
def evaluate_accuracy(
    model:       nn.Module,
    loader:      DataLoader,
    device:      str | torch.device = "cpu",
    k_values:    list[int] = (1, 5),
    max_batches: int | None = None,
) -> dict:
    """
    Evaluate top-K accuracy of a model over a full dataset.

    Args:
        model:       Language model returning ``(logits, loss)``.
        loader:      DataLoader of ``(x, y)`` batches.
        device:      Evaluation device.
        k_values:    List of K values.
        max_batches: Optionally limit evaluation batches.

    Returns:
        Dict with ``top_1``, ``top_5``, etc., plus ``n_tokens``.
    """
    model.eval()
    device     = torch.device(device)
    totals     = {f"top_{k}": 0.0 for k in k_values}
    n_batches  = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y    = x.to(device), y.to(device)
        logits, _ = model(x)
        # Flatten: each position predicts the next token
        N, T, V = logits.shape
        flat_logits  = logits[:, :-1].reshape(-1, V)
        flat_targets = y[:, 1:].reshape(-1)

        for k in k_values:
            totals[f"top_{k}"] += top_k_accuracy(flat_logits, flat_targets, k)
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in totals.items()} | {"n_batches": n_batches}
''')
commit("feat: add top_k_accuracy(), multi_k_accuracy(), evaluate_accuracy() — token metrics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — EvalRunner
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/runner.py", '''\
"""
nanomind/eval/runner.py — Unified evaluation runner for NanoMind models.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.config import BenchmarkConfig
from nanomind.eval.perplexity import compute_perplexity
from nanomind.eval.throughput import benchmark_prefill, benchmark_memory
from nanomind.eval.accuracy import evaluate_accuracy
from nanomind.utils.logger import get_logger

log = get_logger("eval.runner")


class EvalRunner:
    """
    Unified evaluation runner: perplexity, accuracy, throughput, memory.

    Runs all benchmark dimensions in one call and returns a structured report.

    Args:
        model:  Language model to evaluate.
        cfg:    Benchmark configuration.
        name:   Model name for reports.

    Example::

        runner  = EvalRunner(model, BenchmarkConfig(), name="NanoMind-128d")
        results = runner.run(val_loader)
        print(runner.format_report(results))
    """

    def __init__(
        self,
        model: nn.Module,
        cfg:   BenchmarkConfig | None = None,
        name:  str = "Model",
    ) -> None:
        self.model  = model
        self.cfg    = cfg or BenchmarkConfig()
        self.name   = name
        self.device = torch.device(self.cfg.device)

    def run(
        self,
        loader:      DataLoader | None = None,
        max_batches: int | None = 50,
    ) -> dict:
        """
        Run the full evaluation suite.

        Args:
            loader:      DataLoader for perplexity + accuracy evaluation.
            max_batches: Max batches for perplexity/accuracy to keep it fast.

        Returns:
            Dict with all metrics from all benchmark dimensions.
        """
        results: dict = {"model": self.name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        # Memory
        log.info(f"Evaluating memory for {self.name}...")
        mem = benchmark_memory(self.model)
        results.update(mem)

        # Throughput
        log.info(f"Evaluating throughput for {self.name}...")
        speed = benchmark_prefill(self.model, self.cfg)
        results.update(speed)

        # Perplexity + accuracy (if loader provided)
        if loader is not None:
            log.info(f"Evaluating perplexity for {self.name}...")
            ppl = compute_perplexity(self.model, loader, self.device, max_batches)
            results.update(ppl)

            log.info(f"Evaluating accuracy for {self.name}...")
            acc = evaluate_accuracy(self.model, loader, self.device,
                                    self.cfg.top_k_values, max_batches)
            results.update(acc)

        return results

    def format_report(self, results: dict) -> str:
        """Format an evaluation results dict as a human-readable report."""
        lines = [
            f"Evaluation Report: {results.get('model', 'Unknown')}",
            f"  Timestamp : {results.get('timestamp', '')}",
            "─" * 52,
        ]
        if "n_params" in results:
            lines.append(f"  Parameters: {results['n_params']:>14,}")
            lines.append(f"  Memory    : {results.get('total_mb', 0):>12.2f} MB")
        if "tokens_per_sec" in results:
            lines.append(f"  Throughput: {results['tokens_per_sec']:>12,.0f} tok/s")
            lines.append(f"  Latency   : {results.get('ms_per_batch', 0):>12.2f} ms/batch")
        if "perplexity" in results:
            lines.append(f"  Perplexity: {results['perplexity']:>12.4f}")
            lines.append(f"  Bits/char : {results.get('bits_per_char', 0):>12.4f}")
        for k in self.cfg.top_k_values:
            key = f"top_{k}"
            if key in results:
                lines.append(f"  Top-{k:<6}: {results[key]:>11.2%}")
        return "\n".join(lines)

    def compare(self, other_results: list[dict]) -> str:
        """Format a comparison table for multiple model evaluation results."""
        cols = ["model", "n_params", "total_mb", "tokens_per_sec", "perplexity", "top_1"]
        header = f"{'Model':<20} {'Params':>10} {'MB':>8} {'Tok/s':>10} {'PPL':>10} {'Top-1':>8}"
        sep    = "─" * len(header)
        rows   = [header, sep]
        for r in other_results:
            row = (
                f"{r.get('model','?'):<20} "
                f"{r.get('n_params',0):>10,} "
                f"{r.get('total_mb',0):>8.1f} "
                f"{r.get('tokens_per_sec',0):>10,.0f} "
                f"{r.get('perplexity',float('inf')):>10.2f} "
                f"{r.get('top_1',0):>8.2%}"
            )
            rows.append(row)
        return "\n".join(rows)
''')
commit("feat: add EvalRunner — run(), format_report(), compare() unified evaluation interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — update eval __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py", '''\
"""NanoMind Eval sub-package — Benchmarking and evaluation suite.

Measures three dimensions of model quality:
  1. Quality:    perplexity, bits-per-character, top-K accuracy
  2. Speed:      tokens/sec throughput, ms/batch latency
  3. Memory:     parameter MB, buffer MB

Primary exports:
    - :class:`EvalRunner`           — unified run() + format_report() + compare()
    - :class:`BenchmarkConfig`      — batch_size, seq_len, n_trials, top_k_values
    - :func:`compute_perplexity`    — NLL-based PPL over a dataset
    - :func:`perplexity_from_logits`— PPL from logits directly
    - :func:`benchmark_prefill`     — tokens/sec forward pass timing
    - :func:`benchmark_memory`      — parameter + buffer memory in MB
    - :func:`full_benchmark_report` — formatted speed + memory report
    - :func:`top_k_accuracy`        — single-K token accuracy
    - :func:`multi_k_accuracy`      — multiple-K accuracy dict
    - :func:`evaluate_accuracy`     — full dataset accuracy evaluation
"""

from nanomind.eval.config import BenchmarkConfig
from nanomind.eval.perplexity import compute_perplexity, perplexity_from_logits
from nanomind.eval.throughput import benchmark_prefill, benchmark_memory, full_benchmark_report
from nanomind.eval.accuracy import top_k_accuracy, multi_k_accuracy, evaluate_accuracy
from nanomind.eval.runner import EvalRunner

__all__ = [
    "BenchmarkConfig",
    "EvalRunner",
    "compute_perplexity",
    "perplexity_from_logits",
    "benchmark_prefill",
    "benchmark_memory",
    "full_benchmark_report",
    "top_k_accuracy",
    "multi_k_accuracy",
    "evaluate_accuracy",
]
''')
commit("refactor: export all eval components from nanomind/eval/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — example: benchmark_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/benchmark_demo.py", '''\
"""
examples/benchmark_demo.py — NanoMind benchmarking demo.

Trains two models (large vs small) and compares them across:
  - Perplexity (lower = better language model quality)
  - Top-K accuracy (higher = better token predictions)
  - Throughput (tokens/sec, higher = faster inference)
  - Memory (MB, lower = cheaper to serve)

Usage:
    python examples/benchmark_demo.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.eval import (
    BenchmarkConfig, EvalRunner,
    compute_perplexity, benchmark_memory, benchmark_prefill,
    top_k_accuracy, multi_k_accuracy,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog " * 80
tokenizer = CharTokenizer().build(CORPUS)
ids       = torch.tensor(tokenizer.encode(CORPUS))
BLOCK     = 32

xs = torch.stack([ids[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([ids[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=16, shuffle=False)

def make_model(d, layers):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK,
                      d_model=d, n_layers=layers, n_heads=4, dropout=0.0)
    return NanoMind(cfg)

# ── Build models ──────────────────────────────────────────────────────────────
large_model = make_model(128, 4)
small_model = make_model(64,  2)

# ── EvalRunner ────────────────────────────────────────────────────────────────
bench_cfg = BenchmarkConfig(batch_size=4, seq_len=BLOCK, n_warmup=2, n_trials=5)

runner_large = EvalRunner(large_model, bench_cfg, name="NanoMind-128d-4L")
runner_small = EvalRunner(small_model, bench_cfg, name="NanoMind-64d-2L")

results_large = runner_large.run(loader, max_batches=20)
results_small = runner_small.run(loader, max_batches=20)

# ── Reports ───────────────────────────────────────────────────────────────────
print(runner_large.format_report(results_large))
print()
print(runner_small.format_report(results_small))
print()
print("Comparison Table:")
print(runner_large.compare([results_large, results_small]))
''')
commit("feat: add examples/benchmark_demo.py — large vs small model perplexity/speed/memory")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — test: BenchmarkConfig + perplexity
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_eval.py", '''\
"""
tests/test_eval.py — Tests for benchmarking and evaluation suite.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.eval import (
    BenchmarkConfig, EvalRunner,
    compute_perplexity, perplexity_from_logits,
    benchmark_prefill, benchmark_memory, full_benchmark_report,
    top_k_accuracy, multi_k_accuracy, evaluate_accuracy,
)

CORPUS = "abcdefghij " * 8
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)

def tiny_loader():
    xs = torch.randint(0, VOCAB, (16, T))
    ys = torch.randint(0, VOCAB, (16, T))
    return DataLoader(TensorDataset(xs, ys), batch_size=B)


# ── BenchmarkConfig ───────────────────────────────────────────────────────────

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.batch_size == 4
        assert 1 in cfg.top_k_values

    def test_invalid_batch_size(self):
        with pytest.raises(AssertionError):
            BenchmarkConfig(batch_size=0)

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            BenchmarkConfig(dtype="int8")
''')
commit("test: add BenchmarkConfig defaults, invalid batch_size, dtype tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — test: perplexity functions
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Perplexity ────────────────────────────────────────────────────────────────

class TestPerplexity:
    def test_compute_perplexity_keys(self):
        model  = tiny_model()
        loader = tiny_loader()
        result = compute_perplexity(model, loader, max_batches=3)
        for k in ("perplexity", "nll", "n_tokens", "bits_per_char"):
            assert k in result

    def test_perplexity_positive(self):
        model  = tiny_model()
        loader = tiny_loader()
        r      = compute_perplexity(model, loader, max_batches=3)
        assert r["perplexity"] > 1.0

    def test_perplexity_from_logits(self):
        logits  = torch.randn(8, VOCAB)
        targets = torch.randint(0, VOCAB, (8,))
        ppl     = perplexity_from_logits(logits, targets)
        assert ppl > 1.0
        assert math.isfinite(ppl)

    def test_perfect_prediction_low_ppl(self):
        """Near-perfect predictions → PPL close to 1."""
        V      = 10
        logits = torch.zeros(4, V)
        tgts   = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 100.0   # model assigns all prob to token 0
        ppl    = perplexity_from_logits(logits, tgts)
        assert ppl < 1.01
'''
write("tests/test_eval.py", src)
commit("test: add compute_perplexity keys/positive, perplexity_from_logits, perfect prediction tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — test: throughput + memory
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Throughput + Memory ───────────────────────────────────────────────────────

class TestThroughput:
    def test_benchmark_memory_keys(self):
        model = tiny_model()
        mem   = benchmark_memory(model)
        for k in ("params_mb", "total_mb", "n_params"):
            assert k in mem

    def test_params_positive(self):
        model = tiny_model()
        mem   = benchmark_memory(model)
        assert mem["n_params"] > 0
        assert mem["params_mb"] > 0.0

    def test_benchmark_prefill_keys(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        speed = benchmark_prefill(model, cfg)
        for k in ("tokens_per_sec", "ms_per_batch"):
            assert k in speed

    def test_throughput_positive(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        speed = benchmark_prefill(model, cfg)
        assert speed["tokens_per_sec"] > 0.0

    def test_full_report_is_string(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0, n_trials=2)
        report = full_benchmark_report(model, cfg, name="Test")
        assert isinstance(report, str)
        assert "Test" in report
'''
write("tests/test_eval.py", src)
commit("test: add benchmark_memory keys/positive, benchmark_prefill, throughput, report tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: accuracy metrics
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── Accuracy ──────────────────────────────────────────────────────────────────

class TestAccuracy:
    def test_top_1_perfect(self):
        logits  = torch.zeros(4, VOCAB)
        targets = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 100.0
        assert top_k_accuracy(logits, targets, k=1) == 1.0

    def test_top_1_all_wrong(self):
        logits  = torch.zeros(4, VOCAB)
        targets = torch.ones(4, dtype=torch.long)
        logits[:, 0] = 100.0   # always predicts 0, targets are 1
        assert top_k_accuracy(logits, targets, k=1) == 0.0

    def test_top_k_geq_top_1(self):
        logits  = torch.randn(16, VOCAB)
        targets = torch.randint(0, VOCAB, (16,))
        acc1    = top_k_accuracy(logits, targets, k=1)
        acc5    = top_k_accuracy(logits, targets, k=5)
        assert acc5 >= acc1

    def test_multi_k_accuracy_keys(self):
        logits  = torch.randn(8, VOCAB)
        targets = torch.randint(0, VOCAB, (8,))
        result  = multi_k_accuracy(logits, targets, k_values=[1, 5])
        assert "top_1" in result
        assert "top_5" in result

    def test_evaluate_accuracy_dataset(self):
        model  = tiny_model()
        loader = tiny_loader()
        result = evaluate_accuracy(model, loader, k_values=[1, 5], max_batches=3)
        assert "top_1" in result
        assert 0.0 <= result["top_1"] <= 1.0
'''
write("tests/test_eval.py", src)
commit("test: add top_k perfect/wrong, top_k>=top_1, multi_k keys, evaluate_accuracy dataset tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: EvalRunner
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_eval.py")
src += '''

# ── EvalRunner ────────────────────────────────────────────────────────────────

class TestEvalRunner:
    def _runner(self):
        model = tiny_model()
        cfg   = BenchmarkConfig(batch_size=1, seq_len=T, n_warmup=0,
                                 n_trials=2, top_k_values=[1, 5])
        return EvalRunner(model, cfg, name="TestModel")

    def test_run_without_loader(self):
        runner = self._runner()
        result = runner.run(loader=None)
        assert "n_params" in result
        assert "tokens_per_sec" in result

    def test_run_with_loader(self):
        runner = self._runner()
        result = runner.run(tiny_loader(), max_batches=3)
        assert "perplexity" in result
        assert "top_1" in result

    def test_format_report_contains_name(self):
        runner = self._runner()
        result = runner.run()
        report = runner.format_report(result)
        assert "TestModel" in report

    def test_compare_returns_table(self):
        runner  = self._runner()
        result  = runner.run()
        table   = runner.compare([result, result])
        assert "Model" in table
        assert "TestModel" in table
'''
write("tests/test_eval.py", src)
commit("test: add EvalRunner run with/without loader, format_report, compare table tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — bump to v3.1.0 + expose eval in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.0.0\"", "__version__ = \"3.1.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.1.0 — Benchmarking & Evaluation Suite release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `serve` | REST API server + HTTP client |",
    "| `serve` | REST API server + HTTP client |\n"
    "| `eval` | Benchmarking — perplexity, accuracy, throughput, memory |"
)
readme = readme.replace(
    "**Total: 585 commits across 29 days.**",
    "**Total: 647 commits across 31 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.1.0] — 2024 — Benchmarking & Evaluation Suite\n\n### Added\n" \
     "- `EvalRunner` — unified run() + format_report() + compare() interface\n" \
     "- `BenchmarkConfig` — batch_size, seq_len, n_trials, top_k_values\n" \
     "- `compute_perplexity()` — NLL-based PPL with bits-per-char over datasets\n" \
     "- `perplexity_from_logits()` — PPL directly from logits\n" \
     "- `benchmark_prefill()` — tokens/sec timing with warmup + N trials\n" \
     "- `benchmark_memory()` — parameter + buffer memory in MB\n" \
     "- `full_benchmark_report()` — formatted speed + memory string\n" \
     "- `top_k_accuracy()` / `multi_k_accuracy()` — token prediction accuracy\n" \
     "- `evaluate_accuracy()` — full dataset top-K accuracy evaluation\n" \
     "- `examples/benchmark_demo.py` — large vs small model comparison\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.1.0, update README and CHANGELOG for Day 31 Benchmarking")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 31 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.1.0",
    "-m", "NanoMind v3.1.0 — Benchmarking & Evaluation Suite", check=False)
r = run("git", "push", "origin", "v3.1.0", check=False)
print("Tag v3.1.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-15")
print(f"\n=== Last 15 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 31 COMPLETE — v3.1.0 TAGGED! ===")
