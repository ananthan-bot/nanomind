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
