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
