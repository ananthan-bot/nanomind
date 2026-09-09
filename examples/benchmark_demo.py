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
