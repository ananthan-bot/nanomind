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
        f"Benchmark: {name}
"
        f"{'─'*50}
"
        f"  Parameters : {mem['n_params']:>12,}
"
        f"  Memory     : {mem['total_mb']:>10.2f} MB
"
        f"  Throughput : {speed['tokens_per_sec']:>10,.0f} tokens/sec
"
        f"  Latency    : {speed['ms_per_batch']:>10.2f} ms/batch
"
        f"  Batch      : {cfg.batch_size} × {cfg.seq_len} tokens
"
    )
