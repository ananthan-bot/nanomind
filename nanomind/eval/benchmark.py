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
