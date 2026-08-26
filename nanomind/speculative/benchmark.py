"""
nanomind/speculative/benchmark.py — Speed benchmark: speculative vs autoregressive.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn

from nanomind.speculative.config import SpeculativeConfig
from nanomind.speculative.decode import speculative_decode
from nanomind.generate.strategies import sample_next_token


def benchmark_speculative_vs_autoregressive(
    target_model: nn.Module,
    draft_model:  nn.Module,
    idx:          torch.Tensor,
    n_tokens:     int  = 100,
    n_draft:      int  = 5,
    n_runs:       int  = 3,
    device:       torch.device | None = None,
) -> dict:
    """
    Benchmark speculative decoding against standard autoregressive decoding.

    Args:
        target_model: Large target model.
        draft_model:  Small draft model.
        idx:          Seed token sequence ``(1, T)``.
        n_tokens:     Number of tokens to generate per run.
        n_draft:      Draft tokens per speculative step.
        n_runs:       Number of timed runs to average.
        device:       Device to benchmark on.

    Returns:
        Dict with:
        - ``autoregressive_ms``: ms per token (standard)
        - ``speculative_ms``:    ms per token (speculative)
        - ``speedup``:           ratio (autoregressive / speculative)
        - ``acceptance_rate``:   mean acceptance rate
    """
    device = device or next(target_model.parameters()).device
    block_size = getattr(target_model, "cfg", None) and target_model.cfg.block_size or 512

    # Autoregressive baseline
    ar_times: list[float] = []
    for _ in range(n_runs):
        current = idx.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                ctx     = current[:, -block_size:]
                logits, _ = target_model(ctx)
                tok     = sample_next_token(logits[0, -1, :], strategy="greedy")
                current = torch.cat([current, tok.unsqueeze(0).unsqueeze(0)], dim=1)
        ar_times.append((time.perf_counter() - t0) * 1000 / n_tokens)

    # Speculative decoding
    spec_times: list[float] = []
    acceptance_rates: list[float] = []
    cfg = SpeculativeConfig(n_draft=n_draft, max_new_tokens=n_tokens)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, stats = speculative_decode(target_model, draft_model, idx.clone(), cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        spec_times.append(elapsed_ms / max(stats["n_tokens"], 1))
        acceptance_rates.append(stats["acceptance_rate"])

    ar_ms   = sum(ar_times) / len(ar_times)
    spec_ms = sum(spec_times) / len(spec_times)
    return {
        "autoregressive_ms": ar_ms,
        "speculative_ms":    spec_ms,
        "speedup":           ar_ms / max(spec_ms, 1e-9),
        "acceptance_rate":   sum(acceptance_rates) / len(acceptance_rates),
    }
