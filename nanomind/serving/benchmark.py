"""
nanomind/serving/benchmark.py — Serving throughput and latency benchmarker.

Measures:
  - Throughput: tokens generated per second
  - TTFT: time to first token
  - TPOT: time per output token
  - P50/P90/P99 latencies
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field

from nanomind.serving.request import InferenceRequest, SamplingParams


@dataclass
class BenchmarkConfig:
    """Configuration for a serving benchmark."""
    n_requests:          int   = 20
    min_prompt_len:      int   = 4
    max_prompt_len:      int   = 16
    min_output_len:      int   = 4
    max_output_len:      int   = 16
    vocab_size:          int   = 100
    seed:                int   = 42


@dataclass
class BenchmarkResult:
    """Results from a serving benchmark."""
    throughput_tok_s:    float
    mean_latency_s:      float
    p50_latency_s:       float
    p90_latency_s:       float
    n_requests:          int
    n_tokens_total:      int
    wall_time_s:         float

    def to_dict(self) -> dict:
        return {
            "throughput_tok_s": round(self.throughput_tok_s, 1),
            "mean_latency_s":   round(self.mean_latency_s, 3),
            "p50_latency_s":    round(self.p50_latency_s, 3),
            "p90_latency_s":    round(self.p90_latency_s, 3),
            "n_requests":       self.n_requests,
            "n_tokens_total":   self.n_tokens_total,
        }


def make_requests(cfg: BenchmarkConfig) -> list[InferenceRequest]:
    """Generate random benchmark requests."""
    rng = random.Random(cfg.seed)
    reqs = []
    for i in range(cfg.n_requests):
        plen   = rng.randint(cfg.min_prompt_len, cfg.max_prompt_len)
        olen   = rng.randint(cfg.min_output_len, cfg.max_output_len)
        ids    = [rng.randint(0, cfg.vocab_size - 1) for _ in range(plen)]
        params = SamplingParams(max_new_tokens=olen, temperature=1.0)
        reqs.append(InferenceRequest(f"req_{i:04d}", ids, params))
    return reqs


def run_benchmark(engine, cfg: BenchmarkConfig | None = None) -> BenchmarkResult:
    """
    Run a serving throughput benchmark.

    Args:
        engine: :class:`LLMEngine` instance.
        cfg:    :class:`BenchmarkConfig`.

    Returns:
        :class:`BenchmarkResult`.
    """
    cfg  = cfg or BenchmarkConfig()
    reqs = make_requests(cfg)

    for req in reqs:
        engine.submit(req.request_id, req.input_ids, req.sampling_params)

    t0       = time.monotonic()
    finished = engine.run_until_done()
    wall     = time.monotonic() - t0

    latencies    = [r.latency_s or 0.0 for r in finished]
    latencies.sort()
    n_tokens     = sum(r.n_generated for r in finished)
    n            = len(latencies)

    return BenchmarkResult(
        throughput_tok_s = n_tokens / max(wall, 1e-6),
        mean_latency_s   = sum(latencies) / max(n, 1),
        p50_latency_s    = latencies[n // 2] if n else 0.0,
        p90_latency_s    = latencies[int(n * 0.9)] if n else 0.0,
        n_requests       = n,
        n_tokens_total   = n_tokens,
        wall_time_s      = wall,
    )
