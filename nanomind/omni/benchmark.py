"""
nanomind/omni/benchmark.py — Evaluation suite for duplex latency, cross-modal throughput, and TTFA.
"""
import time
from typing import Dict, Any, List
import torch


class OmniBenchmark:
    """
    Measures Time-to-First-Audio (TTFA) latency, text throughput, and barge-in response speed.
    """

    def __init__(self):
        self.latencies_ms: List[float] = []

    def measure_generation_latency(self, model_forward_fn, iterations: int = 5) -> Dict[str, float]:
        """Runs test iterations and records mean latency in ms."""
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model_forward_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        mean_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)

        return {
            "mean_latency_ms": round(mean_ms, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "real_time_dialogue_ready": mean_ms < 300.0,  # Human conversational threshold
        }
