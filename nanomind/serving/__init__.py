"""NanoMind Serving sub-package — Continuous Batching & KV Cache management.

Implements production-grade LLM serving infrastructure:
  1. KVBlock / KVCacheManager  — PagedAttention block pool, copy-on-write
  2. InferenceRequest          — request lifecycle, token accumulation
  3. SamplingParams            — temperature, top_p, repetition_penalty
  4. RequestStatus             — WAITING/RUNNING/FINISHED/ABORTED
  5. ContinuousBatchingScheduler — admit, preempt, step(), SchedulerOutput
  6. PrefixCache               — RadixAttention prefix caching, LRU evict
  7. EngineConfig              — serving configuration
  8. LLMEngine                 — full engine: submit, step, run_until_done
  9. EngineStats               — throughput, total_tokens, steps
  10. BenchmarkConfig          — benchmark parameters
  11. BenchmarkResult          — throughput/latency metrics
  12. run_benchmark            — end-to-end serving benchmark

Primary exports:
    - :class:`KVBlock`                       — single KV cache block
    - :class:`KVCacheManager`                — PagedAttention block pool
    - :class:`InferenceRequest`              — request with lifecycle tracking
    - :class:`SamplingParams`                — sampling configuration
    - :class:`RequestStatus`                 — WAITING/RUNNING/FINISHED
    - :class:`ContinuousBatchingScheduler`   — continuous batching scheduler
    - :class:`SchedulerOutput`               — running/prefill/finished
    - :class:`PrefixCache`                   — RadixAttention prefix caching
    - :class:`EngineConfig`                  — engine configuration
    - :class:`LLMEngine`                     — submit, step, run_until_done
    - :class:`EngineStats`                   — throughput, total_tokens
    - :class:`BenchmarkConfig`               — benchmark settings
    - :class:`BenchmarkResult`               — perf metrics
    - :func:`run_benchmark`                  — run a full serving benchmark
    - :func:`make_requests`                  — generate benchmark requests
"""

from nanomind.serving.kv_cache import KVBlock, KVCacheManager
from nanomind.serving.request import InferenceRequest, SamplingParams, RequestStatus
from nanomind.serving.scheduler import ContinuousBatchingScheduler, SchedulerOutput
from nanomind.serving.prefix_cache import PrefixCache, PrefixNode
from nanomind.serving.engine import LLMEngine, EngineConfig, EngineStats
from nanomind.serving.benchmark import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, make_requests,
)

__all__ = [
    "KVBlock", "KVCacheManager",
    "InferenceRequest", "SamplingParams", "RequestStatus",
    "ContinuousBatchingScheduler", "SchedulerOutput",
    "PrefixCache", "PrefixNode",
    "LLMEngine", "EngineConfig", "EngineStats",
    "BenchmarkConfig", "BenchmarkResult", "run_benchmark", "make_requests",
]
