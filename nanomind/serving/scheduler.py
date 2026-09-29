"""
nanomind/serving/scheduler.py — Continuous batching scheduler.

## Continuous Batching (Orca, Yu et al., 2022)

Traditional batching: pad all sequences to max length, process together.
Problem: short sequences waste GPU cycles waiting for long ones.

Static batching:   |AAAA.....|BBBB.....|CC.......|   (lots of padding!)
Continuous batching: |AAAA|BB|CC|DD|EE|FF|GG|HH|   (no padding, any length!)

In continuous batching:
  - The batch is not fixed per request
  - New requests JOIN the running batch as slots free up
  - Each iteration: run one decode step for all running sequences
  - Finished sequences: remove immediately
  - Waiting sequences: add if KV cache memory permits

This is how vLLM, TensorRT-LLM, and TGI achieve high throughput:
  Traditional: 10-20 req/s
  Continuous:  100-200 req/s (same hardware!)

The key insight: decode step is always 1 token, regardless of history length.
So mixing sequences of different lengths has no waste!

Scheduler decisions:
  1. Prefill phase:  process prompt in one pass (parallel)
  2. Decode phase:   generate one token per step (sequential)

Reference:
  Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based
  Generative Models" OSDI 2022
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.serving.request import InferenceRequest, RequestStatus
from nanomind.serving.kv_cache import KVCacheManager
from nanomind.utils.logger import get_logger

log = get_logger("serving.scheduler")


@dataclass
class SchedulerOutput:
    """Output of one scheduler step."""
    running:    list[InferenceRequest]
    prefill:    list[InferenceRequest]   # newly admitted for prefill
    finished:   list[InferenceRequest]   # just completed
    n_batched_tokens: int = 0


class ContinuousBatchingScheduler:
    """
    Continuous batching scheduler.

    Manages a waiting queue and a running set of sequences.
    Each step: admits new requests, removes finished ones.

    Args:
        kv_manager:    :class:`KVCacheManager`.
        max_batch_size:    Max sequences in flight simultaneously.
        max_tokens_per_step: Max total tokens in one decode step.

    Example::

        scheduler = ContinuousBatchingScheduler(kv_manager, max_batch_size=32)
        scheduler.add_request(req)
        while scheduler.has_work():
            output = scheduler.step()
            # output.running = currently active sequences
            # output.prefill = newly admitted (need prefill)
    """

    def __init__(
        self,
        kv_manager:          KVCacheManager,
        max_batch_size:      int = 32,
        max_tokens_per_step: int = 2048,
    ) -> None:
        self.kv         = kv_manager
        self.max_batch  = max_batch_size
        self.max_tokens = max_tokens_per_step

        self._waiting: list[InferenceRequest] = []
        self._running: list[InferenceRequest] = []
        self._finished: list[InferenceRequest] = []

    def add_request(self, req: InferenceRequest) -> None:
        """Enqueue a new request."""
        self._waiting.append(req)
        log.debug(f"Queued {req.request_id} (prompt={req.prompt_len})")

    def _can_admit(self, req: InferenceRequest) -> bool:
        """Check if KV memory is available for a new request."""
        if len(self._running) >= self.max_batch:
            return False
        return self.kv.can_allocate(req.prompt_len +
                                     req.sampling_params.max_new_tokens)

    def _admit_waiting(self) -> list[InferenceRequest]:
        """Admit as many waiting requests as possible."""
        newly_admitted = []
        still_waiting  = []
        for req in self._waiting:
            if self._can_admit(req):
                import time
                req.status     = RequestStatus.RUNNING
                req.start_time = time.monotonic()
                block_ids = self.kv.allocate(
                    req.request_id,
                    req.prompt_len + req.sampling_params.max_new_tokens,
                )
                req.kv_block_ids = block_ids
                self._running.append(req)
                newly_admitted.append(req)
                log.info(f"Admitted {req.request_id} — blocks={len(block_ids)}")
            else:
                still_waiting.append(req)
        self._waiting = still_waiting
        return newly_admitted

    def step(self) -> SchedulerOutput:
        """
        Execute one scheduling step.

        Returns:
            :class:`SchedulerOutput` with running/prefill/finished lists.
        """
        # 1. Admit new requests (continuous batching!)
        new_prefill = self._admit_waiting()

        # 2. Identify finished sequences
        just_finished = [r for r in self._running if r.is_finished]
        for req in just_finished:
            self.kv.free(req.request_id)
            self._finished.append(req)
        self._running = [r for r in self._running if not r.is_finished]

        # 3. Extend KV blocks for running sequences that need more memory
        for req in self._running:
            if req not in new_prefill:  # already allocated for new prefill
                try:
                    if req.total_len % self.kv.block_size == 0:
                        self.kv.extend(req.request_id)
                except MemoryError:
                    # Preempt: move back to waiting
                    self.kv.free(req.request_id)
                    req.status = RequestStatus.WAITING
                    self._running.remove(req)
                    self._waiting.insert(0, req)   # priority queue front
                    log.warning(f"Preempted {req.request_id} — OOM")

        n_batched = sum(r.total_len for r in self._running)
        return SchedulerOutput(
            running   = list(self._running),
            prefill   = new_prefill,
            finished  = just_finished,
            n_batched_tokens = n_batched,
        )

    def has_work(self) -> bool:
        return bool(self._running or self._waiting)

    def queue_len(self) -> int:
        return len(self._waiting)

    def running_len(self) -> int:
        return len(self._running)

    def stats(self) -> dict:
        return {
            "running": self.running_len(),
            "waiting": self.queue_len(),
            "finished": len(self._finished),
            "kv_stats": self.kv.stats(),
        }
