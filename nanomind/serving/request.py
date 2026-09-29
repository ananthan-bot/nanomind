"""
nanomind/serving/request.py — Inference request and sequence state management.

Tracks the lifecycle of a generation request:
  WAITING  → Queued, not yet started
  RUNNING  → Currently being processed in a batch
  FINISHED → Generation complete (max tokens or EOS)
  ABORTED  → Cancelled or errored

Each request owns a sequence with token history, KV cache block pointers,
and sampling parameters.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    WAITING  = "waiting"
    RUNNING  = "running"
    FINISHED = "finished"
    ABORTED  = "aborted"


@dataclass
class SamplingParams:
    """Sampling parameters for a generation request."""
    max_new_tokens: int   = 64
    temperature:    float = 1.0
    top_p:          float = 1.0
    top_k:          int   = 0       # 0 = disabled
    repetition_penalty: float = 1.0
    stop_token_id:  int | None = None

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
            "top_p":          self.top_p,
        }


@dataclass
class InferenceRequest:
    """
    A single generation request tracked through the serving system.

    Args:
        request_id:     Unique identifier.
        input_ids:      Prompt token IDs.
        sampling_params: Sampling configuration.

    Example::

        req = InferenceRequest("req_001", [1, 2, 3, 4],
                                SamplingParams(max_new_tokens=32))
        req.status = RequestStatus.RUNNING
        req.add_token(42)
        print(req.generated_tokens)   # [42]
        print(req.is_finished)        # False (max_new_tokens=32)
    """
    request_id:      str
    input_ids:       list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    status:           RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int]     = field(default_factory=list)
    kv_block_ids:     list[int]     = field(default_factory=list)
    arrival_time:     float         = field(default_factory=time.monotonic)
    start_time:       float | None  = None
    finish_time:      float | None  = None

    def add_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.generated_tokens.append(token_id)
        sp = self.sampling_params
        # Check stop condition
        if len(self.generated_tokens) >= sp.max_new_tokens:
            self._finish()
        elif sp.stop_token_id and token_id == sp.stop_token_id:
            self._finish()

    def _finish(self) -> None:
        self.status      = RequestStatus.FINISHED
        self.finish_time = time.monotonic()

    def abort(self) -> None:
        self.status      = RequestStatus.ABORTED
        self.finish_time = time.monotonic()

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + len(self.generated_tokens)

    @property
    def n_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def latency_s(self) -> float | None:
        if self.finish_time and self.start_time:
            return self.finish_time - self.start_time
        return None

    @property
    def ttft_s(self) -> float | None:
        """Time to first token."""
        if self.start_time and self.generated_tokens:
            return self.start_time - self.arrival_time
        return None

    @property
    def output_ids(self) -> list[int]:
        return self.input_ids + self.generated_tokens

    def to_dict(self) -> dict:
        return {
            "request_id":   self.request_id,
            "status":       self.status.value,
            "prompt_len":   self.prompt_len,
            "n_generated":  self.n_generated,
            "latency_s":    self.latency_s,
        }
