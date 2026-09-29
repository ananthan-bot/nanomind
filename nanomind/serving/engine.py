"""
nanomind/serving/engine.py — LLM inference engine with continuous batching.

Ties together: scheduler + KV cache + prefix cache + model.
Runs the continuous batching generation loop.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from nanomind.serving.request import InferenceRequest, SamplingParams, RequestStatus
from nanomind.serving.kv_cache import KVCacheManager
from nanomind.serving.scheduler import ContinuousBatchingScheduler
from nanomind.serving.prefix_cache import PrefixCache
from nanomind.utils.logger import get_logger

log = get_logger("serving.engine")


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""
    max_batch_size:      int   = 32
    max_tokens_per_step: int   = 2048
    n_kv_blocks:         int   = 256
    block_size:          int   = 16
    n_layers:            int   = 4
    n_kv_heads:          int   = 1
    d_head:              int   = 16
    vocab_size:          int   = 100
    enable_prefix_cache: bool  = True


@dataclass
class EngineStats:
    """Aggregate statistics for the serving engine."""
    n_requests_finished: int   = 0
    n_requests_aborted:  int   = 0
    total_tokens_gen:    int   = 0
    total_steps:         int   = 0
    wall_time_s:         float = 0.0

    @property
    def throughput(self) -> float:
        return self.total_tokens_gen / max(self.wall_time_s, 1e-6)

    def to_dict(self) -> dict:
        return {
            "finished_requests":  self.n_requests_finished,
            "total_tokens":       self.total_tokens_gen,
            "throughput_tok_s":   round(self.throughput, 1),
            "total_steps":        self.total_steps,
        }


class LLMEngine:
    """
    LLM serving engine with continuous batching.

    Args:
        model:   LM model ``forward(input_ids) → (B, T, V)`` logits.
        cfg:     :class:`EngineConfig`.

    Example::

        engine = LLMEngine(model, EngineConfig(max_batch_size=8))
        req1   = engine.submit("req_001", [1, 2, 3], SamplingParams(max_new_tokens=16))
        req2   = engine.submit("req_002", [4, 5, 6], SamplingParams(max_new_tokens=8))
        results = engine.run_until_done()
        print(results[0].output_ids)
    """

    def __init__(self, model: nn.Module, cfg: EngineConfig | None = None) -> None:
        self.model  = model
        self.cfg    = cfg or EngineConfig()
        self.kv     = KVCacheManager(
            n_blocks   = self.cfg.n_kv_blocks,
            block_size = self.cfg.block_size,
            n_layers   = self.cfg.n_layers,
            n_kv_heads = self.cfg.n_kv_heads,
            d_head     = self.cfg.d_head,
        )
        self.scheduler = ContinuousBatchingScheduler(
            self.kv,
            max_batch_size      = self.cfg.max_batch_size,
            max_tokens_per_step = self.cfg.max_tokens_per_step,
        )
        self.prefix_cache = PrefixCache() if self.cfg.enable_prefix_cache else None
        self.stats        = EngineStats()

    def submit(
        self,
        request_id:      str,
        input_ids:       list[int],
        sampling_params: SamplingParams | None = None,
    ) -> InferenceRequest:
        """Submit a new generation request."""
        req = InferenceRequest(
            request_id      = request_id,
            input_ids       = input_ids,
            sampling_params = sampling_params or SamplingParams(),
        )
        self.scheduler.add_request(req)
        return req

    @torch.no_grad()
    def _run_model(self, requests: list[InferenceRequest]) -> dict[str, int]:
        """Run model on a batch and return next token for each request."""
        results = {}
        for req in requests:
            # Pack input for this request
            ids    = torch.tensor([req.output_ids[-min(16, req.total_len):]])  # (1, T)
            out    = self.model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            last   = logits[0, -1, :]   # (V,)
            sp     = req.sampling_params

            if sp.temperature <= 0.01:
                next_tok = last.argmax().item()
            else:
                probs    = F.softmax(last / sp.temperature, dim=-1)
                next_tok = torch.multinomial(probs + 1e-10, 1).item()
            results[req.request_id] = next_tok
        return results

    def step(self) -> list[InferenceRequest]:
        """
        Execute one continuous batching step.

        Returns:
            List of just-finished requests.
        """
        sched_out = self.scheduler.step()
        self.stats.total_steps += 1

        if not sched_out.running:
            return sched_out.finished

        # Run model for all running sequences
        next_tokens = self._run_model(sched_out.running)

        # Append tokens to each request
        for req in sched_out.running:
            tok = next_tokens.get(req.request_id, 0)
            req.add_token(tok)
            self.stats.total_tokens_gen += 1

        self.stats.n_requests_finished += len(sched_out.finished)
        return sched_out.finished

    def run_until_done(self, max_steps: int = 1000) -> list[InferenceRequest]:
        """Run until all submitted requests finish."""
        t0     = time.monotonic()
        all_fin = []
        for _ in range(max_steps):
            if not self.scheduler.has_work():
                break
            finished = self.step()
            all_fin.extend(finished)
        self.stats.wall_time_s = time.monotonic() - t0
        return all_fin

    def engine_stats(self) -> dict:
        return {
            **self.stats.to_dict(),
            "scheduler": self.scheduler.stats(),
            "prefix_cache": self.prefix_cache.stats() if self.prefix_cache else {},
        }
