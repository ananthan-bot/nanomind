"""
nanomind/serve/config.py — Model server configuration.

NanoMind Serve provides a production-ready HTTP inference server built on
Python's built-in ``http.server`` (no external dependencies required).

Architecture:
  Client → HTTP POST /generate → ModelServer
                                      ↓
                              InferenceEngine (model + tokenizer)
                                      ↓
                              GenerateResponse (JSON)

Features:
  - /generate  — text generation with sampling parameters
  - /health    — liveness probe (returns {status: "ok"})
  - /info      — model metadata (vocab_size, d_model, n_params, etc.)
  - /tokenize  — tokenize text without generating
  - Streaming  — Server-Sent Events (SSE) for token-by-token output
  - Batching   — queue-based micro-batching for throughput
  - Validation — request schema validation with clear error messages

This design mirrors real LLM serving systems (vLLM, Ollama, llama.cpp server)
at a pedagogical scale — production systems add async event loops, GPU
scheduling, continuous batching, and PagedAttention on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ServeConfig:
    """
    Configuration for the NanoMind inference server.

    Attributes:
        host:           Server hostname.
        port:           Server port.
        max_batch_size: Maximum concurrent requests in a micro-batch.
        max_new_tokens: Hard cap on generated tokens per request.
        default_temperature: Default sampling temperature.
        default_top_k:  Default top-K filter.
        default_top_p:  Default nucleus sampling threshold.
        timeout_sec:    Request timeout in seconds.
        log_requests:   Log each request/response to stdout.
        model_name:     Human-readable model name in /info response.
    """

    host:                str   = "127.0.0.1"
    port:                int   = 8080
    max_batch_size:      int   = 4
    max_new_tokens:      int   = 256
    default_temperature: float = 1.0
    default_top_k:       int   = 50
    default_top_p:       float = 1.0
    timeout_sec:         float = 30.0
    log_requests:        bool  = True
    model_name:          str   = "NanoMind"

    def __post_init__(self) -> None:
        assert 1 <= self.port <= 65535
        assert self.max_batch_size >= 1
        assert self.max_new_tokens  >= 1
        assert 0.0 <  self.default_temperature
        assert 0.0 <  self.default_top_p <= 1.0
        assert self.timeout_sec > 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
