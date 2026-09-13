"""
nanomind/streaming/config.py — Streaming inference configuration.

## Why Streaming?

Non-streaming (batch) generation:
  Client sends request → Server runs full generation → Client gets response
  Latency: O(N) tokens * time_per_token (user waits for everything)

Streaming generation:
  Client sends request → Server streams each token as it generates
  Latency: time_to_first_token (user sees output immediately)

This mirrors how ChatGPT, Claude, and Gemini render text progressively.

## Server-Sent Events (SSE)

SSE is the standard protocol for streaming LLM responses over HTTP:
  - One-directional: server → client
  - Plain text format:
      data: {"token": "Hello", "index": 0}\n\n
      data: {"token": " world", "index": 1}\n\n
      data: [DONE]\n\n
  - Native browser support via EventSource API
  - No WebSocket handshake overhead

Used by: OpenAI API, Anthropic Claude API, Google Gemini API.

Reference:
  https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
  https://platform.openai.com/docs/api-reference/streaming
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class StreamConfig:
    """
    Configuration for streaming inference.

    Attributes:
        max_new_tokens:  Maximum tokens to generate per request.
        temperature:     Sampling temperature (0 = greedy, >1 = more random).
        top_k:           Top-K sampling (0 = disabled).
        top_p:           Top-P / nucleus sampling threshold.
        stop_sequences:  List of strings that halt generation when produced.
        stream_interval: Yield every N tokens (1 = every token).
        include_logprobs: Include log-probabilities in stream events.
        timeout_seconds: Maximum seconds for a single generation request.
    """
    max_new_tokens:  int        = 256
    temperature:     float      = 0.8
    top_k:           int        = 40
    top_p:           float      = 0.9
    stop_sequences:  list[str]  = None
    stream_interval: int        = 1
    include_logprobs: bool      = False
    timeout_seconds: float      = 60.0

    def __post_init__(self) -> None:
        if self.stop_sequences is None:
            self.stop_sequences = []
        assert self.max_new_tokens  >= 1
        assert 0.0 <= self.temperature
        assert self.top_k           >= 0
        assert 0.0 <  self.top_p   <= 1.0
        assert self.stream_interval >= 1
