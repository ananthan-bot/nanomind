"""
nanomind/serve/rate_limit.py — Token bucket rate limiter for the inference server.

Prevents a single client from overwhelming the server with too many requests.
Uses a thread-safe token bucket algorithm:

  - Bucket fills at ``rate`` tokens/second up to ``capacity``
  - Each request costs 1 token
  - If bucket is empty, request is rate-limited (returns False)

This is the same algorithm used by most production API rate limiters
(OpenAI, Anthropic, HuggingFace Inference API).
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Args:
        rate:     Tokens added per second (refill rate).
        capacity: Maximum bucket size (burst capacity).

    Example::

        limiter = TokenBucketRateLimiter(rate=10, capacity=20)
        if limiter.allow():
            process_request()
        else:
            return_429_error()
    """

    def __init__(self, rate: float = 10.0, capacity: float = 20.0) -> None:
        self.rate     = rate
        self.capacity = capacity
        self._tokens  = capacity
        self._last    = time.monotonic()
        self._lock    = threading.Lock()

    def allow(self) -> bool:
        """
        Consume one token. Returns True if allowed, False if rate-limited.
        """
        with self._lock:
            now      = time.monotonic()
            elapsed  = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last   = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def available_tokens(self) -> float:
        with self._lock:
            return self._tokens

    def __repr__(self) -> str:
        return f"TokenBucketRateLimiter(rate={self.rate}/s, capacity={self.capacity})"
