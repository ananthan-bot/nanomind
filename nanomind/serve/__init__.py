"""NanoMind Serve sub-package — REST API inference server.

Provides a production-ready HTTP inference server with zero external dependencies
(uses Python stdlib http.server + urllib).

Endpoints:
  GET  /health    — liveness probe
  GET  /info      — model metadata
  POST /generate  — text generation with sampling
  POST /tokenize  — tokenize text

Primary exports:
    - :class:`ModelServer`          — HTTP server lifecycle (start/stop/background)
    - :class:`InferenceEngine`      — model + tokenizer wrapper for generation
    - :class:`NanoMindClient`       — stdlib HTTP client to call the server
    - :class:`ServeConfig`          — host, port, max_tokens, temperature defaults
    - :class:`GenerateRequest`      — POST /generate request schema
    - :class:`GenerateResponse`     — POST /generate response schema
    - :class:`TokenBucketRateLimiter` — token bucket rate limiting
"""

from nanomind.serve.config import ServeConfig
from nanomind.serve.engine import InferenceEngine
from nanomind.serve.server import ModelServer
from nanomind.serve.client import NanoMindClient
from nanomind.serve.schemas import (
    GenerateRequest, GenerateResponse,
    TokenizeRequest, TokenizeResponse,
    HealthResponse, InfoResponse, ErrorResponse,
)
from nanomind.serve.rate_limit import TokenBucketRateLimiter

__all__ = [
    "ServeConfig",
    "InferenceEngine",
    "ModelServer",
    "NanoMindClient",
    "GenerateRequest",
    "GenerateResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "HealthResponse",
    "InfoResponse",
    "ErrorResponse",
    "TokenBucketRateLimiter",
]
