"""
nanomind/serve/schemas.py — HTTP request and response schemas for the inference server.

All endpoints accept and return JSON. Schemas are plain dataclasses
(no Pydantic dependency) with manual validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


# ── Request schemas ────────────────────────────────────────────────────────────

@dataclass
class GenerateRequest:
    """
    POST /generate request body.

    Attributes:
        prompt:         Input text to condition generation on.
        max_new_tokens: Maximum tokens to generate.
        temperature:    Sampling temperature (higher = more random).
        top_k:          Keep only top-K tokens (0 = off).
        top_p:          Nucleus sampling threshold (1.0 = off).
        stream:         If True, return tokens via SSE stream.
        stop:           Optional list of stop strings.
        request_id:     Optional client-assigned request ID.
    """
    prompt:         str
    max_new_tokens: int         = 64
    temperature:    float       = 1.0
    top_k:          int         = 50
    top_p:          float       = 1.0
    stream:         bool        = False
    stop:           list[str]   = field(default_factory=list)
    request_id:     str | None  = None

    def validate(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be ≥ 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @classmethod
    def from_dict(cls, d: dict) -> "GenerateRequest":
        return cls(
            prompt=d["prompt"],
            max_new_tokens=d.get("max_new_tokens", 64),
            temperature=d.get("temperature", 1.0),
            top_k=d.get("top_k", 50),
            top_p=d.get("top_p", 1.0),
            stream=d.get("stream", False),
            stop=d.get("stop", []),
            request_id=d.get("request_id"),
        )


@dataclass
class TokenizeRequest:
    """POST /tokenize request body."""
    text:      str
    add_bos:   bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizeRequest":
        return cls(text=d["text"], add_bos=d.get("add_bos", False))


# ── Response schemas ───────────────────────────────────────────────────────────

@dataclass
class GenerateResponse:
    """
    POST /generate response body.

    Attributes:
        text:           Generated text (excluding prompt).
        prompt_tokens:  Number of tokens in the prompt.
        generated_tokens: Number of generated tokens.
        finish_reason:  ``"stop"`` (EOS/stop string) or ``"length"`` (max reached).
        request_id:     Echoed request ID.
        model:          Model name.
    """
    text:              str
    prompt_tokens:     int
    generated_tokens:  int
    finish_reason:     str        = "length"
    request_id:        str | None = None
    model:             str        = "NanoMind"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class HealthResponse:
    """GET /health response."""
    status:  str  = "ok"
    version: str  = "2.5.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class InfoResponse:
    """GET /info response."""
    model:      str
    vocab_size: int
    d_model:    int
    n_layers:   int
    n_heads:    int
    n_params:   int
    block_size: int
    device:     str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class TokenizeResponse:
    """POST /tokenize response."""
    tokens:      list[int]
    n_tokens:    int
    text:        str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ErrorResponse:
    """Error response for invalid requests."""
    error:   str
    code:    int   = 400
    detail:  str   = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))
