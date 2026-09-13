"""
nanomind/streaming/types.py — Streaming event types for token-by-token generation.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(str, Enum):
    """Type of a streaming generation event."""
    TOKEN     = "token"       # A new token was generated
    FINISH    = "finish"      # Generation complete
    ERROR     = "error"       # An error occurred
    HEARTBEAT = "heartbeat"   # Keep-alive ping


@dataclass
class StreamToken:
    """
    A single generated token in the stream.

    Attributes:
        token:     Decoded token string (e.g. ``" hello"``).
        token_id:  Integer token ID.
        index:     Position in the generated sequence (0-indexed).
        logprob:   Log-probability of this token (optional).
        timestamp: Unix timestamp when this token was generated.
    """
    token:     str
    token_id:  int
    index:     int
    logprob:   float | None = None
    timestamp: float        = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = {
            "token":    self.token,
            "token_id": self.token_id,
            "index":    self.index,
            "ts":       round(self.timestamp, 4),
        }
        if self.logprob is not None:
            d["logprob"] = round(self.logprob, 6)
        return d


@dataclass
class StreamEvent:
    """
    A Server-Sent Event for streaming generation.

    Attributes:
        event_type:    Type of event.
        data:          Event payload (token, finish reason, error message).
        request_id:    ID of the originating request.
    """
    event_type: StreamEventType
    data:       dict
    request_id: str = ""

    def to_sse(self) -> str:
        """
        Serialise to SSE wire format.

        Returns:
            ``data: {json}\n\n`` string ready to send over HTTP.
        """
        payload = {"event": self.event_type.value, **self.data}
        if self.request_id:
            payload["request_id"] = self.request_id
        return f"data: {json.dumps(payload)}

"

    @classmethod
    def token_event(cls, token: StreamToken, request_id: str = "") -> "StreamEvent":
        return cls(StreamEventType.TOKEN, token.to_dict(), request_id)

    @classmethod
    def finish_event(cls, n_tokens: int, finish_reason: str = "stop",
                     request_id: str = "") -> "StreamEvent":
        return cls(StreamEventType.FINISH,
                   {"finish_reason": finish_reason, "n_tokens": n_tokens},
                   request_id)

    @classmethod
    def error_event(cls, message: str, request_id: str = "") -> "StreamEvent":
        return cls(StreamEventType.ERROR, {"message": message}, request_id)

    @classmethod
    def done_sse(cls) -> str:
        """OpenAI-compatible ``data: [DONE]`` terminator."""
        return "data: [DONE]

"
