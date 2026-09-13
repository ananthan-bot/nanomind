"""NanoMind Streaming sub-package — token-by-token streaming inference.

Enables real-time text generation with Server-Sent Events (SSE),
matching the OpenAI/Anthropic/Gemini streaming API format.

Primary exports:
    - :class:`StreamingGenerator` — yields StreamToken one-at-a-time
    - :class:`StreamingServer`    — HTTP server with /stream SSE endpoint
    - :class:`StreamingClient`    — SSE client (urllib, no deps)
    - :class:`StreamConfig`       — max_new_tokens, temperature, stop_sequences
    - :class:`StreamToken`        — token + token_id + index + logprob
    - :class:`StreamEvent`        — SSE event with to_sse() serialiser
    - :class:`StreamEventType`    — TOKEN / FINISH / ERROR / HEARTBEAT
    - :class:`StopSequenceDetector` — multi-token stop sequence detection
    - :class:`TokenBuffer`        — accumulation + callback buffer
    - :func:`print_stream`        — print tokens in real-time to stdout
    - :func:`collect_stream`      — collect all tokens silently
"""

from nanomind.streaming.config import StreamConfig
from nanomind.streaming.types import StreamToken, StreamEvent, StreamEventType
from nanomind.streaming.generator import StreamingGenerator
from nanomind.streaming.server import StreamingServer
from nanomind.streaming.client import StreamingClient
from nanomind.streaming.stop import StopSequenceDetector
from nanomind.streaming.buffer import TokenBuffer, print_stream, collect_stream

__all__ = [
    "StreamConfig",
    "StreamToken", "StreamEvent", "StreamEventType",
    "StreamingGenerator",
    "StreamingServer",
    "StreamingClient",
    "StopSequenceDetector",
    "TokenBuffer", "print_stream", "collect_stream",
]
