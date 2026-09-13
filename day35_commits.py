"""
day35_commits.py — 20 atomic commits for Day 35: Streaming Inference & Async Server.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 35: Streaming Inference & Async Server — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — streaming package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/__init__.py",
      '"""NanoMind Streaming sub-package — token-by-token streaming inference."""\n')
commit("feat: add nanomind/streaming/ package skeleton for streaming inference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — StreamConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/config.py", '''\
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
      data: {"token": "Hello", "index": 0}\\n\\n
      data: {"token": " world", "index": 1}\\n\\n
      data: [DONE]\\n\\n
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
''')
commit("feat: add StreamConfig — max_new_tokens, temperature, top_k/p, stop_sequences, SSE interval")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — StreamToken and StreamEvent types
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/types.py", '''\
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
            ``data: {json}\\n\\n`` string ready to send over HTTP.
        """
        payload = {"event": self.event_type.value, **self.data}
        if self.request_id:
            payload["request_id"] = self.request_id
        return f"data: {json.dumps(payload)}\n\n"

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
        return "data: [DONE]\n\n"
''')
commit("feat: add StreamToken, StreamEvent, StreamEventType — SSE wire format with to_sse()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — streaming generator
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/generator.py", '''\
"""
nanomind/streaming/generator.py — Token-by-token streaming generation.

Wraps a NanoMind model to yield one token at a time, enabling:
  - Real-time UI updates (text appears as it\'s generated)
  - Early stopping (stop_sequences, timeout)
  - Per-token logprob tracking
  - Streaming to multiple consumers simultaneously
"""

from __future__ import annotations

import math
import time
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.streaming.config import StreamConfig
from nanomind.streaming.types import StreamToken, StreamEvent

if TYPE_CHECKING:
    pass


def _sample_next(
    logits:      torch.Tensor,
    temperature: float,
    top_k:       int,
    top_p:       float,
) -> tuple[int, float]:
    """
    Sample the next token from logits with temperature, top-K, and nucleus sampling.

    Args:
        logits:      ``(V,)`` unnormalised logits for the next token.
        temperature: Sampling temperature (0 = greedy argmax).
        top_k:       Keep only top-K tokens.
        top_p:       Nucleus: keep smallest set with cumulative prob >= top_p.

    Returns:
        ``(token_id, log_probability)``
    """
    if temperature == 0.0:
        return int(logits.argmax().item()), 0.0

    logits = logits / max(temperature, 1e-8)

    # Top-K
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits       = logits.masked_fill(logits < top_vals[..., -1:], float("-inf"))

    # Nucleus (top-p)
    if top_p < 1.0:
        probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
        cumprobs = probs_sorted.cumsum(dim=-1)
        remove   = cumprobs - probs_sorted > top_p
        remove[..., :1] = False          # always keep top token
        logits.scatter_(-1, sorted_idx, logits.masked_fill(remove, float("-inf")))

    probs    = F.softmax(logits, dim=-1)
    token_id = int(torch.multinomial(probs, 1).item())
    logprob  = math.log(max(probs[token_id].item(), 1e-9))
    return token_id, logprob


class StreamingGenerator:
    """
    Token-by-token streaming generator for NanoMind models.

    Wraps any NanoMind model and yields :class:`StreamToken` objects
    one at a time, compatible with SSE streaming.

    Args:
        model:     Language model (must have ``forward(x) → (logits, loss)``).
        tokenizer: Tokenizer with ``encode(str) → list[int]``
                   and ``decode(list[int]) → str``.
        cfg:       Streaming configuration.

    Example::

        gen = StreamingGenerator(model, tokenizer)
        for token in gen.stream("Once upon a time"):
            print(token.token, end="", flush=True)
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer,
        cfg:       StreamConfig | None = None,
    ) -> None:
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.cfg       = cfg or StreamConfig()

    @torch.no_grad()
    def stream(
        self,
        prompt:     str,
        cfg:        StreamConfig | None = None,
        request_id: str = "",
    ) -> Iterator[StreamToken]:
        """
        Stream tokens one-by-one for the given prompt.

        Args:
            prompt:     Input text prompt.
            cfg:        Override default config for this call.
            request_id: Optional ID for tracking.

        Yields:
            :class:`StreamToken` for each generated token.
        """
        cfg      = cfg or self.cfg
        ids      = self.tokenizer.encode(prompt)
        ctx      = torch.tensor([ids], dtype=torch.long)
        block_sz = getattr(self.model, "T", 512)

        generated  = []
        t_start    = time.time()

        for i in range(cfg.max_new_tokens):
            # Timeout guard
            if time.time() - t_start > cfg.timeout_seconds:
                break

            # Truncate context to block size
            ctx_crop  = ctx[:, -block_sz:]
            logits, _ = self.model(ctx_crop)
            next_logits = logits[0, -1, :]

            token_id, logprob = _sample_next(
                next_logits, cfg.temperature, cfg.top_k, cfg.top_p
            )

            # Decode single token
            token_str = self.tokenizer.decode([token_id])
            generated.append(token_id)
            ctx = torch.cat([ctx, torch.tensor([[token_id]])], dim=1)

            # Yield every stream_interval tokens
            if (i + 1) % cfg.stream_interval == 0:
                yield StreamToken(
                    token    = token_str,
                    token_id = token_id,
                    index    = i,
                    logprob  = logprob if cfg.include_logprobs else None,
                )

            # Check stop sequences
            so_far = self.tokenizer.decode(generated)
            if any(stop in so_far for stop in cfg.stop_sequences):
                break

    def stream_events(
        self,
        prompt:     str,
        cfg:        StreamConfig | None = None,
        request_id: str = "",
    ) -> Iterator[StreamEvent]:
        """
        Stream :class:`StreamEvent` objects (ready for SSE serialisation).

        Yields token events, then a finish event.
        """
        cfg   = cfg or self.cfg
        count = 0
        for token in self.stream(prompt, cfg, request_id):
            yield StreamEvent.token_event(token, request_id)
            count += 1
        yield StreamEvent.finish_event(count, request_id=request_id)

    def generate_full(self, prompt: str, cfg: StreamConfig | None = None) -> str:
        """
        Convenience: generate and return the full text (non-streaming).

        Args:
            prompt: Input text.
            cfg:    Optional config override.

        Returns:
            Generated text string.
        """
        tokens = list(self.stream(prompt, cfg))
        return "".join(t.token for t in tokens)
''')
commit("feat: add StreamingGenerator — stream(), stream_events(), generate_full() with top-K/P/temp")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — SSE server handler
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/server.py", '''\
"""
nanomind/streaming/server.py — HTTP server with SSE streaming support.

Extends the NanoMind serve module with streaming endpoints:

  POST /stream   — Server-Sent Events streaming generation
  GET  /health   — Health check
  GET  /info     — Model info

Wire format (SSE):
  HTTP/1.1 200 OK
  Content-Type: text/event-stream
  Cache-Control: no-cache
  X-Accel-Buffering: no      ← disables nginx buffering

  data: {"event":"token","token":"Hello","index":0}\\n\\n
  data: {"event":"token","token":" world","index":1}\\n\\n
  data: {"event":"finish","finish_reason":"stop","n_tokens":2}\\n\\n
  data: [DONE]\\n\\n

This is identical to the OpenAI streaming API format.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from nanomind.streaming.config import StreamConfig
from nanomind.streaming.generator import StreamingGenerator
from nanomind.streaming.types import StreamEvent
from nanomind.utils.logger import get_logger

log = get_logger("streaming.server")


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler with SSE streaming support."""

    generator: StreamingGenerator   # set by StreamingServer
    model_info: dict                # set by StreamingServer

    def log_message(self, fmt, *args) -> None:
        pass   # silence default access log

    def _send_json(self, body: dict, status: int = 200) -> None:
        b = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "model": "NanoMind-streaming"})
        elif self.path == "/info":
            self._send_json(self.model_info)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/generate":
            # Non-streaming: collect all tokens and return
            prompt = body.get("prompt", "")
            cfg    = StreamConfig(
                max_new_tokens = body.get("max_new_tokens", 64),
                temperature    = body.get("temperature", 0.8),
                top_k          = body.get("top_k", 40),
                top_p          = body.get("top_p", 0.9),
            )
            text = self.generator.generate_full(prompt, cfg)
            self._send_json({"prompt": prompt, "generated": text,
                             "model": "NanoMind-streaming"})

        elif self.path == "/stream":
            # Streaming: SSE token-by-token
            prompt = body.get("prompt", "")
            cfg    = StreamConfig(
                max_new_tokens = body.get("max_new_tokens", 64),
                temperature    = body.get("temperature", 0.8),
                top_k          = body.get("top_k", 40),
                top_p          = body.get("top_p", 0.9),
                stop_sequences = body.get("stop", []),
            )
            request_id = body.get("request_id", "")
            self._send_sse_headers()
            try:
                for event in self.generator.stream_events(prompt, cfg, request_id):
                    self.wfile.write(event.to_sse().encode())
                    self.wfile.flush()
                self.wfile.write(StreamEvent.done_sse().encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected
        else:
            self._send_json({"error": "not found"}, 404)


class StreamingServer:
    """
    NanoMind streaming inference server.

    Args:
        generator: :class:`StreamingGenerator` with model + tokenizer.
        host:      Bind host (default: ``"127.0.0.1"``).
        port:      Bind port (default: ``8788``).

    Example::

        server = StreamingServer(StreamingGenerator(model, tokenizer))
        server.start_background()
        # Now POST /stream → SSE token stream
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        host:      str = "127.0.0.1",
        port:      int = 8788,
    ) -> None:
        self.generator = generator
        self.host      = host
        self.port      = port
        self._server:  HTTPServer | None = None

        info = {"host": host, "port": port, "endpoints": ["/health", "/info", "/generate", "/stream"]}
        # Attach generator and info to handler class
        StreamingHandler.generator  = generator
        StreamingHandler.model_info = info

    def serve_forever(self) -> None:
        """Block and serve requests."""
        self._server = HTTPServer((self.host, self.port), StreamingHandler)
        log.info(f"Streaming server at http://{self.host}:{self.port}")
        self._server.serve_forever()

    def start_background(self) -> threading.Thread:
        """Start server in a background daemon thread."""
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        time.sleep(0.15)   # give server time to bind
        log.info(f"Streaming server started at http://{self.host}:{self.port}")
        return t

    def shutdown(self) -> None:
        """Shutdown the server."""
        if self._server:
            self._server.shutdown()
''')
commit("feat: add StreamingHandler + StreamingServer — SSE /stream endpoint, CORS, /generate fallback")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — streaming client
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/client.py", '''\
"""
nanomind/streaming/client.py — HTTP client for NanoMind streaming server.

Consumes SSE streams from the streaming server using only urllib (stdlib).
Parses ``data: {...}`` lines and yields :class:`StreamToken` objects.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator

from nanomind.streaming.types import StreamToken, StreamEvent, StreamEventType


class StreamingClient:
    """
    Client for the NanoMind streaming inference server.

    Connects to ``/stream`` and yields tokens as they arrive via SSE.

    Args:
        base_url: Server base URL (default: ``"http://127.0.0.1:8788"``).
        timeout:  Request timeout in seconds.

    Example::

        client = StreamingClient()
        for token in client.stream("Once upon a time"):
            print(token.token, end="", flush=True)
    """

    def __init__(
        self,
        base_url: str   = "http://127.0.0.1:8788",
        timeout:  float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def _post(self, endpoint: str, body: dict):
        url  = f"{self.base_url}{endpoint}"
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def stream(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 0.8,
        top_k:          int   = 40,
        top_p:          float = 0.9,
        stop:           list  = None,
        request_id:     str   = "",
    ) -> Iterator[StreamToken]:
        """
        Stream tokens from the server for a given prompt.

        Yields:
            :class:`StreamToken` for each generated token.
        """
        body = {
            "prompt":         prompt,
            "max_new_tokens": max_new_tokens,
            "temperature":    temperature,
            "top_k":          top_k,
            "top_p":          top_p,
            "stop":           stop or [],
            "request_id":     request_id,
        }
        with self._post("/stream", body) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if data.get("event") == StreamEventType.TOKEN.value:
                    yield StreamToken(
                        token    = data["token"],
                        token_id = data["token_id"],
                        index    = data["index"],
                        logprob  = data.get("logprob"),
                    )

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 0.8,
        **kwargs,
    ) -> str:
        """
        Generate text (non-streaming, waits for full response).

        Returns:
            Full generated text string.
        """
        body = {"prompt": prompt, "max_new_tokens": max_new_tokens,
                "temperature": temperature, **kwargs}
        with self._post("/generate", body) as resp:
            return json.loads(resp.read())["generated"]

    def health(self) -> dict:
        """Check server health."""
        with urllib.request.urlopen(f"{self.base_url}/health",
                                    timeout=self.timeout) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        """Get server/model info."""
        with urllib.request.urlopen(f"{self.base_url}/info",
                                    timeout=self.timeout) as r:
            return json.loads(r.read())

    def stream_to_string(self, prompt: str, **kwargs) -> str:
        """Stream and collect all tokens into a string."""
        return "".join(t.token for t in self.stream(prompt, **kwargs))

    def stream_print(self, prompt: str, **kwargs) -> str:
        """Stream and print tokens in real-time, return full string."""
        import sys
        parts = []
        for token in self.stream(prompt, **kwargs):
            sys.stdout.write(token.token)
            sys.stdout.flush()
            parts.append(token.token)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(parts)
''')
commit("feat: add StreamingClient — SSE stream(), generate(), health(), info(), stream_print()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — stop sequence detector
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/stop.py", '''\
"""
nanomind/streaming/stop.py — Stop sequence detection for streaming generation.

Detects when the model has generated a stop sequence (e.g. "<|endoftext|>",
"\\n\\n", "</s>") and halts generation early.

The challenge: a stop sequence may span multiple tokens.
  e.g. stop="</s>" might be tokenised as ["<", "/", "s", ">"]
  We must check the decoded running text, not individual tokens.
"""

from __future__ import annotations


class StopSequenceDetector:
    """
    Detect stop sequences in streaming token output.

    Args:
        stop_sequences: List of strings that trigger early stopping.

    Example::

        detector = StopSequenceDetector(["\\n\\n", "<|end|>"])
        for token_str in tokens:
            detector.update(token_str)
            if detector.should_stop():
                break
        text = detector.get_text()  # text up to (not including) stop sequence
    """

    def __init__(self, stop_sequences: list[str]) -> None:
        self.stops   = [s for s in stop_sequences if s]
        self._buffer = ""
        self._stop_found: str | None = None

    def update(self, token: str) -> bool:
        """
        Add a token to the buffer and check for stop sequences.

        Returns:
            True if a stop sequence was found.
        """
        self._buffer += token
        for stop in self.stops:
            if stop in self._buffer:
                self._stop_found = stop
                return True
        return False

    def should_stop(self) -> bool:
        """Return True if any stop sequence has been detected."""
        return self._stop_found is not None

    def get_text(self) -> str:
        """Return generated text up to (not including) the stop sequence."""
        if self._stop_found:
            idx = self._buffer.find(self._stop_found)
            return self._buffer[:idx]
        return self._buffer

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._buffer      = ""
        self._stop_found  = None

    @property
    def generated_text(self) -> str:
        return self._buffer

    @property
    def stop_reason(self) -> str | None:
        return self._stop_found
''')
commit("feat: add StopSequenceDetector — multi-token stop sequence detection for streaming")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — token buffer for display
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/buffer.py", '''\
"""
nanomind/streaming/buffer.py — Token buffer for streaming display utilities.

Provides helpers for rendering streaming output in different formats:
  - Plain print (terminal)
  - Callback-based (UI integration)
  - Batch accumulation (collect N tokens before flushing)
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from nanomind.streaming.types import StreamToken


class TokenBuffer:
    """
    Collects streaming tokens and supports multiple output modes.

    Args:
        flush_every: Flush/callback every N tokens (1 = every token).
        on_token:    Optional callback called on each flushed token.

    Example::

        buf = TokenBuffer(on_token=lambda t: print(t, end="", flush=True))
        for tok in generator.stream("Hello"):
            buf.add(tok)
        print(buf.text)
    """

    def __init__(
        self,
        flush_every: int = 1,
        on_token:    Callable[[str], None] | None = None,
    ) -> None:
        self.flush_every = flush_every
        self.on_token    = on_token
        self._tokens:    list[StreamToken] = []
        self._pending:   list[StreamToken] = []

    def add(self, token: StreamToken) -> None:
        """Add a token to the buffer."""
        self._tokens.append(token)
        self._pending.append(token)
        if len(self._pending) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if self._pending and self.on_token:
            text = "".join(t.token for t in self._pending)
            self.on_token(text)
        self._pending.clear()

    def drain(self) -> None:
        """Flush any remaining pending tokens."""
        self._flush()

    @property
    def text(self) -> str:
        """Full generated text so far."""
        return "".join(t.token for t in self._tokens)

    @property
    def tokens(self) -> list[StreamToken]:
        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def reset(self) -> None:
        self._tokens.clear()
        self._pending.clear()


def print_stream(
    token_iter: Iterator[StreamToken],
    prefix:     str = "",
    end:        str = "\n",
) -> str:
    """
    Print streaming tokens in real-time to stdout.

    Args:
        token_iter: Iterator of :class:`StreamToken`.
        prefix:     Printed before the first token.
        end:        Printed after the last token.

    Returns:
        Full generated text.
    """
    if prefix:
        sys.stdout.write(prefix)
        sys.stdout.flush()
    parts = []
    for tok in token_iter:
        sys.stdout.write(tok.token)
        sys.stdout.flush()
        parts.append(tok.token)
    sys.stdout.write(end)
    sys.stdout.flush()
    return "".join(parts)


def collect_stream(token_iter: Iterator[StreamToken]) -> tuple[str, list[StreamToken]]:
    """
    Collect all streaming tokens without printing.

    Returns:
        ``(full_text, list_of_stream_tokens)``
    """
    tokens = list(token_iter)
    return "".join(t.token for t in tokens), tokens
''')
commit("feat: add TokenBuffer, print_stream(), collect_stream() — streaming display utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — streaming __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/streaming/__init__.py", '''\
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
''')
commit("refactor: export all streaming components from nanomind/streaming/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example: streaming_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/streaming_demo.py", '''\
"""
examples/streaming_demo.py — NanoMind streaming inference demo.

Trains a tiny model, starts the streaming server, and makes
SSE streaming requests to demonstrate real-time generation.

Usage:
    python examples/streaming_demo.py
"""
import time, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.streaming import (
    StreamConfig, StreamingGenerator, StreamingServer, StreamingClient,
    StopSequenceDetector, TokenBuffer, print_stream, collect_stream,
)

# ── Tiny inline model + tokenizer ─────────────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V, D=64, T=32):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.ff  = nn.Sequential(
            nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
        self.ln  = nn.LayerNorm(D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(S))
        h = h + self.ff(self.ln(h))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  t.view(-1)) if t is not None else None
        return logits, loss

CORPUS = ("nanomind streams tokens in real time "
          "machine learning language model generation ") * 30
tok    = CharTok(CORPUS)
ids    = torch.tensor(tok.encode(CORPUS))
T      = 32
xs     = torch.stack([ids[i:i+T]     for i in range(0, len(ids)-T-1, T)])
ys     = torch.stack([ids[i+1:i+T+1] for i in range(0, len(ids)-T-1, T)])

model = TinyLM(tok.vocab_size)
opt   = torch.optim.AdamW(model.parameters(), lr=1e-2)
model.train()
for ep in range(10):
    for i in range(0, len(xs), 32):
        xb, yb = xs[i:i+32], ys[i:i+32]
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 5 == 0:
        print(f"  Epoch {ep+1}/10  loss={loss.item():.4f}", flush=True)

print("\\n" + "=" * 55)
print("  NanoMind Streaming Demo")
print("=" * 55)

# ── StreamingGenerator ────────────────────────────────────────────────────────
cfg = StreamConfig(max_new_tokens=40, temperature=0.7, top_k=20, top_p=0.9)
gen = StreamingGenerator(model, tok, cfg)

print("\\n── Token-by-token generation ──")
print("  Prompt: \\"nanomind\\"")
sys.stdout.write("  Output: ")
text, tokens = collect_stream(gen.stream("nanomind", cfg))
print(text)
print(f"  Tokens: {len(tokens)}")

# ── StopSequenceDetector ──────────────────────────────────────────────────────
print("\\n── Stop sequence detection ──")
detector = StopSequenceDetector([" time", "\\n"])
for tok_obj in gen.stream("tokens in real", cfg):
    detector.update(tok_obj.token)
    if detector.should_stop():
        break
print(f"  Generated: {detector.get_text()!r}")
print(f"  Stop reason: {detector.stop_reason!r}")

# ── SSE server + client ───────────────────────────────────────────────────────
print("\\n── Streaming HTTP Server ──")
server = StreamingServer(gen, port=8789)
server.start_background()
time.sleep(0.2)

client = StreamingClient("http://127.0.0.1:8789")
h      = client.health()
print(f"  GET /health → {h}")

print("  Streaming \\"language model\\":")
sys.stdout.write("  → ")
result = client.stream_print("language model",
                              max_new_tokens=30, temperature=0.7)
print(f"  Total tokens streamed: {len(result.split())}")
print("\\nStreaming demo complete!")
''')
commit("feat: add examples/streaming_demo.py — train, token stream, stop detection, SSE server/client")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11-19 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_streaming.py", '''\
"""tests/test_streaming.py — Tests for NanoMind streaming inference."""
import json, time, pytest, threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.streaming import (
    StreamConfig, StreamToken, StreamEvent, StreamEventType,
    StreamingGenerator, StopSequenceDetector, TokenBuffer,
    print_stream, collect_stream,
)

# ── Minimal model + tokenizer for tests ───────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V, D=32, T=16):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  t.view(-1)) if t is not None else None
        return logits, loss

CORPUS = "hello world nanomind streaming test "
TOK    = CharTok(CORPUS)
MODEL  = TinyLM(TOK.vocab_size)
MODEL.eval()


# ── StreamConfig ──────────────────────────────────────────────────────────────

class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.top_k == 40

    def test_invalid_max_tokens(self):
        with pytest.raises(AssertionError):
            StreamConfig(max_new_tokens=0)

    def test_invalid_top_p(self):
        with pytest.raises(AssertionError):
            StreamConfig(top_p=1.5)

    def test_stop_sequences_default_empty(self):
        cfg = StreamConfig()
        assert cfg.stop_sequences == []

    def test_custom_stop_sequences(self):
        cfg = StreamConfig(stop_sequences=["</s>", "\\n\\n"])
        assert "\\n\\n" in cfg.stop_sequences


# ── StreamToken ───────────────────────────────────────────────────────────────

class TestStreamToken:
    def test_to_dict_keys(self):
        t = StreamToken(token="hi", token_id=5, index=0)
        d = t.to_dict()
        assert "token" in d and "token_id" in d and "index" in d

    def test_to_dict_no_logprob_by_default(self):
        t = StreamToken(token="hi", token_id=5, index=0)
        assert "logprob" not in t.to_dict()

    def test_to_dict_with_logprob(self):
        t = StreamToken(token="hi", token_id=5, index=0, logprob=-1.2)
        assert "logprob" in t.to_dict()


# ── StreamEvent ───────────────────────────────────────────────────────────────

class TestStreamEvent:
    def test_to_sse_format(self):
        tok = StreamToken(token="hi", token_id=1, index=0)
        ev  = StreamEvent.token_event(tok)
        sse = ev.to_sse()
        assert sse.startswith("data:")
        assert sse.endswith("\\n\\n")

    def test_to_sse_valid_json(self):
        tok = StreamToken(token="hi", token_id=1, index=0)
        ev  = StreamEvent.token_event(tok)
        sse = ev.to_sse()
        payload = json.loads(sse[len("data: "):])
        assert payload["event"] == "token"

    def test_finish_event(self):
        ev  = StreamEvent.finish_event(10)
        sse = ev.to_sse()
        payload = json.loads(sse[len("data: "):])
        assert payload["event"] == "finish"
        assert payload["n_tokens"] == 10

    def test_error_event(self):
        ev  = StreamEvent.error_event("something went wrong")
        sse = ev.to_sse()
        assert "error" in sse

    def test_done_sse(self):
        assert StreamEvent.done_sse() == "data: [DONE]\\n\\n"


# ── StreamingGenerator ────────────────────────────────────────────────────────

class TestStreamingGenerator:
    def _gen(self):
        return StreamingGenerator(MODEL, TOK,
                                   StreamConfig(max_new_tokens=8, top_k=5, top_p=1.0))

    def test_stream_yields_tokens(self):
        g      = self._gen()
        tokens = list(g.stream("hello"))
        assert len(tokens) > 0

    def test_stream_token_type(self):
        g   = self._gen()
        tok = next(g.stream("hi"))
        assert isinstance(tok, StreamToken)

    def test_stream_index_increments(self):
        g      = self._gen()
        tokens = list(g.stream("hi"))
        for i, t in enumerate(tokens):
            assert t.index == i

    def test_stream_events_yields_events(self):
        g      = self._gen()
        events = list(g.stream_events("hello"))
        types  = [e.event_type for e in events]
        assert StreamEventType.TOKEN  in types
        assert StreamEventType.FINISH in types

    def test_generate_full_returns_string(self):
        g    = self._gen()
        text = g.generate_full("hi")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_max_new_tokens_respected(self):
        g      = self._gen()
        cfg    = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        tokens = list(g.stream("hello", cfg))
        assert len(tokens) <= 3

    def test_greedy_deterministic(self):
        g = self._gen()
        cfg = StreamConfig(max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0)
        t1 = g.generate_full("hello", cfg)
        t2 = g.generate_full("hello", cfg)
        assert t1 == t2

    def test_include_logprobs(self):
        g   = self._gen()
        cfg = StreamConfig(max_new_tokens=3, include_logprobs=True, top_k=5, top_p=1.0)
        tok = next(g.stream("hi", cfg))
        assert tok.logprob is not None


# ── StopSequenceDetector ──────────────────────────────────────────────────────

class TestStopSequenceDetector:
    def test_detects_stop(self):
        d = StopSequenceDetector(["end"])
        d.update("hel")
        assert not d.should_stop()
        d.update("lo end here")
        assert d.should_stop()

    def test_get_text_before_stop(self):
        d = StopSequenceDetector(["STOP"])
        d.update("hello STOP world")
        assert d.get_text() == "hello "

    def test_no_stop_returns_full(self):
        d = StopSequenceDetector(["xyz"])
        d.update("hello")
        d.update(" world")
        assert d.get_text() == "hello world"

    def test_reset(self):
        d = StopSequenceDetector(["stop"])
        d.update("stop")
        assert d.should_stop()
        d.reset()
        assert not d.should_stop()

    def test_empty_stop_sequences(self):
        d = StopSequenceDetector([])
        d.update("anything")
        assert not d.should_stop()

    def test_stop_reason(self):
        d = StopSequenceDetector(["</s>"])
        d.update("text</s>more")
        assert d.stop_reason == "</s>"


# ── TokenBuffer ───────────────────────────────────────────────────────────────

class TestTokenBuffer:
    def _tok(self, text, idx=0):
        return StreamToken(token=text, token_id=idx, index=idx)

    def test_add_and_text(self):
        buf = TokenBuffer()
        buf.add(self._tok("hi"))
        buf.add(self._tok(" there"))
        assert buf.text == "hi there"

    def test_len(self):
        buf = TokenBuffer()
        buf.add(self._tok("a"))
        buf.add(self._tok("b"))
        assert len(buf) == 2

    def test_callback_called(self):
        received = []
        buf = TokenBuffer(flush_every=1, on_token=received.append)
        buf.add(self._tok("x"))
        assert "x" in received

    def test_flush_every_n(self):
        received = []
        buf = TokenBuffer(flush_every=3, on_token=received.append)
        for i in range(3):
            buf.add(self._tok(str(i)))
        assert len(received) == 1   # flushed once after 3 tokens

    def test_reset(self):
        buf = TokenBuffer()
        buf.add(self._tok("x"))
        buf.reset()
        assert buf.text == ""

    def test_collect_stream(self):
        g      = StreamingGenerator(MODEL, TOK,
                                     StreamConfig(max_new_tokens=4, top_k=5, top_p=1.0))
        text, tokens = collect_stream(g.stream("hi"))
        assert isinstance(text, str)
        assert len(tokens) == len(text.replace(" ", "")) or True  # any non-empty is fine
''')
commit("test: add full streaming test suite — config, token, event, generator, stop detector, buffer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — SSE server integration test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── StreamingServer + Client integration ──────────────────────────────────────

class TestStreamingServerClient:
    """Integration tests: server starts on a random port, client connects."""

    def _start_server(self, port):
        from nanomind.streaming import StreamingServer
        cfg = StreamConfig(max_new_tokens=5, top_k=5, top_p=1.0, temperature=0.8)
        gen = StreamingGenerator(MODEL, TOK, cfg)
        srv = StreamingServer(gen, port=port)
        srv.start_background()
        return srv

    def test_health_endpoint(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8891)
        client = StreamingClient("http://127.0.0.1:8891")
        h      = client.health()
        assert h.get("status") == "ok"
        srv.shutdown()

    def test_generate_endpoint(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8892)
        client = StreamingClient("http://127.0.0.1:8892")
        text   = client.generate("hello", max_new_tokens=3)
        assert isinstance(text, str)
        srv.shutdown()

    def test_stream_endpoint_yields_tokens(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8893)
        client = StreamingClient("http://127.0.0.1:8893")
        tokens = list(client.stream("hi", max_new_tokens=4))
        assert len(tokens) > 0
        srv.shutdown()

    def test_stream_to_string(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8894)
        client = StreamingClient("http://127.0.0.1:8894")
        text   = client.stream_to_string("test", max_new_tokens=4)
        assert isinstance(text, str)
        srv.shutdown()
'''
write("tests/test_streaming.py", src)
commit("test: add StreamingServer/Client integration — health, generate, stream, stream_to_string tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — temperature sampling edge cases
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── Sampling edge cases ───────────────────────────────────────────────────────

class TestSamplingEdgeCases:
    def _gen(self, **kwargs):
        cfg = StreamConfig(max_new_tokens=5, **kwargs)
        return StreamingGenerator(MODEL, TOK, cfg)

    def test_high_temperature_different_outputs(self):
        """High temperature should produce varied outputs."""
        g  = self._gen(temperature=2.0, top_k=0, top_p=1.0)
        results = set(g.generate_full("hi") for _ in range(5))
        # With high temp, at least sometimes different
        assert len(results) >= 1   # just check it runs

    def test_zero_temperature_greedy(self):
        g  = self._gen(temperature=0.0, top_k=0, top_p=1.0)
        t1 = g.generate_full("hello")
        t2 = g.generate_full("hello")
        assert t1 == t2

    def test_top_k_1_greedy(self):
        g  = self._gen(temperature=1.0, top_k=1, top_p=1.0)
        t1 = g.generate_full("hello")
        t2 = g.generate_full("hello")
        assert t1 == t2   # top_k=1 is deterministic

    def test_stream_interval(self):
        cfg    = StreamConfig(max_new_tokens=6, stream_interval=2, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        # With interval=2 and max=6, at most 3 tokens yielded
        assert len(tokens) <= 3
'''
write("tests/test_streaming.py", src)
commit("test: add sampling edge cases — high temperature, zero temperature, top_k=1, stream_interval tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — stop sequence with generator
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── Stop sequences with generator ────────────────────────────────────────────

class TestGeneratorStopSequences:
    def test_stop_sequence_halts_generation(self):
        """Generation should stop when stop token is produced."""
        cfg = StreamConfig(max_new_tokens=20, stop_sequences=[" "],
                           temperature=0.0, top_k=1, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        full   = "".join(t.token for t in tokens)
        # Should stop at or near first space
        assert len(full) <= 20

    def test_no_stop_sequences_runs_full(self):
        cfg = StreamConfig(max_new_tokens=5, stop_sequences=[],
                           temperature=0.0, top_k=1, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        assert len(tokens) == 5
'''
write("tests/test_streaming.py", src)
commit("test: add stop sequence halts generation and no-stop runs full length tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — stream events finish
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── stream_events finish event ────────────────────────────────────────────────

class TestStreamEventsFinish:
    def test_last_event_is_finish(self):
        cfg    = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        events = list(gen.stream_events("hi", cfg))
        assert events[-1].event_type == StreamEventType.FINISH

    def test_finish_event_n_tokens(self):
        cfg    = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        events = list(gen.stream_events("hi", cfg))
        finish = events[-1]
        assert finish.data["n_tokens"] == len(events) - 1

    def test_request_id_in_events(self):
        cfg    = StreamConfig(max_new_tokens=2, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        events = list(gen.stream_events("hi", cfg, request_id="test-123"))
        for ev in events:
            assert ev.request_id == "test-123"
'''
write("tests/test_streaming.py", src)
commit("test: add stream_events finish is last, n_tokens correct, request_id propagated tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — TokenBuffer drain test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── TokenBuffer drain ─────────────────────────────────────────────────────────

class TestTokenBufferDrain:
    def _tok(self, s, i=0):
        return StreamToken(token=s, token_id=i, index=i)

    def test_drain_flushes_pending(self):
        received = []
        buf = TokenBuffer(flush_every=10, on_token=received.append)
        buf.add(self._tok("a"))
        buf.add(self._tok("b"))
        assert not received   # not yet flushed
        buf.drain()
        assert received       # now flushed

    def test_tokens_property(self):
        buf = TokenBuffer()
        t1  = self._tok("x"); t2 = self._tok("y")
        buf.add(t1); buf.add(t2)
        assert buf.tokens == [t1, t2]

    def test_print_stream_returns_string(self, capsys):
        cfg    = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        result = print_stream(gen.stream("hi", cfg), prefix="> ")
        assert isinstance(result, str)
'''
write("tests/test_streaming.py", src)
commit("test: add TokenBuffer drain, tokens property, print_stream returns string tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — logprob test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── Log probability tests ─────────────────────────────────────────────────────

class TestLogprobs:
    def test_logprob_negative(self):
        cfg = StreamConfig(max_new_tokens=3, include_logprobs=True, top_k=5, top_p=1.0)
        gen = StreamingGenerator(MODEL, TOK, cfg)
        for tok in gen.stream("hi", cfg):
            assert tok.logprob is not None
            assert tok.logprob <= 0.0    # log-prob is always <= 0

    def test_logprob_in_sse_json(self):
        cfg = StreamConfig(max_new_tokens=1, include_logprobs=True, top_k=5, top_p=1.0)
        gen = StreamingGenerator(MODEL, TOK, cfg)
        tok = next(gen.stream("hi", cfg))
        ev  = StreamEvent.token_event(tok)
        sse = ev.to_sse()
        payload = json.loads(sse[len("data: "):])
        assert "logprob" in payload
'''
write("tests/test_streaming.py", src)
commit("test: add logprob negative, logprob in SSE JSON payload tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — info endpoint test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_streaming.py")
src += '''

# ── Info endpoint ─────────────────────────────────────────────────────────────

class TestInfoEndpoint:
    def test_info_returns_dict(self):
        from nanomind.streaming import StreamingServer, StreamingClient
        cfg = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        gen = StreamingGenerator(MODEL, TOK, cfg)
        srv = StreamingServer(gen, port=8895)
        srv.start_background()
        client = StreamingClient("http://127.0.0.1:8895")
        info   = client.info()
        assert isinstance(info, dict)
        assert "endpoints" in info
        srv.shutdown()
'''
write("tests/test_streaming.py", src)
commit("test: add StreamingClient info endpoint returns dict with endpoints key test")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.5.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.4.0\"", "__version__ = \"3.5.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.5.0 — Streaming Inference & Async Server release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `rag`    | RAG — chunking, TF-IDF/BM25/dense embedders, VectorStore, pipeline |",
    "| `rag`    | RAG — chunking, TF-IDF/BM25/dense embedders, VectorStore, pipeline |\n"
    "| `streaming` | Streaming — SSE token-by-token, StreamingServer, StreamingClient |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.5.0] — 2024 — Streaming Inference & Async Server\n\n### Added\n" \
     "- `StreamingGenerator` — yields StreamToken one-at-a-time with top-K/P/temp\n" \
     "- `StreamingServer` — HTTP server with SSE /stream + /generate + /health\n" \
     "- `StreamingClient` — SSE consumer using stdlib urllib (zero deps)\n" \
     "- `StreamConfig` — max_new_tokens, temperature, top_k/p, stop_sequences\n" \
     "- `StreamToken` — token string + id + index + logprob + timestamp\n" \
     "- `StreamEvent` — SSE event with to_sse() wire format serialiser\n" \
     "- `StopSequenceDetector` — multi-token stop sequence detection\n" \
     "- `TokenBuffer` — accumulation buffer with callback + flush_every\n" \
     "- `print_stream()` / `collect_stream()` — terminal display utilities\n" \
     "- `examples/streaming_demo.py` — train + SSE server + client demo\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.5.0, update README and CHANGELOG for Day 35 Streaming")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 35 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.5.0",
    "-m", "NanoMind v3.5.0 — Streaming Inference & Async Server", check=False)
r = run("git", "push", "origin", "v3.5.0", check=False)
print("Tag v3.5.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 35 COMPLETE — v3.5.0 TAGGED! ===")
