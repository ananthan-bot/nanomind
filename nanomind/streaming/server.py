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

  data: {"event":"token","token":"Hello","index":0}\n\n
  data: {"event":"token","token":" world","index":1}\n\n
  data: {"event":"finish","finish_reason":"stop","n_tokens":2}\n\n
  data: [DONE]\n\n

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
