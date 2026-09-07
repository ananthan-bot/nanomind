"""
nanomind/serve/handler.py — HTTP request handler for the NanoMind inference server.

Implements endpoints using Python stdlib http.server (zero dependencies).

Endpoints:
  GET  /health   — liveness probe
  GET  /info     — model metadata
  POST /generate — text generation
  POST /tokenize — tokenize text
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler

from nanomind.serve.schemas import (
    GenerateRequest, TokenizeRequest,
    GenerateResponse, HealthResponse, InfoResponse,
    TokenizeResponse, ErrorResponse,
)
from nanomind.utils.logger import get_logger

log = get_logger("serve.handler")


def make_handler(engine, cfg):
    """
    Create a request handler class bound to the given engine and config.

    Uses a closure to inject dependencies without globals.

    Args:
        engine: :class:`InferenceEngine` instance.
        cfg:    :class:`ServeConfig` instance.

    Returns:
        A :class:`BaseHTTPRequestHandler` subclass.
    """

    class NanoMindHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if cfg.log_requests:
                log.info(fmt % args)

        def _send_json(self, body: str, status: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw    = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            if self.path == "/health":
                self._send_json(HealthResponse().to_json())
            elif self.path == "/info":
                info = engine.info()
                resp = InfoResponse(
                    model=info["model"],
                    vocab_size=info["vocab_size"],
                    d_model=info["d_model"],
                    n_layers=info["n_layers"],
                    n_heads=info["n_heads"],
                    n_params=info["n_params"],
                    block_size=info["block_size"],
                    device=info["device"],
                )
                self._send_json(resp.to_json())
            else:
                self._send_json(ErrorResponse("Not found", 404).to_json(), 404)

        def do_POST(self):
            try:
                body = self._read_body()
            except Exception as e:
                self._send_json(ErrorResponse(f"Bad JSON: {e}", 400).to_json(), 400)
                return

            if self.path == "/generate":
                try:
                    req = GenerateRequest.from_dict(body)
                    req.validate()
                    # Cap at server max
                    req.max_new_tokens = min(req.max_new_tokens, cfg.max_new_tokens)
                except (KeyError, ValueError) as e:
                    self._send_json(ErrorResponse(str(e), 422).to_json(), 422)
                    return
                try:
                    resp = engine.generate(req)
                    self._send_json(resp.to_json())
                except Exception:
                    tb = traceback.format_exc()
                    log.error(f"Generation error:
{tb}")
                    self._send_json(ErrorResponse("Generation failed", 500, tb).to_json(), 500)

            elif self.path == "/tokenize":
                try:
                    req  = TokenizeRequest.from_dict(body)
                    resp = engine.tokenize(req)
                    self._send_json(resp.to_json())
                except Exception as e:
                    self._send_json(ErrorResponse(str(e), 422).to_json(), 422)
            else:
                self._send_json(ErrorResponse("Not found", 404).to_json(), 404)

    return NanoMindHandler
