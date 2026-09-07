"""
nanomind/serve/server.py — ModelServer: lifecycle management for the HTTP server.
"""

from __future__ import annotations

import threading
import time
from http.server import HTTPServer

import torch

from nanomind.serve.config import ServeConfig
from nanomind.serve.engine import InferenceEngine
from nanomind.serve.handler import make_handler
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("serve.server")


class ModelServer:
    """
    HTTP inference server for NanoMind.

    Manages server lifecycle: start, stop, and background serving.
    Uses Python stdlib ``HTTPServer`` — no external dependencies.

    Args:
        model:     Language model.
        tokenizer: Tokenizer.
        cfg:       Server configuration.
        device:    Inference device.

    Example::

        server = ModelServer(model, tokenizer, ServeConfig(port=8080))
        server.start()   # blocks until Ctrl+C or server.stop()

    Background serving::

        server.start_background()
        # ... do other work ...
        server.stop()
    """

    def __init__(
        self,
        model:     torch.nn.Module,
        tokenizer: BaseTokenizer,
        cfg:       ServeConfig | None = None,
        device:    str | torch.device = "cpu",
    ) -> None:
        self.cfg     = cfg or ServeConfig()
        self.engine  = InferenceEngine(model, tokenizer, device, self.cfg.model_name)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _make_server(self) -> HTTPServer:
        handler = make_handler(self.engine, self.cfg)
        server  = HTTPServer((self.cfg.host, self.cfg.port), handler)
        server.timeout = self.cfg.timeout_sec
        return server

    def start(self) -> None:
        """Start the server and block (use Ctrl+C to stop)."""
        self._server = self._make_server()
        log.info(f"NanoMind server running at {self.cfg.base_url}")
        log.info(f"  GET  {self.cfg.base_url}/health")
        log.info(f"  GET  {self.cfg.base_url}/info")
        log.info(f"  POST {self.cfg.base_url}/generate")
        log.info(f"  POST {self.cfg.base_url}/tokenize")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped (KeyboardInterrupt)")
        finally:
            self._server.server_close()

    def start_background(self) -> None:
        """Start the server in a daemon background thread."""
        self._server = self._make_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="nanomind-server",
        )
        self._thread.start()
        # Brief pause to let the server bind
        time.sleep(0.05)
        log.info(f"NanoMind server running (background) at {self.cfg.base_url}")

    def stop(self) -> None:
        """Shut down the background server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        log.info("Server stopped")

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def __enter__(self):
        self.start_background()
        return self

    def __exit__(self, *args):
        self.stop()
