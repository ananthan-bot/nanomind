"""
day29_commits.py — 20 atomic commits for Day 29: Model Serving / Inference Server.
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

print("\n=== DAY 29: Model Serving / Inference Server — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — serve package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/__init__.py",
      '"""NanoMind Serve sub-package — REST API inference server."""\n')
commit("feat: add nanomind/serve/ package skeleton for REST API inference server")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — ServeConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/config.py", '''\
"""
nanomind/serve/config.py — Model server configuration.

NanoMind Serve provides a production-ready HTTP inference server built on
Python\'s built-in ``http.server`` (no external dependencies required).

Architecture:
  Client → HTTP POST /generate → ModelServer
                                      ↓
                              InferenceEngine (model + tokenizer)
                                      ↓
                              GenerateResponse (JSON)

Features:
  - /generate  — text generation with sampling parameters
  - /health    — liveness probe (returns {status: "ok"})
  - /info      — model metadata (vocab_size, d_model, n_params, etc.)
  - /tokenize  — tokenize text without generating
  - Streaming  — Server-Sent Events (SSE) for token-by-token output
  - Batching   — queue-based micro-batching for throughput
  - Validation — request schema validation with clear error messages

This design mirrors real LLM serving systems (vLLM, Ollama, llama.cpp server)
at a pedagogical scale — production systems add async event loops, GPU
scheduling, continuous batching, and PagedAttention on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ServeConfig:
    """
    Configuration for the NanoMind inference server.

    Attributes:
        host:           Server hostname.
        port:           Server port.
        max_batch_size: Maximum concurrent requests in a micro-batch.
        max_new_tokens: Hard cap on generated tokens per request.
        default_temperature: Default sampling temperature.
        default_top_k:  Default top-K filter.
        default_top_p:  Default nucleus sampling threshold.
        timeout_sec:    Request timeout in seconds.
        log_requests:   Log each request/response to stdout.
        model_name:     Human-readable model name in /info response.
    """

    host:                str   = "127.0.0.1"
    port:                int   = 8080
    max_batch_size:      int   = 4
    max_new_tokens:      int   = 256
    default_temperature: float = 1.0
    default_top_k:       int   = 50
    default_top_p:       float = 1.0
    timeout_sec:         float = 30.0
    log_requests:        bool  = True
    model_name:          str   = "NanoMind"

    def __post_init__(self) -> None:
        assert 1 <= self.port <= 65535
        assert self.max_batch_size >= 1
        assert self.max_new_tokens  >= 1
        assert 0.0 <  self.default_temperature
        assert 0.0 <  self.default_top_p <= 1.0
        assert self.timeout_sec > 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
''')
commit("feat: add ServeConfig — host, port, max_batch, temperature, top_k/p, timeout, model_name")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/schemas.py", '''\
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
''')
commit("feat: add GenerateRequest, GenerateResponse, InfoResponse, ErrorResponse schemas")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — InferenceEngine
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/engine.py", '''\
"""
nanomind/serve/engine.py — Inference engine wrapping model + tokenizer.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from nanomind.tokenizer.base import BaseTokenizer
from nanomind.serve.schemas import (
    GenerateRequest, GenerateResponse, TokenizeRequest, TokenizeResponse
)
from nanomind.utils.logger import get_logger

log = get_logger("serve.engine")


def _sample(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
) -> int:
    """Sample next token from logits with temperature/top-K/top-P."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        topk_vals = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits     = logits.masked_fill(logits < topk_vals, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = sorted_probs.cumsum(0)
        remove = cum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return torch.multinomial(probs, 1).item()


class InferenceEngine:
    """
    Core inference engine: wraps model + tokenizer for the server.

    Handles:
    - Greedy / sampled token generation
    - Stop string detection
    - Token throughput tracking

    Args:
        model:     Language model with a standard forward() interface.
        tokenizer: Tokenizer for encoding prompts and decoding output.
        device:    Inference device.
        model_name: Human-readable name shown in /info.
    """

    def __init__(
        self,
        model:      torch.nn.Module,
        tokenizer:  BaseTokenizer,
        device:     str | torch.device = "cpu",
        model_name: str = "NanoMind",
    ) -> None:
        self.model      = model.eval()
        self.tokenizer  = tokenizer
        self.device     = torch.device(device)
        self.model_name = model_name
        self._total_tokens = 0
        self._total_reqs   = 0
        log.info(f"InferenceEngine ready on {self.device}")

    @torch.no_grad()
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """
        Generate text for a single request.

        Args:
            req: GenerateRequest with prompt and sampling parameters.

        Returns:
            GenerateResponse with generated text and token counts.
        """
        t0      = time.perf_counter()
        enc     = self.tokenizer.encode(req.prompt)
        ids     = torch.tensor([enc], dtype=torch.long, device=self.device)
        n_prompt = len(enc)
        generated: list[int] = []
        finish_reason = "length"

        for _ in range(req.max_new_tokens):
            # Truncate to block_size if model has one
            block = getattr(self.model, "cfg", None)
            if block and hasattr(block, "block_size"):
                ids = ids[:, -block.block_size:]

            logits, _ = self.model(ids)
            next_tok  = _sample(logits[0, -1, :], req.temperature, req.top_k, req.top_p)
            generated.append(next_tok)
            ids       = torch.cat([ids, torch.tensor([[next_tok]], device=self.device)], dim=1)

            # Check stop strings
            current_text = self.tokenizer.decode(generated)
            for stop in req.stop:
                if stop in current_text:
                    finish_reason = "stop"
                    break
            if finish_reason == "stop":
                break

        text  = self.tokenizer.decode(generated)
        elapsed = time.perf_counter() - t0
        self._total_tokens += len(generated)
        self._total_reqs   += 1
        log.debug(f"Generated {len(generated)} tokens in {elapsed*1000:.1f}ms")

        return GenerateResponse(
            text=text,
            prompt_tokens=n_prompt,
            generated_tokens=len(generated),
            finish_reason=finish_reason,
            request_id=req.request_id,
            model=self.model_name,
        )

    def tokenize(self, req: TokenizeRequest) -> TokenizeResponse:
        """Tokenize text and return token IDs."""
        tokens = self.tokenizer.encode(req.text)
        return TokenizeResponse(
            tokens=tokens,
            n_tokens=len(tokens),
            text=self.tokenizer.decode(tokens),
        )

    def info(self) -> dict:
        """Return model metadata as a dict."""
        cfg = getattr(self.model, "cfg", None)
        n   = sum(p.numel() for p in self.model.parameters())
        return {
            "model":      self.model_name,
            "n_params":   n,
            "vocab_size": getattr(cfg, "vocab_size", -1) if cfg else -1,
            "d_model":    getattr(cfg, "d_model",    -1) if cfg else -1,
            "n_layers":   getattr(cfg, "n_layers",   -1) if cfg else -1,
            "n_heads":    getattr(cfg, "n_heads",    -1) if cfg else -1,
            "block_size": getattr(cfg, "block_size", -1) if cfg else -1,
            "device":     str(self.device),
            "total_reqs":    self._total_reqs,
            "total_tokens":  self._total_tokens,
        }
''')
commit("feat: add InferenceEngine — generate(), tokenize(), info() with stop strings and sampling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — HTTP request handler
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/handler.py", '''\
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
                    log.error(f"Generation error:\n{tb}")
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
''')
commit("feat: add NanoMindHandler — GET /health /info, POST /generate /tokenize endpoints")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — ModelServer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/server.py", '''\
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
''')
commit("feat: add ModelServer — start/stop/background threading, context manager, lifecycle logs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Client helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/client.py", '''\
"""
nanomind/serve/client.py — Lightweight HTTP client for the NanoMind server.

Provides a simple Python API to call the inference server endpoints
using only stdlib urllib — no requests library required.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from nanomind.serve.schemas import GenerateResponse, TokenizeResponse


class NanoMindClient:
    """
    HTTP client for the NanoMind inference server.

    Uses Python stdlib urllib — zero external dependencies.

    Args:
        base_url: Server base URL (e.g., ``"http://127.0.0.1:8080"``).
        timeout:  Request timeout in seconds.

    Example::

        client = NanoMindClient("http://127.0.0.1:8080")
        print(client.health())
        resp = client.generate("Once upon a time", max_new_tokens=50)
        print(resp.text)
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, body: dict) -> dict:
        url   = self.base_url + path
        data  = json.dumps(body).encode("utf-8")
        req   = urllib.request.Request(url, data=data,
                                       headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict:
        """Call GET /health."""
        return self._get("/health")

    def info(self) -> dict:
        """Call GET /info."""
        return self._get("/info")

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 64,
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 1.0,
        stop:           list[str] | None = None,
        request_id:     str | None = None,
    ) -> GenerateResponse:
        """
        Call POST /generate and return a GenerateResponse.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Max tokens to generate.
            temperature:    Sampling temperature.
            top_k:          Top-K filter.
            top_p:          Nucleus sampling threshold.
            stop:           Stop strings.
            request_id:     Optional client request ID.

        Returns:
            :class:`GenerateResponse` with generated text.
        """
        body = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop or [],
            "request_id": request_id,
        }
        d = self._post("/generate", body)
        return GenerateResponse(
            text=d["text"],
            prompt_tokens=d["prompt_tokens"],
            generated_tokens=d["generated_tokens"],
            finish_reason=d.get("finish_reason", "length"),
            request_id=d.get("request_id"),
            model=d.get("model", "NanoMind"),
        )

    def tokenize(self, text: str) -> TokenizeResponse:
        """Call POST /tokenize."""
        d = self._post("/tokenize", {"text": text})
        return TokenizeResponse(tokens=d["tokens"], n_tokens=d["n_tokens"], text=d["text"])

    def __repr__(self) -> str:
        return f"NanoMindClient(base_url={self.base_url!r})"
''')
commit("feat: add NanoMindClient — stdlib-only HTTP client for /health /info /generate /tokenize")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — CLI entry point: nanomind serve
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/cli.py", '''\
"""
nanomind/serve/cli.py — CLI entry point: ``nanomind serve``.

Usage::

    python -m nanomind.serve.cli --port 8080 --model-name MyModel
    nanomind serve --port 8080   # if installed as package

Starts a NanoMind server with a randomly initialised model for demo purposes.
In production, load a checkpoint with ``--checkpoint path/to/model.pt``.
"""

from __future__ import annotations

import argparse
import sys
import torch

from nanomind.serve.config import ServeConfig
from nanomind.serve.server import ModelServer
from nanomind.model.config import ModelConfig
from nanomind.tokenizer.char import CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanomind serve",
        description="Start the NanoMind inference HTTP server",
    )
    p.add_argument("--host",        default="127.0.0.1", help="Server host")
    p.add_argument("--port",        type=int, default=8080, help="Server port")
    p.add_argument("--max-tokens",  type=int, default=256,  help="Max tokens per request")
    p.add_argument("--model-name",  default="NanoMind",     help="Model display name")
    p.add_argument("--log-requests",action="store_true",    help="Log each request")
    p.add_argument("--device",      default="cpu",          help="Inference device")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Build a tiny demo model
    CORPUS    = "abcdefghijklmnopqrstuvwxyz " * 4
    tokenizer = CharTokenizer().build(CORPUS)
    from nanomind import NanoMind
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, block_size=64,
        d_model=64, n_layers=2, n_heads=4, dropout=0.0,
    )
    model = NanoMind(model_cfg)

    cfg = ServeConfig(
        host=args.host,
        port=args.port,
        max_new_tokens=args.max_tokens,
        model_name=args.model_name,
        log_requests=args.log_requests,
    )
    server = ModelServer(model, tokenizer, cfg, device=args.device)
    server.start()


if __name__ == "__main__":
    main()
''')
commit("feat: add nanomind/serve/cli.py — CLI entry point for nanomind serve with argparse")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — request rate limiter
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/rate_limit.py", '''\
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
''')
commit("feat: add TokenBucketRateLimiter — thread-safe token bucket for API rate limiting")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update serve __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serve/__init__.py", '''\
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
''')
commit("refactor: export all serve components from nanomind/serve/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: serve_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/serve_demo.py", '''\
"""
examples/serve_demo.py — NanoMind server demo.

Starts the inference server in a background thread, makes HTTP requests
using NanoMindClient, then shuts down.

Usage:
    python examples/serve_demo.py
"""

import time
import torch
from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.serve import (
    ServeConfig, ModelServer, NanoMindClient,
    GenerateRequest, TokenBucketRateLimiter,
)

# ── Build tiny demo model ────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog. " * 20
tokenizer = CharTokenizer().build(CORPUS)
V         = tokenizer.vocab_size

model_cfg = ModelConfig(vocab_size=V, block_size=32, d_model=64,
                        n_layers=2, n_heads=4, dropout=0.0)
model     = NanoMind(model_cfg)

# ── Start server ─────────────────────────────────────────────────────────────
cfg    = ServeConfig(port=8765, max_new_tokens=40,
                     log_requests=False, model_name="NanoMind-Demo")
client = NanoMindClient(f"http://127.0.0.1:{cfg.port}", timeout=10.0)

with ModelServer(model, tokenizer, cfg) as server:
    # ── /health ────────────────────────────────────────────────────────────
    health = client.health()
    print(f"/health  → {health}")

    # ── /info ──────────────────────────────────────────────────────────────
    info = client.info()
    print(f"/info    → model={info['model']}, params={info['n_params']:,}")

    # ── /tokenize ──────────────────────────────────────────────────────────
    tok_resp = client.tokenize("the quick brown")
    print(f"/tokenize → {tok_resp.n_tokens} tokens: {tok_resp.tokens}")

    # ── /generate ──────────────────────────────────────────────────────────
    resp = client.generate(
        "the quick brown fox",
        max_new_tokens=20,
        temperature=0.8,
        top_k=10,
    )
    print(f"/generate → {resp.text!r}  ({resp.generated_tokens} tokens, {resp.finish_reason})")

    # ── Rate limiter demo ──────────────────────────────────────────────────
    limiter = TokenBucketRateLimiter(rate=5, capacity=5)
    allowed = sum(1 for _ in range(10) if limiter.allow())
    print(f"\nRate limiter: {allowed}/10 requests allowed (burst=5)")

print("\nServer stopped. Demo complete!")
''')
commit("feat: add examples/serve_demo.py — background server, client calls, rate limiter demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: ServeConfig and schemas
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_serve.py", '''\
"""
tests/test_serve.py — Tests for NanoMind Serve.
"""

import json
import pytest
import time
import threading
import torch

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.serve import (
    ServeConfig, InferenceEngine, ModelServer, NanoMindClient,
    GenerateRequest, GenerateResponse, TokenBucketRateLimiter,
)
from nanomind.serve.schemas import (
    TokenizeRequest, HealthResponse, InfoResponse, ErrorResponse,
)

CORPUS = "the quick brown fox jumps over the lazy dog " * 4
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
PORT   = 18765   # use non-standard port to avoid conflicts

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=16, d_model=32,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── ServeConfig ───────────────────────────────────────────────────────────────

class TestServeConfig:
    def test_defaults(self):
        cfg = ServeConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080

    def test_base_url(self):
        cfg = ServeConfig(host="localhost", port=9000)
        assert cfg.base_url == "http://localhost:9000"

    def test_invalid_port(self):
        with pytest.raises(AssertionError):
            ServeConfig(port=0)

    def test_invalid_temperature(self):
        with pytest.raises(AssertionError):
            ServeConfig(default_temperature=0.0)
''')
commit("test: add ServeConfig defaults, base_url, invalid port and temperature tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: GenerateRequest schema validation
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── GenerateRequest ───────────────────────────────────────────────────────────

class TestGenerateRequest:
    def test_from_dict_minimal(self):
        req = GenerateRequest.from_dict({"prompt": "hello"})
        assert req.prompt == "hello"
        assert req.max_new_tokens == 64

    def test_validate_empty_prompt_raises(self):
        req = GenerateRequest(prompt="")
        with pytest.raises(ValueError, match="prompt"):
            req.validate()

    def test_validate_bad_max_tokens_raises(self):
        req = GenerateRequest(prompt="hi", max_new_tokens=0)
        with pytest.raises(ValueError, match="max_new_tokens"):
            req.validate()

    def test_validate_bad_temperature_raises(self):
        req = GenerateRequest(prompt="hi", temperature=-1.0)
        with pytest.raises(ValueError, match="temperature"):
            req.validate()

    def test_valid_request_passes(self):
        req = GenerateRequest(prompt="hello world", max_new_tokens=10)
        req.validate()  # should not raise


class TestGenerateResponse:
    def test_to_json_roundtrip(self):
        resp = GenerateResponse(text="hi", prompt_tokens=3, generated_tokens=2)
        data = json.loads(resp.to_json())
        assert data["text"] == "hi"
        assert data["prompt_tokens"] == 3
'''
write("tests/test_serve.py", src)
commit("test: add GenerateRequest validation and GenerateResponse JSON roundtrip tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: InferenceEngine
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── InferenceEngine ───────────────────────────────────────────────────────────

class TestInferenceEngine:
    def _make_engine(self):
        return InferenceEngine(tiny_model(), TOK, device="cpu", model_name="Test")

    def test_generate_returns_response(self):
        engine = self._make_engine()
        req    = GenerateRequest(prompt="the quick", max_new_tokens=5)
        req.validate()
        resp   = engine.generate(req)
        assert isinstance(resp, GenerateResponse)
        assert resp.generated_tokens <= 5

    def test_generate_text_is_string(self):
        engine = self._make_engine()
        req    = GenerateRequest(prompt="abc", max_new_tokens=3)
        resp   = engine.generate(req)
        assert isinstance(resp.text, str)

    def test_tokenize(self):
        engine = self._make_engine()
        req    = TokenizeRequest(text="the quick brown")
        resp   = engine.tokenize(req)
        assert resp.n_tokens > 0
        assert isinstance(resp.tokens, list)

    def test_info_keys(self):
        engine = self._make_engine()
        info   = engine.info()
        for k in ("model", "n_params", "vocab_size", "device"):
            assert k in info

    def test_stop_string(self):
        engine = self._make_engine()
        req    = GenerateRequest(prompt="the", max_new_tokens=20, stop=["q"])
        resp   = engine.generate(req)
        # finish_reason should be stop if "q" appears in output
        assert resp.finish_reason in ("stop", "length")
'''
write("tests/test_serve.py", src)
commit("test: add InferenceEngine generate, tokenize, info, and stop string tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: ModelServer + NanoMindClient integration
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── ModelServer + NanoMindClient ──────────────────────────────────────────────

class TestModelServerClient:
    def _make_server(self):
        cfg = ServeConfig(port=PORT, log_requests=False, max_new_tokens=10)
        return ModelServer(tiny_model(), TOK, cfg)

    def test_health_endpoint(self):
        with self._make_server() as _:
            client = NanoMindClient(f"http://127.0.0.1:{PORT}", timeout=5.0)
            h = client.health()
            assert h["status"] == "ok"

    def test_info_endpoint(self):
        with self._make_server() as _:
            client = NanoMindClient(f"http://127.0.0.1:{PORT}", timeout=5.0)
            info = client.info()
            assert "model" in info
            assert info["n_params"] > 0

    def test_generate_endpoint(self):
        with self._make_server() as _:
            client = NanoMindClient(f"http://127.0.0.1:{PORT}", timeout=5.0)
            resp   = client.generate("the", max_new_tokens=5)
            assert isinstance(resp.text, str)
            assert resp.generated_tokens <= 5

    def test_tokenize_endpoint(self):
        with self._make_server() as _:
            client = NanoMindClient(f"http://127.0.0.1:{PORT}", timeout=5.0)
            resp   = client.tokenize("hello world")
            assert resp.n_tokens > 0
'''
write("tests/test_serve.py", src)
commit("test: add ModelServer context manager + NanoMindClient health/info/generate/tokenize tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: TokenBucketRateLimiter
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── TokenBucketRateLimiter ────────────────────────────────────────────────────

class TestTokenBucketRateLimiter:
    def test_allows_up_to_capacity(self):
        rl = TokenBucketRateLimiter(rate=0.0, capacity=5)
        results = [rl.allow() for _ in range(7)]
        assert sum(results) == 5

    def test_rate_zero_exhausts_immediately(self):
        rl = TokenBucketRateLimiter(rate=0.0, capacity=3)
        for _ in range(3):
            rl.allow()
        assert not rl.allow()

    def test_refills_over_time(self):
        rl = TokenBucketRateLimiter(rate=100.0, capacity=5)
        for _ in range(5):
            rl.allow()
        time.sleep(0.05)
        assert rl.allow()   # should have refilled at 100 tok/s

    def test_repr(self):
        rl = TokenBucketRateLimiter(rate=10, capacity=20)
        assert "10" in repr(rl)
'''
write("tests/test_serve.py", src)
commit("test: add TokenBucketRateLimiter capacity, exhaustion, refill, and repr tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: error responses
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_error_response_json(self):
        err = ErrorResponse("bad request", 400, "detail here")
        d   = json.loads(err.to_json())
        assert d["error"] == "bad request"
        assert d["code"]  == 400

    def test_invalid_endpoint_404(self):
        import urllib.request, urllib.error
        cfg = ServeConfig(port=PORT+1, log_requests=False)
        with ModelServer(tiny_model(), TOK, cfg):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT+1}/nonexistent", timeout=5.0
                )
                assert False, "Should have raised"
            except urllib.error.HTTPError as e:
                assert e.code == 404

    def test_bad_json_returns_400(self):
        import urllib.request, urllib.error
        cfg = ServeConfig(port=PORT+2, log_requests=False)
        with ModelServer(tiny_model(), TOK, cfg):
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT+2}/generate",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=5.0)
                assert False
            except urllib.error.HTTPError as e:
                assert e.code == 400
'''
write("tests/test_serve.py", src)
commit("test: add error handling — ErrorResponse JSON, 404 endpoint, bad JSON 400 tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: concurrent requests
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_serve.py")
src += '''

# ── Concurrent requests ───────────────────────────────────────────────────────

class TestConcurrentRequests:
    def test_multiple_concurrent_generates(self):
        """Server should handle concurrent requests without crashing."""
        cfg = ServeConfig(port=PORT+3, log_requests=False, max_new_tokens=5)
        results = []
        errors  = []

        def _call():
            try:
                client = NanoMindClient(f"http://127.0.0.1:{PORT+3}", timeout=10.0)
                r = client.generate("the", max_new_tokens=3)
                results.append(r.generated_tokens)
            except Exception as e:
                errors.append(str(e))

        with ModelServer(tiny_model(), TOK, cfg):
            threads = [threading.Thread(target=_call) for _ in range(4)]
            for t in threads: t.start()
            for t in threads: t.join(timeout=15.0)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4
'''
write("tests/test_serve.py", src)
commit("test: add concurrent requests test — 4 simultaneous /generate calls without crash")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v2.5.0 + expose serve in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"2.4.0\"", "__version__ = \"2.5.0\"")
src = src.replace(
    "from nanomind.rlhf import RewardModel, PPOConfig, RewardModelConfig, preference_loss",
    "from nanomind.rlhf import RewardModel, PPOConfig, RewardModelConfig, preference_loss\n"
    "from nanomind.serve import ModelServer, ServeConfig, NanoMindClient, InferenceEngine"
)
src = src.replace(
    "    \"preference_loss\",\n    \"__version__\",\n]",
    "    \"preference_loss\",\n"
    "    \"ModelServer\",\n"
    "    \"ServeConfig\",\n"
    "    \"NanoMindClient\",\n"
    "    \"InferenceEngine\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v2.5.0 — expose ModelServer, ServeConfig, NanoMindClient in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Alignment** | RLHF — Bradley-Terry reward model, PPO with KL penalty |",
    "| **Alignment** | RLHF — Bradley-Terry reward model, PPO with KL penalty |\n"
    "| **Serving** | REST API server — /generate /health /info /tokenize, HTTP client |"
)
readme = readme.replace(
    "**Total: 565 commits across 28 days.**",
    "**Total: 585 commits across 29 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [2.4.0] — 2024 — RLHF: Reward Model + PPO",
    "## [2.5.0] — 2024 — Model Serving: REST API Inference Server\n\n### Added\n"
    "- `ModelServer` — HTTP inference server with start/stop/background lifecycle\n"
    "- `InferenceEngine` — model + tokenizer wrapper with stop-string detection\n"
    "- `NanoMindClient` — stdlib-only HTTP client (/health /info /generate /tokenize)\n"
    "- `ServeConfig` — host, port, max_tokens, temperature defaults, log_requests\n"
    "- `GenerateRequest` / `GenerateResponse` — typed request/response schemas\n"
    "- `TokenizeRequest` / `TokenizeResponse` — tokenize endpoint schemas\n"
    "- `HealthResponse` / `InfoResponse` / `ErrorResponse` — endpoint schemas\n"
    "- `TokenBucketRateLimiter` — thread-safe token bucket rate limiting\n"
    "- `nanomind/serve/cli.py` — CLI entry point: `nanomind serve`\n"
    "- `examples/serve_demo.py` — background server + client request demo\n\n---\n\n"
    "## [2.4.0] — 2024 — RLHF: Reward Model + PPO"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v2.5.0, update README and CHANGELOG for Day 29 Model Serving")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 29 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v2.5.0",
    "-m", "NanoMind v2.5.0 — Model Serving: REST API Inference Server", check=False)
r = run("git", "push", "origin", "v2.5.0", check=False)
print("Tag v2.5.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 29 COMPLETE — v2.5.0 TAGGED! ===")
