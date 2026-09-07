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
