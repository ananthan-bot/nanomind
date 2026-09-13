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
        sys.stdout.write("
")
        sys.stdout.flush()
        return "".join(parts)
