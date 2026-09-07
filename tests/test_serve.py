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
