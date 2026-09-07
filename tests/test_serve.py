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
