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
    print(f"
Rate limiter: {allowed}/10 requests allowed (burst=5)")

print("
Server stopped. Demo complete!")
