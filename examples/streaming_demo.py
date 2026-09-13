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

print("\n" + "=" * 55)
print("  NanoMind Streaming Demo")
print("=" * 55)

# ── StreamingGenerator ────────────────────────────────────────────────────────
cfg = StreamConfig(max_new_tokens=40, temperature=0.7, top_k=20, top_p=0.9)
gen = StreamingGenerator(model, tok, cfg)

print("\n── Token-by-token generation ──")
print("  Prompt: \"nanomind\"")
sys.stdout.write("  Output: ")
text, tokens = collect_stream(gen.stream("nanomind", cfg))
print(text)
print(f"  Tokens: {len(tokens)}")

# ── StopSequenceDetector ──────────────────────────────────────────────────────
print("\n── Stop sequence detection ──")
detector = StopSequenceDetector([" time", "\n"])
for tok_obj in gen.stream("tokens in real", cfg):
    detector.update(tok_obj.token)
    if detector.should_stop():
        break
print(f"  Generated: {detector.get_text()!r}")
print(f"  Stop reason: {detector.stop_reason!r}")

# ── SSE server + client ───────────────────────────────────────────────────────
print("\n── Streaming HTTP Server ──")
server = StreamingServer(gen, port=8789)
server.start_background()
time.sleep(0.2)

client = StreamingClient("http://127.0.0.1:8789")
h      = client.health()
print(f"  GET /health → {h}")

print("  Streaming \"language model\":")
sys.stdout.write("  → ")
result = client.stream_print("language model",
                              max_new_tokens=30, temperature=0.7)
print(f"  Total tokens streamed: {len(result.split())}")
print("\nStreaming demo complete!")
