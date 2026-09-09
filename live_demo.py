"""
live_demo.py — NanoMind live inference server demo.
Trains a real character-level model and starts the REST API server.
"""
import sys, os, pathlib, time, threading, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Inline minimal model (avoids import chain issues) ──────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.w   = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt() * self.w

class SelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out  = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, D))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1  = RMSNorm(d); self.attn = SelfAttention(d, h)
        self.n2  = RMSNorm(d)
        dff = d * 4
        self.w1  = nn.Linear(d, dff, bias=False)
        self.w2  = nn.Linear(d, dff, bias=False)
        self.w3  = nn.Linear(dff, d, bias=False)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        x = x + self.w3(F.silu(self.w1(h)) * self.w2(h))
        return x

class NanoLM(nn.Module):
    def __init__(self, V, D, L, H, T):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(L)])
        self.norm   = RMSNorm(D)
        self.head   = nn.Linear(D, V, bias=False)
        self.T = T
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.norm(x))
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=40):
        for _ in range(max_new):
            ctx    = idx[:, -self.T:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)
        return idx

# ── Tokenizer ──────────────────────────────────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return ''.join(self.i2s.get(i, '?') for i in ids)

# ── Corpus & training ──────────────────────────────────────────────────────────
CORPUS = """
NanoMind is a production-grade language model library built from scratch in 30 days.
It implements tokenizers, transformers, attention, beam search, LoRA fine-tuning,
speculative decoding, quantization, mixture of experts, flash attention, KV cache,
mixed precision training, RLHF, DPO, knowledge distillation, and REST API serving.

The quick brown fox jumps over the lazy dog.
Python is a high level programming language used for machine learning and AI.
Transformers use self attention to process sequences in parallel efficiently.
The loss function measures how wrong the model predictions are during training.
Gradient descent updates model weights to minimize the training loss over time.
Language models predict the next token given all previous tokens in a sequence.
NanoMind v3.0.0 has 632 commits, 21 sub-packages, built over 30 days of coding.

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.model.config import ModelConfig
from nanomind.serve import ModelServer, ServeConfig
from nanomind.dpo import DPOTrainer, DPOConfig
""" * 15

print("=" * 55)
print("  NanoMind v3.0.0 — Live Inference Demo")
print("=" * 55)

tok   = CharTok(CORPUS)
ids   = torch.tensor(tok.encode(CORPUS))
BLOCK = 128
V, D, L, H = tok.vocab_size, 256, 6, 8

xs = torch.stack([ids[i:i+BLOCK]     for i in range(0, len(ids)-BLOCK-1, BLOCK)])
ys = torch.stack([ids[i+1:i+BLOCK+1] for i in range(0, len(ids)-BLOCK-1, BLOCK)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True, drop_last=True)

model = NanoLM(V, D, L, H, BLOCK)
n = sum(p.numel() for p in model.parameters())
print(f"  Model: {n:,} params | vocab={V} | d={D} | layers={L} | heads={H}")
print(f"  Training on {len(xs)} batches per epoch...")
print()

opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)
model.train()
for epoch in range(12):
    total = 0
    for x, y in loader:
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    avg = total / len(loader)
    print(f"  Epoch {epoch+1:>2}/12  loss={avg:.4f}", flush=True)

# ── Live generation ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  🚀 LIVE GENERATION")
print("=" * 55)
model.eval()

prompts = [
    ("NanoMind is",      80, 0.7),
    ("import torch",     80, 0.8),
    ("The transformer",  80, 0.7),
    ("from nanomind",    80, 0.8),
]

for prompt, length, temp in prompts:
    enc = torch.tensor([tok.encode(prompt)]).long()
    out = model.generate(enc, max_new=length, temperature=temp, top_k=40)
    generated = tok.decode(out[0].tolist())
    print(f'\n  Prompt: "{prompt}"')
    print(f'  Output: {generated}')

# ── Start REST API server ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  🌐 STARTING REST API SERVER on port 8787")
print("=" * 55)

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # silence access logs

    def _json(self, body, status=200):
        b = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._json(json.dumps({"status": "ok", "model": "NanoMind-v3.0.0", "params": n}))
        elif self.path == "/info":
            self._json(json.dumps({
                "model": "NanoMind-v3.0.0", "params": n,
                "vocab_size": V, "d_model": D, "n_layers": L, "n_heads": H,
                "block_size": BLOCK, "device": "cpu"
            }))
        else:
            self._json(json.dumps({"error": "not found"}), 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        if self.path == "/generate":
            prompt      = body.get("prompt", "NanoMind")
            max_tokens  = min(body.get("max_new_tokens", 80), 200)
            temperature = body.get("temperature", 0.8)
            top_k       = body.get("top_k", 40)

            enc = torch.tensor([tok.encode(prompt)]).long()
            with torch.no_grad():
                out = model.generate(enc, max_new=max_tokens, temperature=temperature, top_k=top_k)
            text = tok.decode(out[0].tolist())
            generated = text[len(prompt):]
            self._json(json.dumps({
                "text": generated,
                "full": text,
                "prompt_tokens": len(enc[0]),
                "generated_tokens": max_tokens,
                "model": "NanoMind-v3.0.0"
            }))
        else:
            self._json(json.dumps({"error": "not found"}), 404)

server = HTTPServer(("127.0.0.1", 8787), Handler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
print("  Server live at http://127.0.0.1:8787")
print()

# ── Live client test ──────────────────────────────────────────────────────────
import urllib.request

def call(endpoint, body=None):
    url = f"http://127.0.0.1:8787{endpoint}"
    if body:
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"})
    else:
        req = url
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

time.sleep(0.2)

h = call("/health")
print(f"  GET /health  → status={h['status']}, params={h['params']:,}")

info = call("/info")
print(f"  GET /info    → d_model={info['d_model']}, layers={info['n_layers']}, vocab={info['vocab_size']}")

for prompt in ["NanoMind is a", "import torch\n", "The transformer"]:
    resp = call("/generate", {"prompt": prompt, "max_new_tokens": 60, "temperature": 0.75})
    print(f'\n  POST /generate  prompt="{prompt.strip()}"')
    print(f"  → {resp['full'][:120]!r}")

print("\n" + "=" * 55)
print("  ✅ NanoMind REST API running! Ctrl+C to stop.")
print("=" * 55)
print("\n  Try it yourself:")
print('  curl -X POST http://127.0.0.1:8787/generate \\')
print('       -H "Content-Type: application/json" \\')
print('       -d \'{"prompt": "NanoMind is", "max_new_tokens": 80}\'')
print()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServer stopped.")
