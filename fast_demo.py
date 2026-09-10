"""
fast_demo.py — NanoMind fast live demo (CPU-optimised, token-by-token streaming)
"""
import sys, time, math, random, threading, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Tiny fast model (no scaled_dot_product_attention — plain matmul) ──────────
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out  = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.dh, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.dh)
        att   = (q @ k.transpose(-2, -1)) * scale
        mask  = torch.tril(torch.ones(T, T)).to(x.device)
        att   = att.masked_fill(mask == 0, float('-inf'))
        att   = F.softmax(att, dim=-1)
        return self.out((att @ v).transpose(1, 2).reshape(B, T, -1))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = Attn(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ff  = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, V, D=64, L=2, H=4, T=32):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(L)])
        self.ln  = nn.LayerNorm(D)
        self.lm  = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.lm(self.ln(x))
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate_one(self, idx, temperature=0.8, top_k=20):
        ctx    = idx[:, -self.T:]
        logits, _ = self(ctx)
        logits = logits[:, -1, :] / temperature
        v, _   = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
class Tok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return ''.join(self.i2s.get(i,'?') for i in ids)

# ── Data ───────────────────────────────────────────────────────────────────────
CORPUS = (
    "the quick brown fox jumps over the lazy dog\n"
    "NanoMind is a language model built from scratch in 30 days\n"
    "machine learning uses data to train neural network weights\n"
    "transformers learn from text using attention mechanisms\n"
    "python is great for deep learning and natural language processing\n"
) * 60

tok   = Tok(CORPUS)
V     = tok.vocab_size
T     = 32
ids   = torch.tensor(tok.encode(CORPUS))
xs    = torch.stack([ids[i:i+T]   for i in range(0, len(ids)-T-1, T)])
ys    = torch.stack([ids[i+1:i+T+1] for i in range(0, len(ids)-T-1, T)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=64, shuffle=True, drop_last=True)

# ── Train ──────────────────────────────────────────────────────────────────────
model = TinyLM(V, D=64, L=2, H=4, T=T)
n     = sum(p.numel() for p in model.parameters())
opt   = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.1)

print("=" * 55, flush=True)
print("  NanoMind — Fast Live Demo", flush=True)
print("=" * 55, flush=True)
print(f"  Model  : {n:,} params | vocab={V} | d=64 | layers=2", flush=True)
print(f"  Data   : {len(xs)} sequences of length {T}", flush=True)
print(flush=True)

model.train()
for epoch in range(6):
    total, steps = 0.0, 0
    for x, y in loader:
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); steps += 1
    print(f"  Epoch {epoch+1}/6  loss={total/steps:.4f}", flush=True)

# ── Generate ───────────────────────────────────────────────────────────────────
print(flush=True)
print("=" * 55, flush=True)
print("  LIVE GENERATION (token by token)", flush=True)
print("=" * 55, flush=True)

model.eval()
prompts = ["NanoMind is", "the quick", "machine learning", "python is"]
for prompt in prompts:
    enc = torch.tensor([tok.encode(prompt)]).long()
    sys.stdout.write(f"\n  >> {prompt}")
    sys.stdout.flush()
    for _ in range(40):
        nxt  = model.generate_one(enc, temperature=0.75, top_k=20)
        char = tok.decode(nxt[0].tolist())
        sys.stdout.write(char)
        sys.stdout.flush()
        enc  = torch.cat([enc, nxt], dim=1)
    sys.stdout.write("\n")
    sys.stdout.flush()

# ── REST API Server ────────────────────────────────────────────────────────────
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, code=200):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._send(json.dumps({"status":"ok","model":"NanoMind-v3.1.0","params":n}))
        elif self.path == "/info":
            self._send(json.dumps({"model":"NanoMind","params":n,"vocab":V,"layers":2,"d_model":64}))
        else: self._send(json.dumps({"error":"not found"}), 404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        if self.path == "/generate":
            prompt = body.get("prompt","hello")
            n_tok  = min(body.get("max_new_tokens", 40), 60)
            temp   = body.get("temperature", 0.8)
            enc    = torch.tensor([tok.encode(prompt)]).long()
            out    = []
            for _ in range(n_tok):
                nxt = model.generate_one(enc, temperature=temp, top_k=20)
                out.append(nxt.item())
                enc = torch.cat([enc, nxt], dim=1)
            self._send(json.dumps({"prompt":prompt,"generated":tok.decode(out),"tokens":n_tok}))
        else: self._send(json.dumps({"error":"not found"}), 404)

srv = HTTPServer(("127.0.0.1", 8787), H)
th  = threading.Thread(target=srv.serve_forever, daemon=True)
th.start()
time.sleep(0.1)

print(flush=True)
print("=" * 55, flush=True)
print("  REST API live at http://127.0.0.1:8787", flush=True)
print("=" * 55, flush=True)

import urllib.request
def call(path, body=None):
    url = f"http://127.0.0.1:8787{path}"
    if body:
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                      headers={"Content-Type":"application/json"})
    else: req = url
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

h = call("/health")
print(f"\n  GET /health   -> status={h['status']} | params={h['params']:,}", flush=True)

info = call("/info")
print(f"  GET /info     -> d_model={info['d_model']} | vocab={info['vocab']}", flush=True)

for p in ["NanoMind is", "the quick brown"]:
    r = call("/generate", {"prompt": p, "max_new_tokens": 30, "temperature": 0.75})
    print(f"\n  POST /generate", flush=True)
    print(f"    prompt    : {r['prompt']!r}", flush=True)
    print(f"    generated : {r['generated']!r}", flush=True)

print(flush=True)
print("  Server still running! Press Ctrl+C to stop.", flush=True)
print("=" * 55, flush=True)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    print("\n  Server stopped.")
