"""
examples/federated_demo.py — NanoMind Federated Learning demo.

Simulates 5 clients training a tiny LM with:
  - FedAvg aggregation
  - Differential Privacy (ε=1.0)
  - Top-K gradient compression

Usage:
    python examples/federated_demo.py
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.federated import (
    FederatedConfig, FederatedClient, FederatedServer,
    DifferentialPrivacyEngine, TopKCompressor,
    SecureAggregator, iid_partition, dirichlet_partition,
    fedavg, fedmedian,
)

# ── Tiny model + tokenizer ────────────────────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V=16, D=32, T=8):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

CORPUS = "federated learning privacy nanomind"
tok    = CharTok(CORPUS)
V      = tok.vocab_size

# ── Build dataset ─────────────────────────────────────────────────────────────
ids  = tok.encode(CORPUS)
T    = 6
data = []
for i in range(0, len(ids) - T - 1, 2):
    x = torch.tensor(ids[i:i+T])
    y = torch.tensor(ids[i+1:i+T+1])
    data.append((x, y))

print("=" * 60)
print("NanoMind Federated Learning Demo")
print("=" * 60)

# ── IID Partition ─────────────────────────────────────────────────────────────
cfg = FederatedConfig(n_clients=5, clients_per_round=3, n_rounds=3,
                      local_epochs=2, local_lr=1e-2, aggregation="fedavg",
                      dp_epsilon=1.0)
datasets = iid_partition(data, n_clients=5, batch_size=2)
print(f"
IID partition: {[d.n_samples for d in datasets]} batches per client")

# ── Dirichlet partition ───────────────────────────────────────────────────────
dir_datasets = dirichlet_partition(data, n_clients=5, alpha=0.5, batch_size=2)
print(f"Dirichlet (α=0.5): {[d.n_samples for d in dir_datasets]} batches per client")

# ── DP Engine ─────────────────────────────────────────────────────────────────
dp = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)
print(f"
DP noise multiplier: σ={dp.noise_multiplier:.4f}")
privacy = dp.privacy_spent(n_steps=3, n_samples=len(data), batch_size=2)
print(f"Privacy spent: {privacy}")

# ── Top-K Compression ─────────────────────────────────────────────────────────
global_model = TinyLM(V)
comp  = TopKCompressor(ratio=0.5)
# Quick forward to generate grads
x_t, y_t = data[0]
_, loss   = global_model(x_t.unsqueeze(0), y_t.unsqueeze(0))
loss.backward()
sparse = comp.compress(global_model)
print(f"
Top-K compression: kept {len(sparse['values'])}/{sparse['n_total']} grads "
      f"({comp.compression_ratio():.1%})")

# ── FedAvg training ───────────────────────────────────────────────────────────
print("
── FedAvg Training (3 rounds, 5 clients, DP enabled) ──")
global_model = TinyLM(V)
server = FederatedServer(global_model, cfg)
for i, ds in enumerate(datasets):
    dp_engine = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)
    client    = FederatedClient(f"client-{i}", copy.deepcopy(global_model),
                                ds.batches, cfg, dp_engine=dp_engine)
    server.add_client(client)

logs = server.train()
for entry in logs:
    print(f"  Round {entry['round']}: loss={entry['avg_loss']:.4f}, "
          f"clients={entry['n_clients']}")

# ── Secure Aggregation ────────────────────────────────────────────────────────
print("
── Secure Aggregation ──")
agg    = SecureAggregator("round-1")
grad   = torch.randn(100)
masked = agg.mask(grad, "client-0")
unmasked_sum = agg.unmask(masked, ["client-0"])
print(f"  Original:  {grad[:5].tolist()}")
print(f"  Masked:    {masked[:5].tolist()}")
print(f"  Unmasked:  {unmasked_sum[:5].tolist()}")
print(f"  Match: {torch.allclose(grad, unmasked_sum, atol=1e-5)}")
print("
Federated demo complete!")
