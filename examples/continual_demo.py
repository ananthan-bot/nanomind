"""
examples/continual_demo.py — NanoMind Continual Learning demo.

Simulates 3 sequential tasks:
  Task 0: train on "hello world" corpus
  Task 1: train on "federated privacy" corpus
  Task 2: train on "neural architecture" corpus

Demonstrates:
  1. Naive baseline (catastrophic forgetting)
  2. EWC (penalise important weight changes)
  3. Experience Replay (mix old data with new)
  4. PackNet (prune and pack per task)
  5. Continual metrics: AA, BWT, forgetting

Usage:
    python examples/continual_demo.py
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.continual import (
    ContinualConfig, EWC, SynapticIntelligence,
    ReplayBuffer, PackNet, ContinualTrainer,
    ContinualEvaluator, ContinualMetrics,
)

# ── Tiny model ────────────────────────────────────────────────────────────────
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

V = 16
N_TASKS = 3

def make_task(seed, T=6, n=8):
    """Create synthetic task batches."""
    torch.manual_seed(seed)
    batches = []
    for _ in range(n):
        x = torch.randint(0, V, (2, T))
        y = torch.randint(0, V, (2, T))
        batches.append((x, y))
    return batches

tasks = [make_task(seed=i*10) for i in range(N_TASKS)]

print("=" * 60)
print("NanoMind Continual Learning Demo")
print("=" * 60)

# ── EWC ───────────────────────────────────────────────────────────────────────
print("
── EWC (Elastic Weight Consolidation) ──")
model_ewc = TinyLM(V)
cfg_ewc   = ContinualConfig(strategy="ewc", n_tasks=N_TASKS, ewc_lambda=100.0)
trainer   = ContinualTrainer(model_ewc, cfg_ewc)
for t, task in enumerate(tasks):
    log = trainer.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
ewc_summary = trainer._ewc.fisher_summary(task_id=0)
print(f"  Fisher summary task 0: {ewc_summary}")

# ── SI ────────────────────────────────────────────────────────────────────────
print("
── Synaptic Intelligence (SI) ──")
model_si = TinyLM(V)
cfg_si   = ContinualConfig(strategy="si", n_tasks=N_TASKS, ewc_lambda=50.0)
trainer_si = ContinualTrainer(model_si, cfg_si)
for t, task in enumerate(tasks):
    log = trainer_si.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
print(f"  SI tasks registered: {trainer_si._si.n_tasks}")

# ── Replay ────────────────────────────────────────────────────────────────────
print("
── Experience Replay ──")
model_rp = TinyLM(V)
cfg_rp   = ContinualConfig(strategy="replay", n_tasks=N_TASKS, replay_buffer_size=100)
trainer_rp = ContinualTrainer(model_rp, cfg_rp)
for t, task in enumerate(tasks):
    log = trainer_rp.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
buf = trainer_rp._replay
print(f"  Buffer: {len(buf)} samples, tasks={buf.task_counts()}")

# ── PackNet ───────────────────────────────────────────────────────────────────
print("
── PackNet (pruning + masking) ──")
model_pn = TinyLM(V)
cfg_pn   = ContinualConfig(strategy="packnet", n_tasks=N_TASKS, packnet_prune_ratio=0.3)
trainer_pn = ContinualTrainer(model_pn, cfg_pn)
for t, task in enumerate(tasks):
    log = trainer_pn.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
pn = trainer_pn._packnet
print(f"  PackNet tasks: {pn.n_tasks}, free_ratio={pn.free_ratio:.2%}")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("
── Continual Metrics ──")
# Simulate accuracy matrix: 3 tasks, diagonal = task accuracy
acc_matrix = [
    [0.90, 0.72, 0.65],
    [0.00, 0.88, 0.70],
    [0.00, 0.00, 0.85],
]
metrics = ContinualMetrics(accuracy_matrix=acc_matrix, n_tasks=3)
print(f"  Average Accuracy:   {metrics.average_accuracy:.2%}")
print(f"  Backward Transfer:  {metrics.backward_transfer:.4f}")
print(f"  Forgetting:         {metrics.forgetting:.4f}")
m_dict = metrics.to_dict()
print(f"  Dict: {m_dict}")
print("
Continual learning demo complete!")
