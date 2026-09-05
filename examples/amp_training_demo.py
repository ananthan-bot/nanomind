"""
examples/amp_training_demo.py — AMP + Gradient Checkpointing training demo.

Demonstrates:
  1. Mixed precision (bfloat16) training with AMPTrainer
  2. Gradient accumulation for large effective batch sizes
  3. Activation memory estimation
  4. Model parameter memory breakdown

Usage:
    python examples/amp_training_demo.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.optim import get_optimizer
from nanomind.amp import (
    AMPConfig, AMPTrainer, GradAccumulator,
    mixed_precision_context, estimate_activation_memory,
    model_parameter_memory_mb,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog. " * 50
tokenizer = CharTokenizer().build(CORPUS)
ids       = torch.tensor(tokenizer.encode(CORPUS))
BLOCK     = 32

xs = torch.stack([ids[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([ids[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=16, shuffle=True, drop_last=True)

device    = torch.device("cpu")
model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK,
                        d_model=64, n_layers=4, n_heads=4, dropout=0.1)
model     = NanoMind(model_cfg).to(device)
optimizer = get_optimizer(model, lr=3e-4)

# ── Parameter memory ──────────────────────────────────────────────────────────
mem = model_parameter_memory_mb(model)
print(f"Model params : {mem['n_params']:,}")
print(f"Param memory : {mem['total_mb']:.3f} MB")

# ── Activation memory estimate ────────────────────────────────────────────────
act = estimate_activation_memory(batch=16, seq_len=BLOCK,
                                  d_model=64, n_layers=4)
print(f"
Activation memory (standard)     : {act['standard_mb']:.3f} MB")
print(f"Activation memory (checkpointed) : {act['checkpointed_mb']:.3f} MB")
print(f"Memory savings                   : {act['savings_ratio']:.1f}×")

# ── AMP training with gradient accumulation ───────────────────────────────────
amp_cfg = AMPConfig(
    enabled=True,
    dtype="bfloat16",
    grad_accum_steps=2,   # effective batch = 32
    clip_grad_norm=1.0,
)
trainer = AMPTrainer(model, optimizer, amp_cfg, device)

print(f"
Training with bfloat16 AMP, grad_accum_steps={amp_cfg.grad_accum_steps}")
for epoch in range(3):
    metrics = trainer.train_epoch(loader)
    print(f"  Epoch {epoch+1}: loss={metrics['loss']:.4f}, steps={metrics['steps']}")

print("
Done!")
