"""
examples/train_tiny.py — Minimal end-to-end NanoMind training example.

Trains a tiny NanoMind model on a short text string to verify the whole
stack works correctly: tokenizer -> dataset -> model -> trainer -> checkpoint.

Usage:
    python examples/train_tiny.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.trainer import Trainer, TrainConfig
from nanomind.optim import get_optimizer, get_lr_scheduler
from nanomind.checkpoint import CheckpointManager, CheckpointConfig

# ── 1. Data ───────────────────────────────────────────────────────────────────
TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 50
)

tokenizer = CharTokenizer().build(TEXT)
ids       = tokenizer.encode(TEXT)
BLOCK     = 32

tokens = torch.tensor(ids)
xs = torch.stack([tokens[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])

dataset    = TensorDataset(xs, ys)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader   = DataLoader(dataset, batch_size=16, drop_last=True)

# ── 2. Model ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK,
    d_model=64,
    n_layers=2,
    n_heads=2,
    dropout=0.1,
)
model = NanoMind(model_cfg).to(device)
print(f"Model: {model}")
print(f"Parameters: {model.num_parameters():,}")

# ── 3. Optimizer & Schedule ───────────────────────────────────────────────────
optimizer = get_optimizer(model, lr=3e-4, weight_decay=0.1)
schedule  = get_lr_scheduler(
    "warmup_cosine", max_lr=3e-4, min_lr=3e-5,
    warmup_steps=50, total_steps=500,
)

# ── 4. Trainer ────────────────────────────────────────────────────────────────
train_cfg = TrainConfig(
    max_iters=500,
    eval_interval=100,
    log_interval=50,
    grad_clip=1.0,
)
trainer = Trainer(model, optimizer, train_loader, val_loader, train_cfg, device)
result  = trainer.train(lr_scheduler=schedule)
print(f"\nTraining done! Best val loss: {result['best_val']:.4f}")

# ── 5. Checkpoint ─────────────────────────────────────────────────────────────
ckpt_cfg = CheckpointConfig(out_dir="checkpoints/tiny", keep_last_n=1)
mgr = CheckpointManager(ckpt_cfg)
mgr.save(model, step=500, val_loss=result["best_val"],
         model_config=model_cfg.to_dict())
print("Checkpoint saved!")

# ── 6. Generate ───────────────────────────────────────────────────────────────
from nanomind.generate import Generator, GenerationConfig

gen_cfg = GenerationConfig(max_new_tokens=80, strategy="top_k", top_k=10, temperature=0.8)
generator = Generator(model, tokenizer, device=device)
print("\nGenerated text:")
print(">>> " + generator.generate("The ", gen_cfg))
