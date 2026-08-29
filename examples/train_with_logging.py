"""
examples/train_with_logging.py — Full training run with TensorBoard and W&B logging.

Shows how to use TrainingLogger alongside the Trainer to log all metrics
to the console (always) and optionally TensorBoard / Weights & Biases.

Usage:
    # Console only (no extra deps):
    python examples/train_with_logging.py

    # Console + TensorBoard:
    python examples/train_with_logging.py --backend tensorboard
    tensorboard --logdir logs/

    # Console + W&B:
    pip install wandb && wandb login
    python examples/train_with_logging.py --backend wandb
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.trainer import Trainer, TrainConfig
from nanomind.optim import get_optimizer, get_lr_scheduler
from nanomind.logging import LogConfig, TrainingLogger

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="console",
                    choices=["console", "tensorboard", "wandb"])
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
TEXT      = "the quick brown fox jumps over the lazy dog " * 50
tokenizer = CharTokenizer().build(TEXT)
ids       = tokenizer.encode(TEXT)
BLOCK     = 32
tokens    = torch.tensor(ids)
xs = torch.stack([tokens[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=16, shuffle=True, drop_last=True)

# ── Model ─────────────────────────────────────────────────────────────────────
device    = torch.device("cpu")
model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK,
                        d_model=64, n_layers=2, n_heads=4, dropout=0.1)
model     = NanoMind(model_cfg).to(device)
optimizer = get_optimizer(model, lr=3e-4)
schedule  = get_lr_scheduler("warmup_cosine", max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=50, total_steps=500)

# ── Logger setup ──────────────────────────────────────────────────────────────
log_cfg = LogConfig(
    backend=[args.backend],
    log_dir="logs",
    project="nanomind-demo",
    run_name=f"tiny_{args.backend}",
    log_interval=50,
    log_grad_norm=True,
)

with TrainingLogger(log_cfg) as logger:
    logger.log_config({
        "d_model":    model_cfg.d_model,
        "n_layers":   model_cfg.n_layers,
        "n_heads":    model_cfg.n_heads,
        "vocab_size": model_cfg.vocab_size,
        "lr":         3e-4,
        "backend":    args.backend,
    })

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(loader)
    for step in range(1, 501):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        lr = schedule(step)

        logger.log_step(step, {
            "train/loss": loss.item(),
            "lr":          lr,
            "grad_norm":   grad_norm,
        })

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x, y   = next(iter(loader))
        _, val_loss = model(x, y)
    logger.log_validation(500, {"loss": val_loss.item()})

print("\nDone! Check your logs.")
