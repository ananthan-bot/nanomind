"""
train.py — Training script for MiniGPT

Usage:
    python train.py --data data.txt
    python train.py --data data.txt --max_iters 5000 --d_model 256 --n_layers 6

All arguments are optional; defaults are defined in config.py.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import get_dataloaders
from model import MiniGPT
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="Train MiniGPT on a text file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data / output ---
    parser.add_argument("--data", default="data.txt", help="Path to training text file")
    parser.add_argument("--out_dir", default="checkpoints", help="Checkpoint output directory")

    # --- Model ---
    parser.add_argument("--d_model", type=int, default=128, help="Model embedding dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--block_size", type=int, default=128, help="Context window size (tokens)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # --- Training ---
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate every N iters")
    parser.add_argument("--eval_iters", type=int, default=50, help="Number of batches per eval")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum LR (cosine decay)")
    parser.add_argument("--warmup_iters", type=int, default=100, help="LR warmup iterations")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iters")

    # --- Resume ---
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        data_path=args.data,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    return model_cfg, train_cfg, args.resume


# ---------------------------------------------------------------------------
# Learning Rate Schedule (cosine decay with linear warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    # Linear warmup
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    # After max_iters, return min_lr
    if step >= cfg.max_iters:
        return cfg.min_lr
    # Cosine decay
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    loaders: dict,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results = {}
    for split, loader in loaders.items():
        losses = []
        it = iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: MiniGPT, optimizer, step: int, val_loss: float):
    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_cfg": model.cfg,
        },
        path,
    )


def load_checkpoint(path: str, model: MiniGPT, optimizer) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_cfg, train_cfg, resume_path = parse_args()

    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.resolve_device())
    print(f"\n{'='*60}")
    print(f"  NanoMind Training")
    print(f"{'='*60}")
    print(f"  Device  : {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, tokenizer = get_dataloaders(
        data_path=train_cfg.data_path,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg.vocab_size = tokenizer.vocab_size
    model = MiniGPT(model_cfg).to(device)
    print(f"\n  {model}")
    print(f"  Parameters: {model.num_parameters():,}\n")

    # ── Optimizer ─────────────────────────────────────────────────────
    # Only apply weight decay to weight matrices (not biases / LayerNorm)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_cfg.learning_rate,
        betas=train_cfg.betas,
        fused=device.type == "cuda",  # fused kernel if on CUDA
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if resume_path:
        start_step, best_val_loss = load_checkpoint(resume_path, model, optimizer)
        print(f"  Resumed from '{resume_path}' at step {start_step}")

    # ── Output directory ──────────────────────────────────────────────
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "vocab.json")

    # ── Training Loop ─────────────────────────────────────────────────
    loaders = {"train": train_loader, "val": val_loader}
    train_iter = iter(train_loader)
    model.train()

    print(f"{'─'*60}")
    print(f"  {'Step':>6}  {'LR':>8}  {'Train Loss':>10}  {'Val Loss':>10}  {'Time':>6}")
    print(f"{'─'*60}")

    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, train_cfg.max_iters):
        # Fetch next batch (cycle through loader)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update LR
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()
        running_loss += loss.item()

        # ── Logging ───────────────────────────────────────────────────
        if (step + 1) % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            elapsed = time.time() - t0
            print(
                f"  {step+1:>6}  {lr:>8.2e}  {avg_loss:>10.4f}  {'':>10}  {elapsed:>5.1f}s",
                flush=True,
            )
            running_loss = 0.0
            t0 = time.time()

        # ── Evaluation ────────────────────────────────────────────────
        if (step + 1) % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, loaders, train_cfg.eval_iters, device)
            val_loss = losses["val"]
            print(
                f"\n  ► Eval @ step {step+1}: "
                f"train={losses['train']:.4f}  val={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = out_dir / "best.pt"
                save_checkpoint(ckpt_path, model, optimizer, step + 1, val_loss)
                print(f"  ✓ New best val loss {val_loss:.4f} — saved to '{ckpt_path}'")

            # Always save latest
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step + 1, val_loss)
            print(f"{'─'*60}")
            t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {out_dir.resolve()}")
    print(f"{'='*60}\n")
    print("  Generate text with:")
    print(f'  python generate.py --checkpoint {out_dir / "best.pt"} --prompt "Your prompt here"\n')


if __name__ == "__main__":
    main()
