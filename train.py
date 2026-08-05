"""
train.py - NanoMind training script.

Usage:
    python train.py --data data.txt
    python train.py --data data.txt --max_iters 5000 --d_model 256 --n_layers 6
"""

import argparse
import math
import time
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import get_dataloaders
from model import NanoMind


def parse_args():
    p = argparse.ArgumentParser(description="Train NanoMind",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data.txt")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    if step >= cfg.max_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, eval_iters, device):
    model.eval()
    results = {}
    for split, loader in loaders.items():
        losses, it = [], iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(it)
            except StopIteration:
                break
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return results


def save_checkpoint(path, model, optimizer, step, val_loss):
    torch.save({
        "step": step, "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_cfg": model.cfg,
    }, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    print(f"\n{'='*60}")
    print(f"  NanoMind Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    train_loader, val_loader, tokenizer = get_dataloaders(
        args.data, args.block_size, args.batch_size)

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        data_path=args.data, batch_size=args.batch_size,
        max_iters=args.max_iters, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, learning_rate=args.learning_rate,
        min_lr=args.min_lr, warmup_iters=args.warmup_iters,
        grad_clip=args.grad_clip, weight_decay=args.weight_decay,
        device=args.device, seed=args.seed, log_interval=args.log_interval,
    )

    model = NanoMind(model_cfg).to(device)
    print(f"\n  {model}\n")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=train_cfg.learning_rate, betas=train_cfg.betas,
    )

    start_step, best_val = 0, float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed from step {start_step}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "vocab.json")

    loaders = {"train": train_loader, "val": val_loader}
    train_iter = iter(train_loader)
    model.train()

    print(f"{'─'*60}")
    print(f"  {'Step':>6}  {'LR':>8}  {'Loss':>10}")
    print(f"{'─'*60}")

    t0, running = time.time(), 0.0
    for step in range(start_step, train_cfg.max_iters):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        running += loss.item()

        if (step + 1) % train_cfg.log_interval == 0:
            print(f"  {step+1:>6}  {lr:.2e}  {running/train_cfg.log_interval:.4f}  ({time.time()-t0:.1f}s)", flush=True)
            running, t0 = 0.0, time.time()

        if (step + 1) % train_cfg.eval_interval == 0:
            L = estimate_loss(model, loaders, train_cfg.eval_iters, device)
            print(f"\n  Eval @ {step+1}: train={L['train']:.4f}  val={L['val']:.4f}")
            if L["val"] < best_val:
                best_val = L["val"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, step + 1, best_val)
                print(f"  Best val {best_val:.4f} saved")
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step + 1, L["val"])
            print(f"{'─'*60}")
            t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Done! Best val loss: {best_val:.4f}")
    print(f"  Generate: python generate.py --checkpoint {out_dir}/best.pt --prompt \'ROMEO:\'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
