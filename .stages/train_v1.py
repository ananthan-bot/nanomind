"""
train.py - NanoMind basic training loop (LR schedule + checkpointing coming next)
"""

import argparse
import sys
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import get_dataloaders
from model import NanoMind
from tokenizer import CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoMind")
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)

    train_loader, val_loader, tokenizer = get_dataloaders(
        args.data, args.block_size, args.batch_size
    )

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
    )
    model = NanoMind(model_cfg).to(device)
    print(f"NanoMind | params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "vocab.json")

    train_iter = iter(train_loader)
    model.train()

    for step in range(args.max_iters):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"step {step+1:>5} | loss {loss.item():.4f}")

    torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg},
               out_dir / "latest.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
