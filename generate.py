"""
generate.py — Text generation CLI for MiniGPT

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "To be or not"
    python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --tokens 500 --temp 0.8 --top_k 40
"""

import argparse
import sys
import torch
from pathlib import Path

from model import MiniGPT
from tokenizer import CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained MiniGPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint file (e.g. checkpoints/best.pt)"
    )
    parser.add_argument(
        "--prompt", default="",
        help="Seed text to start generation from"
    )
    parser.add_argument(
        "--tokens", type=int, default=300,
        help="Number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", "--temp", type=float, default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Top-k sampling (0 = disabled, sample from full distribution)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto | cpu | cuda | mps"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of independent samples to generate"
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # ── Load checkpoint ────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint '{ckpt_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from '{ckpt_path}' ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_cfg = ckpt["model_cfg"]
    model = MiniGPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded: {model}")

    # ── Load tokenizer ────────────────────────────────────────────────
    vocab_path = ckpt_path.parent / "vocab.json"
    if not vocab_path.exists():
        print(f"Error: vocab file not found at '{vocab_path}'.", file=sys.stderr)
        sys.exit(1)

    tokenizer = CharTokenizer.load(vocab_path)

    # ── Generate ──────────────────────────────────────────────────────
    top_k = args.top_k if args.top_k > 0 else None
    prompt = args.prompt

    print(f"\nPrompt: {repr(prompt)}")
    print(f"Temperature: {args.temperature}  |  Top-k: {top_k}  |  Tokens: {args.tokens}")
    print("─" * 60)

    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n── Sample {sample_idx + 1} ──")

        # Encode prompt
        if prompt:
            ids = tokenizer.encode(prompt)
        else:
            # Start from a random character if no prompt given
            import random
            ids = [random.randint(2, tokenizer.vocab_size - 1)]

        idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)

        # Generate
        with torch.no_grad():
            output = model.generate(
                idx,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=top_k,
            )

        generated_ids = output[0].tolist()
        text = tokenizer.decode(generated_ids)
        print(text)

    print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
