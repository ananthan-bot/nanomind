"""
nanomind/cli/args.py — Argument parser for the NanoMind CLI.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level NanoMind argument parser with subcommands.

    Subcommands:
        train    — train a NanoMind model
        generate — generate text from a trained model
        eval     — evaluate a trained model
        info     — display model architecture summary

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="nanomind",
        description="NanoMind — A GPT-style language model trainer and generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nanomind train --config configs/small.yaml
  nanomind generate --checkpoint checkpoints/best.pt --prompt "Hello"
  nanomind eval --checkpoint checkpoints/best.pt --data data/val.txt
  nanomind info --checkpoint checkpoints/best.pt
        """,
    )
    parser.add_argument(
        "--version", action="version", version="NanoMind v0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _add_train_parser(subparsers)
    _add_generate_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_info_parser(subparsers)

    return parser


def _add_train_parser(subparsers) -> None:
    """Add the 'train' subcommand."""
    p = subparsers.add_parser("train", help="Train a NanoMind model.")
    p.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to JSON or YAML config file."
    )
    p.add_argument("--data",    type=str, help="Path to training text file.")
    p.add_argument("--out-dir", type=str, help="Output directory for checkpoints.")
    p.add_argument("--run-name",type=str, help="Experiment name.")
    # Model overrides
    p.add_argument("--d-model", type=int, help="Embedding dimension.")
    p.add_argument("--n-layers",type=int, help="Number of transformer layers.")
    p.add_argument("--n-heads", type=int, help="Number of attention heads.")
    p.add_argument("--block-size", type=int, help="Context window length.")
    # Training overrides
    p.add_argument("--max-iters",  type=int,   help="Maximum training steps.")
    p.add_argument("--lr",         type=float, help="Peak learning rate.")
    p.add_argument("--batch-size", type=int,   help="Batch size.")
    p.add_argument("--dropout",    type=float, help="Dropout probability.")
    p.add_argument("--seed",       type=int,   help="Random seed.")
    p.add_argument("--device",     type=str,   help="Device: auto, cpu, cuda, mps.")


def _add_generate_parser(subparsers) -> None:
    """Add the 'generate' subcommand."""
    p = subparsers.add_parser("generate", help="Generate text from a trained model.")
    p.add_argument("--checkpoint", "-c", type=str, required=True,
                   help="Path to model checkpoint (.pt).")
    p.add_argument("--prompt",  "-p", type=str, default="",
                   help="Seed text prompt.")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Number of tokens to generate.")
    p.add_argument("--strategy", type=str, default="temperature",
                   choices=["greedy", "temperature", "top_k", "top_p", "beam"],
                   help="Sampling strategy.")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature.")
    p.add_argument("--top-k",  type=int,   default=50,  help="Top-K filter.")
    p.add_argument("--top-p",  type=float, default=0.0, help="Top-P (nucleus) filter.")
    p.add_argument("--num-beams", type=int, default=1,  help="Number of beams.")
    p.add_argument("--seed",   type=int,   default=None, help="Random seed.")
    p.add_argument("--stream", action="store_true",
                   help="Stream output token by token.")
    p.add_argument("--device", type=str, default="auto")
