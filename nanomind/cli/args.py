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
