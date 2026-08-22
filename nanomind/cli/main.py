"""
nanomind/cli/main.py — NanoMind CLI entry point.

Usage:
    nanomind train    --config configs/small.yaml
    nanomind generate --checkpoint checkpoints/best.pt --prompt "Hello"
    nanomind eval     --checkpoint checkpoints/best.pt --data data/val.txt
    nanomind info     --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import sys

from nanomind.cli.args import build_parser
from nanomind.cli.commands import run_train, run_generate, run_eval, run_info


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "train":    run_train,
        "generate": run_generate,
        "eval":     run_eval,
        "info":     run_info,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
