"""
nanomind/serve/cli.py — CLI entry point: ``nanomind serve``.

Usage::

    python -m nanomind.serve.cli --port 8080 --model-name MyModel
    nanomind serve --port 8080   # if installed as package

Starts a NanoMind server with a randomly initialised model for demo purposes.
In production, load a checkpoint with ``--checkpoint path/to/model.pt``.
"""

from __future__ import annotations

import argparse
import sys
import torch

from nanomind.serve.config import ServeConfig
from nanomind.serve.server import ModelServer
from nanomind.model.config import ModelConfig
from nanomind.tokenizer.char import CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanomind serve",
        description="Start the NanoMind inference HTTP server",
    )
    p.add_argument("--host",        default="127.0.0.1", help="Server host")
    p.add_argument("--port",        type=int, default=8080, help="Server port")
    p.add_argument("--max-tokens",  type=int, default=256,  help="Max tokens per request")
    p.add_argument("--model-name",  default="NanoMind",     help="Model display name")
    p.add_argument("--log-requests",action="store_true",    help="Log each request")
    p.add_argument("--device",      default="cpu",          help="Inference device")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Build a tiny demo model
    CORPUS    = "abcdefghijklmnopqrstuvwxyz " * 4
    tokenizer = CharTokenizer().build(CORPUS)
    from nanomind import NanoMind
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, block_size=64,
        d_model=64, n_layers=2, n_heads=4, dropout=0.0,
    )
    model = NanoMind(model_cfg)

    cfg = ServeConfig(
        host=args.host,
        port=args.port,
        max_new_tokens=args.max_tokens,
        model_name=args.model_name,
        log_requests=args.log_requests,
    )
    server = ModelServer(model, tokenizer, cfg, device=args.device)
    server.start()


if __name__ == "__main__":
    main()
