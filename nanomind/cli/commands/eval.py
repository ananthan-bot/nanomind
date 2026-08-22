"""
nanomind/cli/commands/eval.py — CLI implementation of the eval subcommand.
"""

from __future__ import annotations

import argparse

from nanomind.utils.logger import get_logger
from nanomind.cli.device import resolve_device

log = get_logger("cli.eval")


def run_eval(args: argparse.Namespace) -> None:
    """
    Execute the 'nanomind eval' command.

    Loads model, tokenizes eval data, runs Evaluator, and prints results.
    """
    import torch
    from torch.utils.data import DataLoader
    from nanomind.checkpoint.io import load_checkpoint
    from nanomind.checkpoint.info import checkpoint_info
    from nanomind.model import NanoMind, ModelConfig
    from nanomind.eval import Evaluator, EvalConfig, print_comparison

    device = resolve_device(args.device)

    # Load model
    meta = checkpoint_info(args.checkpoint)
    model_cfg_dict = meta.get("model_config", {})
    model_cfg = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()
    model = NanoMind(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    eval_cfg = EvalConfig(
        max_batches=args.max_batches,
        top_k=args.top_k,
    )
    evaluator = Evaluator(model, eval_cfg, device)

    log.info("Reading eval data...")
    text = open(args.data, encoding="utf-8").read()

    # Build a simple char-level dataset
    from nanomind.tokenizer.char import CharTokenizer
    tokenizer = CharTokenizer().build(text)
    ids = tokenizer.encode(text)

    import torch
    block_size = model_cfg.block_size
    xs = torch.stack([torch.tensor(ids[i:i+block_size])   for i in range(0, len(ids)-block_size-1, block_size)])
    ys = torch.stack([torch.tensor(ids[i+1:i+block_size+1]) for i in range(0, len(ids)-block_size-1, block_size)])
    from torch.utils.data import TensorDataset
    loader = DataLoader(TensorDataset(xs, ys), batch_size=args.batch_size, drop_last=True)

    result = evaluator.full_eval(loader)
    print(f"
{result}
")
