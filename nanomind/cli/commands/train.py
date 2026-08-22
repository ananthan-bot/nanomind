"""
nanomind/cli/commands/train.py — CLI implementation of the train subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanomind.utils.logger import get_logger
from nanomind.utils.seed import set_seed
from nanomind.config import NanoMindConfig
from nanomind.cli.config_io import load_config, merge_cli_overrides, save_config
from nanomind.cli.device import resolve_device

log = get_logger("cli.train")


def run_train(args: argparse.Namespace) -> None:
    """
    Execute the 'nanomind train' command.

    Loads config (file + CLI overrides), builds model, data, optimizer,
    scheduler, and trainer, then runs the training loop.
    """
    # 1. Load base config
    cfg = load_config(args.config) if args.config else NanoMindConfig()

    # 2. Apply CLI overrides
    overrides = {
        "model.d_model":     getattr(args, "d_model",    None),
        "model.n_layers":    getattr(args, "n_layers",   None),
        "model.n_heads":     getattr(args, "n_heads",    None),
        "model.block_size":  getattr(args, "block_size", None),
        "model.dropout":     getattr(args, "dropout",    None),
        "train.max_iters":   getattr(args, "max_iters",  None),
        "train.seed":        getattr(args, "seed",       None),
        "run_name":          getattr(args, "run_name",   None),
    }
    merge_cli_overrides(cfg, overrides)
    if getattr(args, "out_dir", None):
        cfg.checkpoint.out_dir = args.out_dir

    device = resolve_device(getattr(args, "device", "auto"))

    log.info(f"Training config: {cfg}")
    log.info(f"Device: {device}")

    # 3. Set seed
    set_seed(cfg.train.seed)

    # 4. Save resolved config
    out_dir = Path(cfg.checkpoint.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.json")
    log.info(f"Config saved to {out_dir / 'config.json'}")

    # 5. Training (deferred import to avoid circular deps)
    from nanomind.model import NanoMind
    model = NanoMind(cfg.model).to(device)
    log.info(repr(model))
    log.info(f"Parameters: {model.num_parameters():,}")
    log.info("Ready to train — wire up your DataLoader and call trainer.train().")
