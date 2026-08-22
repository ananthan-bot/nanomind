"""
nanomind/cli/commands/info.py — CLI implementation of the info subcommand.
"""

from __future__ import annotations

import argparse

from nanomind.utils.logger import get_logger

log = get_logger("cli.info")


def run_info(args: argparse.Namespace) -> None:
    """
    Execute the 'nanomind info' command.

    Displays model architecture summary from a checkpoint.
    """
    from nanomind.checkpoint.info import print_checkpoint_info, checkpoint_info
    from nanomind.model import NanoMind, ModelConfig

    print_checkpoint_info(args.checkpoint)

    meta = checkpoint_info(args.checkpoint)
    model_cfg_dict = meta.get("model_config", {})
    if model_cfg_dict:
        model_cfg = ModelConfig(**model_cfg_dict)
        model = NanoMind(model_cfg)
        print(f"
Model: {model}")
        print(f"Trainable parameters: {model.num_parameters():,}")
        print(f"Head dim:             {model_cfg.head_dim}")
        print(f"Effective FFN dim:    {model_cfg.effective_d_ff}")
    else:
        log.warning("No model_config found in checkpoint metadata.")
