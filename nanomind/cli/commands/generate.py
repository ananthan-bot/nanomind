"""
nanomind/cli/commands/generate.py — CLI implementation of the generate subcommand.
"""

from __future__ import annotations

import argparse

from nanomind.utils.logger import get_logger
from nanomind.cli.device import resolve_device
from nanomind.generate.config import GenerationConfig

log = get_logger("cli.generate")


def run_generate(args: argparse.Namespace) -> None:
    """
    Execute the 'nanomind generate' command.

    Loads model weights from checkpoint, builds generator, and
    generates text from the given prompt.
    """
    import torch
    from nanomind.checkpoint.io import load_checkpoint
    from nanomind.checkpoint.info import checkpoint_info
    from nanomind.model import NanoMind, ModelConfig
    from nanomind.tokenizer.char import CharTokenizer
    from nanomind.generate import Generator, GenerationConfig

    device = resolve_device(args.device)

    # Load checkpoint metadata
    meta = checkpoint_info(args.checkpoint)
    model_cfg_dict = meta.get("model_config", {})
    model_cfg = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()

    # Build and load model
    model = NanoMind(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Build tokenizer (char-level default)
    tokenizer = CharTokenizer()
    prompt = args.prompt or ""

    # Build generation config
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        seed=args.seed,
    )

    generator = Generator(model, tokenizer, device=device)
    log.info(f"Generating {args.max_new_tokens} tokens with strategy='{args.strategy}'")

    if args.stream:
        print(prompt, end="", flush=True)
        for tok in generator.stream(prompt, gen_cfg):
            print(tok, end="", flush=True)
        print()
    else:
        output = generator.generate(prompt, gen_cfg)
        print(prompt + output)
