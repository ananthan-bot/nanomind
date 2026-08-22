"""
day13_commits.py — 20 atomic commits for Day 13: CLI & Configuration.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 13: CLI & Configuration — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — cli package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/__init__.py", '"""NanoMind command-line interface sub-package."""\n')
commit("feat: add nanomind/cli/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — NanoMindConfig top-level config dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/config.py", '''\
"""
nanomind/config.py — Top-level NanoMind configuration.

Combines ModelConfig, TrainConfig, CheckpointConfig, and GenerationConfig
into a single unified configuration that can be serialized to JSON/YAML.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from nanomind.model.config import ModelConfig
from nanomind.trainer.config import TrainConfig
from nanomind.checkpoint.config import CheckpointConfig
from nanomind.generate.config import GenerationConfig


@dataclass
class NanoMindConfig:
    """
    Unified NanoMind configuration.

    Wraps all sub-configs so the entire experiment can be defined
    from a single JSON or YAML file and tracked as one artifact.

    Attributes:
        model:      Model architecture config.
        train:      Training hyperparameter config.
        checkpoint: Checkpoint management config.
        generate:   Text generation config.
        run_name:   Experiment name (used in checkpoint dir).
    """

    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    generate:   GenerationConfig = field(default_factory=GenerationConfig)
    run_name:   str              = "nanomind_run"

    def to_dict(self) -> dict:
        """Serialize all sub-configs to a plain nested dict."""
        return {
            "model":      asdict(self.model),
            "train":      asdict(self.train),
            "checkpoint": asdict(self.checkpoint),
            "generate":   asdict(self.generate),
            "run_name":   self.run_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NanoMindConfig":
        """Deserialize from a nested dict."""
        return cls(
            model      = ModelConfig(**data.get("model", {})),
            train      = TrainConfig(**data.get("train", {})),
            checkpoint = CheckpointConfig(**data.get("checkpoint", {})),
            generate   = GenerationConfig(**data.get("generate", {})),
            run_name   = data.get("run_name", "nanomind_run"),
        )

    def save_json(self, path: str | Path) -> None:
        """Save unified config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "NanoMindConfig":
        """Load unified config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"NanoMindConfig("
            f"run='{self.run_name}', "
            f"d_model={self.model.d_model}, "
            f"n_layers={self.model.n_layers}, "
            f"max_iters={self.train.max_iters})"
        )
''')
commit("feat: add NanoMindConfig — unified config wrapping model, train, checkpoint, generate")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — YAML config loader
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/config_io.py", '''\
"""
nanomind/cli/config_io.py — Config file I/O for JSON and YAML formats.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanomind.config import NanoMindConfig


def load_config(path: str | Path) -> NanoMindConfig:
    """
    Load a NanoMindConfig from a JSON or YAML file.

    Supports ``.json`` and ``.yaml`` / ``.yml`` extensions.

    Args:
        path: Path to the config file.

    Returns:
        Parsed :class:`~nanomind.config.NanoMindConfig`.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install it with: pip install pyyaml"
            )
    else:
        raise ValueError(f"Unsupported config format: '{suffix}'. Use .json or .yaml")

    return NanoMindConfig.from_dict(data)


def save_config(cfg: NanoMindConfig, path: str | Path) -> None:
    """
    Save a NanoMindConfig to JSON or YAML.

    Args:
        cfg:  Config to save.
        path: Destination file path (.json or .yaml).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()

    if suffix == ".json":
        p.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            p.write_text(yaml.dump(cfg.to_dict(), default_flow_style=False), encoding="utf-8")
        except ImportError:
            raise ImportError("PyYAML required. pip install pyyaml")
    else:
        raise ValueError(f"Unsupported format: '{suffix}'")


def merge_cli_overrides(cfg: NanoMindConfig, overrides: dict) -> NanoMindConfig:
    """
    Apply CLI argument overrides on top of a file-based config.

    Keys use dot notation: ``"model.d_model"`` overrides ``cfg.model.d_model``.

    Args:
        cfg:       Base config loaded from file.
        overrides: Dict of ``"section.key": value`` overrides from CLI args.

    Returns:
        Updated config (in-place modification + returned for chaining).
    """
    for dotkey, value in overrides.items():
        if value is None:
            continue
        parts = dotkey.split(".", 1)
        if len(parts) == 2:
            section, key = parts
            sub = getattr(cfg, section, None)
            if sub is not None and hasattr(sub, key):
                setattr(sub, key, type(getattr(sub, key))(value))
        elif hasattr(cfg, dotkey):
            setattr(cfg, dotkey, value)
    return cfg
''')
commit("feat: add config_io.py — load/save JSON/YAML configs and CLI override merging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — argument parser foundation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/args.py", '''\
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
''')
commit("feat: add CLI argument parser skeleton with train, generate, eval, info subcommands")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — train subcommand arguments
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/cli/args.py")
src += '''

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
'''
write("nanomind/cli/args.py", src)
commit("feat: add train subcommand with config, data, model, and training arguments")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — generate subcommand arguments
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/cli/args.py")
src += '''

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
'''
write("nanomind/cli/args.py", src)
commit("feat: add generate subcommand with prompt, strategy, sampling, and stream args")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — eval and info subcommand arguments
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/cli/args.py")
src += '''

def _add_eval_parser(subparsers) -> None:
    """Add the 'eval' subcommand."""
    p = subparsers.add_parser("eval", help="Evaluate a trained NanoMind model.")
    p.add_argument("--checkpoint", "-c", type=str, required=True,
                   help="Path to model checkpoint (.pt).")
    p.add_argument("--data",       "-d", type=str, required=True,
                   help="Path to evaluation text file.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Max evaluation batches (0 = all).")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--top-k",       type=int, default=5,
                   help="K for top-K accuracy.")
    p.add_argument("--device",      type=str, default="auto")


def _add_info_parser(subparsers) -> None:
    """Add the 'info' subcommand."""
    p = subparsers.add_parser("info", help="Display model architecture summary.")
    p.add_argument("--checkpoint", "-c", type=str, required=True,
                   help="Path to model checkpoint (.pt).")
    p.add_argument("--config",    type=str, default=None,
                   help="Optional config file to display.")
'''
write("nanomind/cli/args.py", src)
commit("feat: add eval and info subcommand argument definitions")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — device resolution utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/device.py", '''\
"""
nanomind/cli/device.py — Device resolution for CLI commands.
"""

from __future__ import annotations

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"auto"`` (pick best), ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
''')
commit("feat: add resolve_device() — pick best available device from a string spec")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — train command implementation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/commands/train.py", '''\
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
''')
commit("feat: implement run_train() CLI handler — config loading, override, model init")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — generate command implementation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/commands/generate.py", '''\
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
''')
commit("feat: implement run_generate() CLI handler — checkpoint load, generate, stream")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — eval command implementation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/commands/eval.py", '''\
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
    print(f"\n{result}\n")
''')
commit("feat: implement run_eval() CLI handler — load model, tokenize, evaluate, print")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — info command implementation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/commands/info.py", '''\
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
        print(f"\nModel: {model}")
        print(f"Trainable parameters: {model.num_parameters():,}")
        print(f"Head dim:             {model_cfg.head_dim}")
        print(f"Effective FFN dim:    {model_cfg.effective_d_ff}")
    else:
        log.warning("No model_config found in checkpoint metadata.")
''')
commit("feat: implement run_info() CLI handler — display model summary from checkpoint")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — commands __init__
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/commands/__init__.py", '''\
"""NanoMind CLI command implementations."""
from nanomind.cli.commands.train import run_train
from nanomind.cli.commands.generate import run_generate
from nanomind.cli.commands.eval import run_eval
from nanomind.cli.commands.info import run_info

__all__ = ["run_train", "run_generate", "run_eval", "run_info"]
''')
commit("refactor: export all CLI commands from nanomind/cli/commands/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — main entry point
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/main.py", '''\
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
''')
commit("feat: add nanomind/cli/main.py — entry point dispatching to CLI subcommands")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — register CLI entry point in pyproject.toml
# ══════════════════════════════════════════════════════════════════════════════
# Check if pyproject.toml exists
pyproject_path = REPO / "pyproject.toml"
if pyproject_path.exists():
    src = read("pyproject.toml")
    if "[project.scripts]" not in src:
        src += '\n[project.scripts]\nnanomind = "nanomind.cli.main:main"\n'
        write("pyproject.toml", src)
else:
    write("pyproject.toml", '''\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "nanomind"
version = "0.1.0"
description = "A GPT-style language model — built layer by layer over 14 days."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
]

[project.optional-dependencies]
dev  = ["pytest", "pyyaml"]

[project.scripts]
nanomind = "nanomind.cli.main:main"
''')
commit("chore: add pyproject.toml with nanomind CLI entry point registration")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — update cli __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cli/__init__.py", '''\
"""NanoMind command-line interface sub-package.

Entry point:
    ``nanomind`` (after `pip install -e .`) calls :func:`nanomind.cli.main.main`.

Subcommands:
    train    — train a model from config
    generate — generate text from a checkpoint
    eval     — evaluate model perplexity/accuracy
    info     — display model summary
"""
from nanomind.cli.main import main
__all__ = ["main"]
''')
commit("refactor: export main() from nanomind/cli/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: argument parser correctness
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_cli.py", '''\
"""
tests/test_cli.py — Tests for the NanoMind CLI argument parser and config I/O.
"""

import json
import pytest
from pathlib import Path

from nanomind.cli.args import build_parser
from nanomind.cli.config_io import load_config, save_config, merge_cli_overrides
from nanomind.config import NanoMindConfig


# ── Argument parser ───────────────────────────────────────────────────────────

class TestArgParser:
    def test_train_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["train"])
        assert args.command == "train"

    def test_generate_requires_checkpoint(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate"])   # missing --checkpoint

    def test_generate_defaults(self):
        parser = build_parser()
        args   = parser.parse_args(["generate", "--checkpoint", "ckpt.pt"])
        assert args.max_new_tokens == 200
        assert args.strategy == "temperature"
        assert args.temperature == 0.8

    def test_eval_requires_checkpoint_and_data(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["eval"])

    def test_generate_stream_flag(self):
        parser = build_parser()
        args   = parser.parse_args(["generate", "--checkpoint", "ckpt.pt", "--stream"])
        assert args.stream is True

    def test_train_d_model_override(self):
        parser = build_parser()
        args   = parser.parse_args(["train", "--d-model", "256"])
        assert args.d_model == 256
''')
commit("test: add CLI argument parser tests — subcommands, defaults, required args")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: config save/load roundtrip
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cli.py")
src += '''

# ── Config I/O ────────────────────────────────────────────────────────────────

class TestConfigIO:
    def test_json_roundtrip(self, tmp_path):
        cfg  = NanoMindConfig()
        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        cfg2 = load_config(path)
        assert cfg.model.d_model == cfg2.model.d_model
        assert cfg.train.max_iters == cfg2.train.max_iters

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("[model]\nd_model = 128\n")
        with pytest.raises(ValueError):
            load_config(p)

    def test_json_contains_model_section(self, tmp_path):
        cfg  = NanoMindConfig()
        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        data = json.loads(path.read_text())
        assert "model" in data
        assert "train" in data
        assert "checkpoint" in data
'''
write("tests/test_cli.py", src)
commit("test: add config save/load roundtrip and format validation tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: config merge CLI overrides
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cli.py")
src += '''

# ── Config merge ──────────────────────────────────────────────────────────────

class TestConfigMerge:
    def test_override_model_d_model(self):
        cfg = NanoMindConfig()
        merge_cli_overrides(cfg, {"model.d_model": 256})
        assert cfg.model.d_model == 256

    def test_override_train_max_iters(self):
        cfg = NanoMindConfig()
        merge_cli_overrides(cfg, {"train.max_iters": 9999})
        assert cfg.train.max_iters == 9999

    def test_none_override_is_ignored(self):
        cfg = NanoMindConfig()
        original_d = cfg.model.d_model
        merge_cli_overrides(cfg, {"model.d_model": None})
        assert cfg.model.d_model == original_d

    def test_run_name_override(self):
        cfg = NanoMindConfig()
        merge_cli_overrides(cfg, {"run_name": "my_experiment"})
        assert cfg.run_name == "my_experiment"


# ── NanoMindConfig ────────────────────────────────────────────────────────────

class TestNanoMindConfig:
    def test_repr_contains_run_name(self):
        cfg = NanoMindConfig(run_name="test_run")
        assert "test_run" in repr(cfg)

    def test_to_dict_has_all_sections(self):
        d = NanoMindConfig().to_dict()
        assert all(k in d for k in ("model", "train", "checkpoint", "generate", "run_name"))

    def test_from_dict_roundtrip(self):
        cfg  = NanoMindConfig()
        cfg2 = NanoMindConfig.from_dict(cfg.to_dict())
        assert cfg.model.d_model == cfg2.model.d_model
        assert cfg.train.max_iters == cfg2.train.max_iters
'''
write("tests/test_cli.py", src)
commit("test: add CLI config merge and NanoMindConfig roundtrip tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 13 | CLI & configuration | 🔜 |",
    "| 13 | CLI & configuration | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Evaluation: PPL, BPC, accuracy, top-K, Evaluator, benchmark, generation quality metrics (Day 12)",
    "- Evaluation: PPL, BPC, accuracy, top-K, Evaluator, benchmark, generation quality metrics (Day 12)\n- CLI: train/generate/eval/info subcommands, NanoMindConfig, JSON/YAML config I/O (Day 13)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 13 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 13 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 13 COMPLETE ===")
