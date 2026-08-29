"""
day21_commits.py — 20 atomic commits for Day 21: Training Logging (TensorBoard / W&B).
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

print("\n=== DAY 21: Training Logging (TensorBoard / W&B) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — logging package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/__init__.py",
      '"""NanoMind training logging sub-package — TensorBoard, W&B, and console."""\n')
commit("feat: add nanomind/logging/ package skeleton for training experiment logging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — LogConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/config.py", '''\
"""
nanomind/logging/config.py — Logging configuration for training runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """
    Configuration for training experiment logging.

    Attributes:
        backend:       Logging backend(s). Options: ``"console"``, ``"tensorboard"``,
                       ``"wandb"``, or a list of multiple backends.
        log_dir:       Directory for TensorBoard event files and local logs.
        project:       W&B project name (used when backend includes ``"wandb"``).
        run_name:      Human-readable name for this training run.
        log_interval:  Log training metrics every N steps.
        log_grad_norm: Whether to log gradient norm at each log step.
        log_lr:        Whether to log learning rate at each log step.
        log_params:    Whether to log parameter histograms (expensive; TB only).
        tags:          Optional list of tags for W&B run organisation.
        notes:         Optional notes string for W&B.
    """

    backend:       str | list[str]  = "console"
    log_dir:       str              = "logs"
    project:       str              = "nanomind"
    run_name:      str              = "run"
    log_interval:  int              = 50
    log_grad_norm: bool             = True
    log_lr:        bool             = True
    log_params:    bool             = False
    tags:          list[str]        = field(default_factory=list)
    notes:         str              = ""

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = [self.backend]
        valid = {"console", "tensorboard", "wandb"}
        for b in self.backend:
            assert b in valid, f"Unknown backend '{b}'. Choose from {valid}."
        assert self.log_interval >= 1

    @property
    def backends(self) -> list[str]:
        return self.backend if isinstance(self.backend, list) else [self.backend]
''')
commit("feat: add LogConfig — backend, log_dir, project, run_name, log_interval, grad_norm")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — BaseLogger abstract class
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/base.py", '''\
"""
nanomind/logging/base.py — Abstract base class for all training loggers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Abstract base class for NanoMind training loggers.

    All logger backends (console, TensorBoard, W&B) implement this interface,
    allowing the :class:`TrainingLogger` to multiplex across multiple backends.
    """

    @abstractmethod
    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at a given step."""

    @abstractmethod
    def log_config(self, config: dict) -> None:
        """Log hyperparameter configuration at the start of training."""

    def log_histogram(self, name: str, values, step: int) -> None:
        """Log a histogram (optional — not all backends support it)."""

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text sample (optional — not all backends support it)."""

    def finish(self) -> None:
        """Called at end of training run to flush and close the logger."""
''')
commit("feat: add BaseLogger abstract class — log_scalars, log_config, log_histogram, finish")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — ConsoleLogger
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/console.py", '''\
"""
nanomind/logging/console.py — Rich terminal logger for training progress.
"""

from __future__ import annotations

import sys
from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

_log = get_logger("training")


class ConsoleLogger(BaseLogger):
    """
    Console (stdout) logger — always available, no external dependencies.

    Formats training metrics as a clean one-liner per log step::

        step  250 | loss 2.3451 | lr 3.00e-04 | grad_norm 1.23 | tok/s 12540

    Args:
        run_name:     Name displayed in the header.
        log_interval: Print frequency (steps).
    """

    def __init__(self, run_name: str = "run", log_interval: int = 50) -> None:
        self.run_name     = run_name
        self.log_interval = log_interval
        self._config      = {}

    def log_config(self, config: dict) -> None:
        print("=" * 60)
        print(f"  NanoMind Training Run: {self.run_name}")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20}: {v}")
        print("=" * 60)
        self._config = config

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        parts = [f"step {step:>6}"]
        for key, val in metrics.items():
            if isinstance(val, float):
                if "loss" in key or "ppl" in key:
                    parts.append(f"{key} {val:.4f}")
                elif "lr" in key:
                    parts.append(f"{key} {val:.2e}")
                elif "norm" in key:
                    parts.append(f"{key} {val:.3f}")
                elif "tok" in key or "rate" in key:
                    parts.append(f"{key} {val:.0f}")
                else:
                    parts.append(f"{key} {val:.4f}")
            else:
                parts.append(f"{key} {val}")
        print(" | ".join(parts), flush=True)

    def log_histogram(self, name: str, values, step: int) -> None:
        pass  # Console logger skips histograms

    def log_text(self, tag: str, text: str, step: int) -> None:
        print(f"[{step}] {tag}: {text[:120]}")

    def finish(self) -> None:
        print("=" * 60)
        print(f"  Training complete: {self.run_name}")
        print("=" * 60)
''')
commit("feat: add ConsoleLogger — formatted one-liner training metrics to stdout")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — TensorBoardLogger
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/tensorboard.py", '''\
"""
nanomind/logging/tensorboard.py — TensorBoard logger for NanoMind training.

Requires: ``pip install tensorboard``
Usage:    ``tensorboard --logdir logs/``
"""

from __future__ import annotations

from pathlib import Path
from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

log = get_logger("logging.tensorboard")


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard training logger.

    Logs scalars, optional parameter histograms, and text summaries.
    Falls back gracefully if ``tensorboard`` is not installed.

    Args:
        log_dir:     Directory for TensorBoard event files.
        run_name:    Sub-directory name for this run.
        log_params:  Log weight histograms every N steps (expensive).
    """

    def __init__(
        self,
        log_dir:    str = "logs",
        run_name:   str = "run",
        log_params: bool = False,
    ) -> None:
        self.log_dir    = Path(log_dir) / run_name
        self.log_params = log_params
        self._writer    = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer    = SummaryWriter(log_dir=str(self.log_dir))
            self._available = True
            log.info(f"TensorBoard logging to: {self.log_dir}")
        except ImportError:
            log.warning(
                "TensorBoard not installed. "
                "Install with: pip install tensorboard"
            )

    def log_config(self, config: dict) -> None:
        if not self._available:
            return
        # Log hyperparameters
        try:
            self._writer.add_hparams(
                hparam_dict={k: str(v) for k, v in config.items()},
                metric_dict={},
            )
        except Exception:
            pass  # add_hparams can fail with certain config types

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if not self._available:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self._writer.add_scalar(key, val, global_step=step)

    def log_histogram(self, name: str, values, step: int) -> None:
        if not self._available:
            return
        try:
            self._writer.add_histogram(name, values, global_step=step)
        except Exception:
            pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        if not self._available:
            return
        self._writer.add_text(tag, text, global_step=step)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            log.info("TensorBoard writer closed.")
''')
commit("feat: add TensorBoardLogger — scalars, histograms, hparams, text with graceful fallback")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — WandbLogger
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/wandb_logger.py", '''\
"""
nanomind/logging/wandb_logger.py — Weights & Biases logger for NanoMind training.

Requires: ``pip install wandb``
Usage:    ``wandb login`` then set ``backend: wandb`` in LogConfig.
"""

from __future__ import annotations

from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

log = get_logger("logging.wandb")


class WandbLogger(BaseLogger):
    """
    Weights & Biases training logger.

    Logs metrics, config, text samples, and optionally model gradients.
    Falls back gracefully if ``wandb`` is not installed or not authenticated.

    Args:
        project:   W&B project name.
        run_name:  Name for this run.
        tags:      List of tags for the W&B run.
        notes:     Notes string for the W&B run.
        config:    Initial config dict (can also be set via log_config).
    """

    def __init__(
        self,
        project:  str = "nanomind",
        run_name: str = "run",
        tags:     list[str] | None = None,
        notes:    str = "",
    ) -> None:
        self.project  = project
        self.run_name = run_name
        self.tags     = tags or []
        self.notes    = notes
        self._run     = None
        self._available = False
        self._pending_config: dict = {}

    def _ensure_init(self, config: dict | None = None) -> None:
        if self._run is not None:
            return
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                tags=self.tags,
                notes=self.notes,
                config=config or self._pending_config,
                reinit=True,
            )
            self._available = True
            log.info(f"W&B run started: {self._run.url}")
        except ImportError:
            log.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            log.warning(f"W&B init failed: {e}. Falling back to console only.")

    def log_config(self, config: dict) -> None:
        self._pending_config = config
        self._ensure_init(config)
        if self._available:
            try:
                import wandb
                wandb.config.update(config, allow_val_change=True)
            except Exception:
                pass

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass

    def log_histogram(self, name: str, values, step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
        except Exception:
            pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._available:
            try:
                import wandb
                wandb.finish()
                log.info("W&B run finished.")
            except Exception:
                pass
''')
commit("feat: add WandbLogger — W&B integration with graceful fallback if not installed")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — MetricsBuffer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/metrics.py", '''\
"""
nanomind/logging/metrics.py — Metrics accumulation and averaging buffer.
"""

from __future__ import annotations

from collections import defaultdict


class MetricsBuffer:
    """
    Accumulate scalar metrics over multiple steps and compute running averages.

    Useful for computing per-epoch or per-interval averages of training metrics
    without keeping the full history in memory.

    Example::

        buf = MetricsBuffer()
        for x, y in loader:
            loss = model(x, y)
            buf.update({"loss": loss.item()})
        print(buf.averages())   # {"loss": 2.345}
        buf.reset()
    """

    def __init__(self) -> None:
        self._sums:   dict[str, float] = defaultdict(float)
        self._counts: dict[str, int]   = defaultdict(int)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        """
        Add metric values to the buffer.

        Args:
            metrics: Dict of metric_name → value.
            n:       Number of samples these metrics are averaged over.
        """
        for k, v in metrics.items():
            self._sums[k]   += v * n
            self._counts[k] += n

    def averages(self) -> dict[str, float]:
        """
        Compute the running average for each accumulated metric.

        Returns:
            Dict of metric_name → average_value.
        """
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def __len__(self) -> int:
        return len(self._sums)

    def __contains__(self, key: str) -> bool:
        return key in self._sums
''')
commit("feat: add MetricsBuffer — accumulate and average scalar metrics over training steps")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — LoggerFactory
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/factory.py", '''\
"""
nanomind/logging/factory.py — Logger factory: build loggers from LogConfig.
"""

from __future__ import annotations

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.console import ConsoleLogger
from nanomind.logging.tensorboard import TensorBoardLogger
from nanomind.logging.wandb_logger import WandbLogger


def build_loggers(cfg: LogConfig) -> list[BaseLogger]:
    """
    Build a list of logger instances from a :class:`LogConfig`.

    Args:
        cfg: Logging configuration.

    Returns:
        List of instantiated logger backends.
    """
    loggers: list[BaseLogger] = []

    for backend in cfg.backends:
        if backend == "console":
            loggers.append(
                ConsoleLogger(run_name=cfg.run_name, log_interval=cfg.log_interval)
            )
        elif backend == "tensorboard":
            loggers.append(
                TensorBoardLogger(
                    log_dir=cfg.log_dir,
                    run_name=cfg.run_name,
                    log_params=cfg.log_params,
                )
            )
        elif backend == "wandb":
            loggers.append(
                WandbLogger(
                    project=cfg.project,
                    run_name=cfg.run_name,
                    tags=cfg.tags,
                    notes=cfg.notes,
                )
            )

    return loggers
''')
commit("feat: add build_loggers() factory — instantiate backends from LogConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — TrainingLogger: multiplexes across backends
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/training_logger.py", '''\
"""
nanomind/logging/training_logger.py — High-level training logger multiplexer.

TrainingLogger fans out all log calls to multiple backends simultaneously.
It also manages the MetricsBuffer for step-level accumulation and provides
convenience methods for training-specific events (epoch start, validation, etc.)
"""

from __future__ import annotations

import time
import torch.nn as nn

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.factory import build_loggers
from nanomind.logging.metrics import MetricsBuffer
from nanomind.utils.logger import get_logger


class TrainingLogger:
    """
    High-level training logger that multiplexes across multiple backends.

    Accepts a :class:`LogConfig` and fans out all log calls to every enabled
    backend (console, TensorBoard, W&B) simultaneously.

    Args:
        cfg: Logging configuration.

    Example::

        log_cfg = LogConfig(backend=["console", "tensorboard"], run_name="my_run")
        logger  = TrainingLogger(log_cfg)
        logger.log_config({"lr": 3e-4, "batch_size": 32})

        for step, (x, y) in enumerate(loader):
            loss = train_step(x, y)
            logger.log_step(step, {"train/loss": loss, "lr": get_lr()})

        logger.finish()
    """

    def __init__(self, cfg: LogConfig) -> None:
        self.cfg      = cfg
        self._loggers = build_loggers(cfg)
        self._buffer  = MetricsBuffer()
        self._t0      = time.perf_counter()
        self._log     = get_logger("training.logger")

    def log_config(self, config: dict) -> None:
        """Broadcast config to all backends at run start."""
        for lg in self._loggers:
            try:
                lg.log_config(config)
            except Exception as e:
                self._log.warning(f"log_config failed for {type(lg).__name__}: {e}")

    def log_step(
        self,
        step:    int,
        metrics: dict[str, float],
    ) -> None:
        """
        Accumulate step metrics; broadcast to backends every ``log_interval`` steps.

        Args:
            step:    Current training step.
            metrics: Scalar metrics dict (e.g. ``{"train/loss": 2.3, "lr": 3e-4}``).
        """
        self._buffer.update(metrics)

        if step > 0 and step % self.cfg.log_interval == 0:
            averaged = self._buffer.averages()
            averaged["step"] = step
            averaged["elapsed_s"] = time.perf_counter() - self._t0
            self._broadcast(averaged, step)
            self._buffer.reset()

    def log_validation(self, step: int, metrics: dict[str, float]) -> None:
        """Log validation metrics immediately (not buffered)."""
        prefixed = {f"val/{k}": v for k, v in metrics.items()}
        self._broadcast(prefixed, step)

    def log_histogram(self, name: str, values, step: int) -> None:
        """Log a parameter histogram to all backends that support it."""
        for lg in self._loggers:
            try:
                lg.log_histogram(name, values, step)
            except Exception:
                pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text sample to all backends."""
        for lg in self._loggers:
            try:
                lg.log_text(tag, text, step)
            except Exception:
                pass

    def log_model_params(self, model: nn.Module, step: int) -> None:
        """Log weight histograms for all named parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"params/{name}", param.data, step)

    def finish(self) -> None:
        """Flush and close all backends."""
        for lg in self._loggers:
            try:
                lg.finish()
            except Exception:
                pass

    def _broadcast(self, metrics: dict[str, float], step: int) -> None:
        for lg in self._loggers:
            try:
                lg.log_scalars(metrics, step)
            except Exception as e:
                self._log.warning(f"log_scalars failed for {type(lg).__name__}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
''')
commit("feat: add TrainingLogger — multiplex log_step, log_validation, log_histogram to all backends")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update logging __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/logging/__init__.py", '''\
"""NanoMind training logging sub-package.

Provides unified logging to console, TensorBoard, and Weights & Biases,
with a single high-level :class:`TrainingLogger` that multiplexes across
all enabled backends simultaneously.

Backends:
    - ``"console"``     — always available; rich formatted terminal output
    - ``"tensorboard"`` — requires ``pip install tensorboard``
    - ``"wandb"``       — requires ``pip install wandb`` + ``wandb login``

Primary exports:
    - :class:`TrainingLogger`   — high-level multiplex logger (recommended entry point)
    - :class:`LogConfig`        — backend, log_dir, project, run_name, log_interval
    - :class:`MetricsBuffer`    — accumulate and average metrics over steps
    - :class:`ConsoleLogger`    — formatted stdout logger
    - :class:`TensorBoardLogger`— TensorBoard backend
    - :class:`WandbLogger`      — Weights & Biases backend
    - :func:`build_loggers`     — instantiate backends from LogConfig
"""

from nanomind.logging.config import LogConfig
from nanomind.logging.base import BaseLogger
from nanomind.logging.console import ConsoleLogger
from nanomind.logging.tensorboard import TensorBoardLogger
from nanomind.logging.wandb_logger import WandbLogger
from nanomind.logging.metrics import MetricsBuffer
from nanomind.logging.factory import build_loggers
from nanomind.logging.training_logger import TrainingLogger

__all__ = [
    "LogConfig",
    "BaseLogger",
    "ConsoleLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "MetricsBuffer",
    "build_loggers",
    "TrainingLogger",
]
''')
commit("refactor: export all logging components from nanomind/logging/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — update Trainer to accept a TrainingLogger
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
# Add logger import and parameter
if "TrainingLogger" not in src:
    src = src.replace(
        "from nanomind.utils.logger import get_logger",
        "from nanomind.utils.logger import get_logger\nfrom nanomind.logging import TrainingLogger, LogConfig"
    )
    # Add training_logger param to __init__
    src = src.replace(
        "        device: torch.device | None = None,\n    ) -> None:",
        "        device: torch.device | None = None,\n"
        "        training_logger: \"TrainingLogger | None\" = None,\n    ) -> None:"
    )
    # Store it
    src = src.replace(
        "        self.log = get_logger(\"trainer\")",
        "        self.log = get_logger(\"trainer\")\n        self.training_logger = training_logger"
    )
write("nanomind/trainer/trainer.py", src)
commit("feat: update Trainer to accept an optional TrainingLogger for experiment tracking")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — example: train_with_logging.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/train_with_logging.py", '''\
"""
examples/train_with_logging.py — Full training run with TensorBoard and W&B logging.

Shows how to use TrainingLogger alongside the Trainer to log all metrics
to the console (always) and optionally TensorBoard / Weights & Biases.

Usage:
    # Console only (no extra deps):
    python examples/train_with_logging.py

    # Console + TensorBoard:
    python examples/train_with_logging.py --backend tensorboard
    tensorboard --logdir logs/

    # Console + W&B:
    pip install wandb && wandb login
    python examples/train_with_logging.py --backend wandb
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.trainer import Trainer, TrainConfig
from nanomind.optim import get_optimizer, get_lr_scheduler
from nanomind.logging import LogConfig, TrainingLogger

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="console",
                    choices=["console", "tensorboard", "wandb"])
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
TEXT      = "the quick brown fox jumps over the lazy dog " * 50
tokenizer = CharTokenizer().build(TEXT)
ids       = tokenizer.encode(TEXT)
BLOCK     = 32
tokens    = torch.tensor(ids)
xs = torch.stack([tokens[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=16, shuffle=True, drop_last=True)

# ── Model ─────────────────────────────────────────────────────────────────────
device    = torch.device("cpu")
model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK,
                        d_model=64, n_layers=2, n_heads=4, dropout=0.1)
model     = NanoMind(model_cfg).to(device)
optimizer = get_optimizer(model, lr=3e-4)
schedule  = get_lr_scheduler("warmup_cosine", max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=50, total_steps=500)

# ── Logger setup ──────────────────────────────────────────────────────────────
log_cfg = LogConfig(
    backend=[args.backend],
    log_dir="logs",
    project="nanomind-demo",
    run_name=f"tiny_{args.backend}",
    log_interval=50,
    log_grad_norm=True,
)

with TrainingLogger(log_cfg) as logger:
    logger.log_config({
        "d_model":    model_cfg.d_model,
        "n_layers":   model_cfg.n_layers,
        "n_heads":    model_cfg.n_heads,
        "vocab_size": model_cfg.vocab_size,
        "lr":         3e-4,
        "backend":    args.backend,
    })

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(loader)
    for step in range(1, 501):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        lr = schedule(step)

        logger.log_step(step, {
            "train/loss": loss.item(),
            "lr":          lr,
            "grad_norm":   grad_norm,
        })

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x, y   = next(iter(loader))
        _, val_loss = model(x, y)
    logger.log_validation(500, {"loss": val_loss.item()})

print("\\nDone! Check your logs.")
''')
commit("feat: add examples/train_with_logging.py — full training with TensorBoard/W&B logging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: ConsoleLogger no-crash
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_logging.py", '''\
"""
tests/test_logging.py — Tests for NanoMind training logging.
"""

import pytest
from unittest.mock import MagicMock, patch

from nanomind.logging import (
    LogConfig,
    ConsoleLogger,
    TensorBoardLogger,
    WandbLogger,
    MetricsBuffer,
    TrainingLogger,
    build_loggers,
)


# ── LogConfig ─────────────────────────────────────────────────────────────────

class TestLogConfig:
    def test_defaults(self):
        cfg = LogConfig()
        assert cfg.backends == ["console"]
        assert cfg.log_interval == 50

    def test_string_backend_becomes_list(self):
        cfg = LogConfig(backend="tensorboard")
        assert cfg.backends == ["tensorboard"]

    def test_list_backend(self):
        cfg = LogConfig(backend=["console", "tensorboard"])
        assert len(cfg.backends) == 2

    def test_invalid_backend_raises(self):
        with pytest.raises(AssertionError):
            LogConfig(backend="mlflow")

    def test_invalid_log_interval(self):
        with pytest.raises(AssertionError):
            LogConfig(log_interval=0)


# ── ConsoleLogger ─────────────────────────────────────────────────────────────

class TestConsoleLogger:
    def test_log_config_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.log_config({"lr": 3e-4, "batch_size": 32})
        out = capsys.readouterr().out
        assert "lr" in out

    def test_log_scalars_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"train/loss": 2.3, "lr": 3e-4}, step=100)
        out = capsys.readouterr().out
        assert "step" in out

    def test_finish_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.finish()

    def test_loss_formatted_to_4dp(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"loss": 2.123456}, step=1)
        out = capsys.readouterr().out
        assert "2.1235" in out

    def test_lr_formatted_scientific(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"lr": 3e-4}, step=1)
        out = capsys.readouterr().out
        assert "e-04" in out or "3.00" in out
''')
commit("test: add LogConfig validation and ConsoleLogger no-crash and format tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: MetricsBuffer
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_logging.py")
src += '''

# ── MetricsBuffer ─────────────────────────────────────────────────────────────

class TestMetricsBuffer:
    def test_empty_averages(self):
        buf = MetricsBuffer()
        assert buf.averages() == {}

    def test_single_update(self):
        buf = MetricsBuffer()
        buf.update({"loss": 2.0})
        assert abs(buf.averages()["loss"] - 2.0) < 1e-9

    def test_multiple_updates_averaged(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.update({"loss": 3.0})
        assert abs(buf.averages()["loss"] - 2.0) < 1e-6

    def test_weighted_update(self):
        buf = MetricsBuffer()
        buf.update({"loss": 2.0}, n=10)
        buf.update({"loss": 4.0}, n=10)
        # weighted average: (2*10 + 4*10) / 20 = 3.0
        assert abs(buf.averages()["loss"] - 3.0) < 1e-6

    def test_reset_clears(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.reset()
        assert buf.averages() == {}

    def test_multiple_metrics(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0, "acc": 0.8})
        avgs = buf.averages()
        assert "loss" in avgs and "acc" in avgs

    def test_contains(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        assert "loss" in buf
        assert "lr" not in buf

    def test_len(self):
        buf = MetricsBuffer()
        buf.update({"a": 1, "b": 2, "c": 3})
        assert len(buf) == 3
'''
write("tests/test_logging.py", src)
commit("test: add MetricsBuffer update, weighted average, reset, contains, and len tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: build_loggers factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_logging.py")
src += '''

# ── build_loggers factory ─────────────────────────────────────────────────────

class TestBuildLoggers:
    def test_console_backend(self):
        cfg     = LogConfig(backend="console")
        loggers = build_loggers(cfg)
        assert len(loggers) == 1
        assert isinstance(loggers[0], ConsoleLogger)

    def test_multi_backend(self):
        cfg     = LogConfig(backend=["console", "tensorboard"])
        loggers = build_loggers(cfg)
        assert len(loggers) == 2
        types   = [type(lg).__name__ for lg in loggers]
        assert "ConsoleLogger" in types
        assert "TensorBoardLogger" in types

    def test_wandb_backend(self):
        cfg     = LogConfig(backend="wandb")
        loggers = build_loggers(cfg)
        assert len(loggers) == 1
        assert isinstance(loggers[0], WandbLogger)
'''
write("tests/test_logging.py", src)
commit("test: add build_loggers factory tests for console, tensorboard, and wandb backends")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: TrainingLogger log_step buffering
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_logging.py")
src += '''

# ── TrainingLogger ────────────────────────────────────────────────────────────

class TestTrainingLogger:
    def _make(self, interval=10):
        cfg = LogConfig(backend="console", log_interval=interval)
        return TrainingLogger(cfg)

    def test_log_config_no_crash(self, capsys):
        logger = self._make()
        logger.log_config({"lr": 3e-4})
        capsys.readouterr()

    def test_log_step_buffered(self, capsys):
        logger = self._make(interval=5)
        for s in range(1, 5):
            logger.log_step(s, {"train/loss": 1.0})
        out = capsys.readouterr().out
        # Should not have printed yet (haven't hit interval)
        assert "step" not in out

    def test_log_step_flushes_at_interval(self, capsys):
        logger = self._make(interval=5)
        for s in range(1, 6):
            logger.log_step(s, {"train/loss": 1.0})
        out = capsys.readouterr().out
        assert "step" in out

    def test_log_validation_immediate(self, capsys):
        logger = self._make()
        logger.log_validation(100, {"loss": 1.5})
        out = capsys.readouterr().out
        assert "val" in out.lower() or "loss" in out

    def test_finish_no_crash(self):
        logger = self._make()
        logger.finish()

    def test_context_manager(self):
        cfg = LogConfig(backend="console", log_interval=5)
        with TrainingLogger(cfg) as logger:
            logger.log_step(5, {"train/loss": 1.0})
'''
write("tests/test_logging.py", src)
commit("test: add TrainingLogger log_step buffering, flush, validation, and context manager tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: TensorBoardLogger graceful fallback
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_logging.py")
src += '''

# ── TensorBoardLogger ─────────────────────────────────────────────────────────

class TestTensorBoardLogger:
    def test_no_crash_when_not_installed(self, tmp_path):
        """TensorBoardLogger should not raise even if tensorboard missing."""
        with patch("nanomind.logging.tensorboard.TensorBoardLogger._setup",
                   lambda self: setattr(self, "_available", False)):
            lg = TensorBoardLogger(log_dir=str(tmp_path))
            lg.log_config({"lr": 3e-4})
            lg.log_scalars({"loss": 1.0}, step=1)
            lg.finish()

    def test_log_scalars_no_crash_unavailable(self, tmp_path):
        lg = TensorBoardLogger(log_dir=str(tmp_path))
        lg._available = False
        lg.log_scalars({"loss": 1.0}, step=1)   # should not raise


# ── WandbLogger ───────────────────────────────────────────────────────────────

class TestWandbLogger:
    def test_no_crash_when_not_installed(self):
        """WandbLogger should not raise even if wandb missing."""
        lg = WandbLogger()
        lg._available = False
        lg.log_scalars({"loss": 1.0}, step=1)
        lg.finish()

    def test_log_scalars_unavailable(self):
        lg = WandbLogger()
        lg._available = False
        lg.log_scalars({"loss": 1.0}, step=1)
'''
write("tests/test_logging.py", src)
commit("test: add TensorBoardLogger and WandbLogger graceful fallback tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: TrainingLogger with model param logging
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_logging.py")
src += '''

# ── log_model_params ──────────────────────────────────────────────────────────

class TestLogModelParams:
    def test_no_crash(self):
        import torch
        import torch.nn as nn
        cfg    = LogConfig(backend="console", log_interval=5)
        logger = TrainingLogger(cfg)
        model  = nn.Linear(8, 4)
        logger.log_model_params(model, step=10)
        logger.finish()
'''
write("tests/test_logging.py", src)
commit("test: add log_model_params no-crash test with a simple Linear model")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump version + expose logging in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.6.0\"", "__version__ = \"1.7.0\"")
src = src.replace(
    "from nanomind.quant import QuantConfig, quantize_model",
    "from nanomind.quant import QuantConfig, quantize_model\n"
    "from nanomind.logging import LogConfig, TrainingLogger"
)
src = src.replace(
    "    \"quantize_model\",\n    \"__version__\",\n]",
    "    \"quantize_model\",\n"
    "    \"LogConfig\",\n"
    "    \"TrainingLogger\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v1.7.0 — expose LogConfig and TrainingLogger in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Quantization** | INT8 weight-only & dynamic quant — 4x smaller, 2x faster |",
    "| **Quantization** | INT8 weight-only & dynamic quant — 4x smaller, 2x faster |\n"
    "| **Logging** | Console, TensorBoard, W&B — unified TrainingLogger API |"
)
readme = readme.replace(
    "**Total: 400 commits across 20 days.**",
    "**Total: 420 commits across 21 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.6.0] — 2024 — INT8 Quantization",
    "## [1.7.0] — 2024 — Training Logging\n\n### Added\n"
    "- `TrainingLogger` — multiplex logger: fans out to all enabled backends\n"
    "- `ConsoleLogger` — formatted one-liner training metrics to stdout\n"
    "- `TensorBoardLogger` — scalar, histogram, hparam logging (graceful fallback)\n"
    "- `WandbLogger` — W&B integration with graceful fallback\n"
    "- `LogConfig` — backend, log_dir, project, run_name, log_interval config\n"
    "- `MetricsBuffer` — step-level metric accumulation and averaging\n"
    "- `build_loggers()` — factory to build backends from LogConfig\n"
    "- `ActivationCalibrator` (Day 20 — already in quant package)\n"
    "- `Trainer` now accepts an optional `TrainingLogger`\n"
    "- `examples/train_with_logging.py` — full training + logging demo\n\n---\n\n"
    "## [1.6.0] — 2024 — INT8 Quantization"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.7.0, update README and CHANGELOG for Day 21 training logging")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 21 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.7.0",
    "-m", "NanoMind v1.7.0 — Training Logging (TensorBoard / W&B)", check=False)
r = run("git", "push", "origin", "v1.7.0", check=False)
print("Tag v1.7.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 21 COMPLETE — v1.7.0 TAGGED! ===")
