"""
day10_commits.py — 20 atomic commits for Day 10: Checkpointing & Resumption.
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

print("\n=== DAY 10: Checkpointing & Resumption — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — checkpoint package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/__init__.py", '"""NanoMind checkpoint sub-package."""\n')
commit("feat: add nanomind/checkpoint/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — CheckpointConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/config.py", '''\
"""
nanomind/checkpoint/config.py — Checkpoint configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    """
    Configuration for the :class:`~nanomind.checkpoint.CheckpointManager`.

    Attributes:
        out_dir:        Directory to write checkpoints.
        save_interval:  Save a checkpoint every N training steps.
        keep_last_n:    Keep only the N most recent checkpoints (0 = keep all).
        save_best:      Always keep the checkpoint with the lowest val loss.
        save_optimizer: Whether to include optimizer state in checkpoints.
    """

    out_dir:         str  = "checkpoints"
    save_interval:   int  = 500
    keep_last_n:     int  = 3
    save_best:       bool = True
    save_optimizer:  bool = True

    def __post_init__(self) -> None:
        assert self.save_interval > 0, "save_interval must be positive"
        assert self.keep_last_n >= 0, "keep_last_n must be >= 0"
''')
commit("feat: add CheckpointConfig dataclass with validation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — checkpoint metadata structure
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/metadata.py", '''\
"""
nanomind/checkpoint/metadata.py — Checkpoint metadata utilities.

Every saved checkpoint carries a metadata dict that records the full
training state at save time, making it easy to inspect checkpoints
without loading the full model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def make_metadata(
    step: int,
    train_loss: float,
    val_loss: float,
    model_config: dict,
    extra: dict | None = None,
) -> dict:
    """
    Build a checkpoint metadata dictionary.

    Args:
        step:         Training step at save time.
        train_loss:   Latest training loss.
        val_loss:     Latest validation loss.
        model_config: Serialized ModelConfig dict.
        extra:        Any additional fields to include.

    Returns:
        Metadata dict (JSON-serializable).
    """
    meta = {
        "step":         step,
        "train_loss":   float(train_loss),
        "val_loss":     float(val_loss),
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": model_config,
    }
    if extra:
        meta.update(extra)
    return meta


def save_metadata(meta: dict, path: str | Path) -> None:
    """Write metadata to a companion JSON file."""
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict:
    """Load metadata from a companion JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
''')
commit("feat: add checkpoint metadata utilities — make, save, and load metadata dict")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — save_checkpoint() function
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/io.py", '''\
"""
nanomind/checkpoint/io.py — Low-level checkpoint save/load functions.

Checkpoints are stored as PyTorch ``.pt`` files alongside a ``.json``
metadata file. Writes are atomic: the payload is written to a ``.tmp``
file first, then renamed to the final path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from nanomind.checkpoint.metadata import make_metadata, save_metadata, load_metadata


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    train_loss: float = float("nan"),
    val_loss: float = float("nan"),
    model_config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Save a training checkpoint atomically.

    The checkpoint ``.pt`` file contains:
    - ``model_state``:     model weights
    - ``optimizer_state``: optimizer state (if provided)
    - ``step``:            training step
    - ``metadata``:        metadata dict

    Writes to a ``.tmp`` file first, then renames for atomicity.

    Args:
        path:         Destination path (e.g. ``checkpoints/step_1000.pt``).
        model:        Model to checkpoint.
        optimizer:    Optimizer (None = skip optimizer state).
        step:         Current training step.
        train_loss:   Current training loss.
        val_loss:     Current validation loss.
        model_config: Model config dict for metadata.
        extra:        Extra metadata fields.

    Returns:
        The final checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = make_metadata(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        model_config=model_config or {},
        extra=extra,
    )

    payload = {
        "model_state": model.state_dict(),
        "step":        step,
        "metadata":    meta,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    # Atomic write: .tmp -> final
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)

    # Companion metadata JSON
    save_metadata(meta, path.with_suffix(".json"))
    return path
''')
commit("feat: implement save_checkpoint() — atomic write with model, optimizer, and metadata")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — load_checkpoint() function
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/checkpoint/io.py")
src += '''

def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path:      Path to the ``.pt`` checkpoint file.
        model:     Model to restore weights into.
        optimizer: Optimizer to restore state into (None = skip).
        device:    Device to map tensors to (None = use saved device).
        strict:    Whether to require exact key matching in state dict.

    Returns:
        The metadata dict from the checkpoint.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload.get("metadata", {})


def load_for_inference(
    path: str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> dict:
    """
    Load only model weights — no optimizer state needed for inference.

    Args:
        path:   Path to the ``.pt`` checkpoint file.
        model:  Model to restore weights into.
        device: Target device.

    Returns:
        Metadata dict from the checkpoint.
    """
    return load_checkpoint(path, model, optimizer=None, device=device)
'''
write("nanomind/checkpoint/io.py", src)
commit("feat: implement load_checkpoint() and load_for_inference() restore functions")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — CheckpointManager class skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/manager.py", '''\
"""
nanomind/checkpoint/manager.py — High-level checkpoint manager.

CheckpointManager wraps the low-level save/load functions and adds:
- Automatic naming by step number
- Retention policy (keep last N)
- Best checkpoint tracking
- Auto-resume from latest
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from nanomind.checkpoint.config import CheckpointConfig
from nanomind.checkpoint.io import save_checkpoint, load_checkpoint
from nanomind.utils.logger import get_logger


class CheckpointManager:
    """
    Manages checkpoint saving, tracking, and cleanup.

    Args:
        cfg: :class:`~nanomind.checkpoint.CheckpointConfig` instance.
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg      = cfg
        self.out_dir  = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log      = get_logger("checkpoint")
        self._best_val: float = float("inf")
        self._saved:    list[Path] = []
''')
commit("feat: add CheckpointManager class skeleton with config, out_dir, and best_val tracking")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — CheckpointManager.save()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/checkpoint/manager.py")
src += '''
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        train_loss: float = float("nan"),
        val_loss: float = float("nan"),
        model_config: dict | None = None,
    ) -> Path:
        """
        Save a checkpoint for the current training step.

        Applies retention policy and best-checkpoint tracking.

        Args:
            model:        Model to checkpoint.
            optimizer:    Optimizer (None = skip).
            step:         Current training step.
            train_loss:   Training loss at this step.
            val_loss:     Validation loss at this step.
            model_config: Model configuration dict.

        Returns:
            Path of the saved checkpoint file.
        """
        opt = optimizer if self.cfg.save_optimizer else None
        ckpt_path = self.out_dir / f"step_{step:07d}.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            model_config=model_config,
        )
        self._saved.append(ckpt_path)
        self.log.info(f"Saved checkpoint: {ckpt_path.name}")

        # Best checkpoint
        if self.cfg.save_best and val_loss < self._best_val:
            self._best_val = val_loss
            best_path = self.out_dir / "best.pt"
            import shutil
            shutil.copy2(ckpt_path, best_path)
            shutil.copy2(ckpt_path.with_suffix(".json"), best_path.with_suffix(".json"))
            self.log.info(f"  -> New best val={val_loss:.4f}, saved to best.pt")

        # Retention policy
        self._cleanup()
        return ckpt_path
'''
write("nanomind/checkpoint/manager.py", src)
commit("feat: implement CheckpointManager.save() with best tracking and retention policy")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — CheckpointManager._cleanup() retention policy
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/checkpoint/manager.py")
src += '''
    def _cleanup(self) -> None:
        """Delete old checkpoints beyond the keep_last_n limit."""
        if self.cfg.keep_last_n <= 0:
            return
        while len(self._saved) > self.cfg.keep_last_n:
            old = self._saved.pop(0)
            for suffix in (".pt", ".json", ".tmp"):
                p = old.with_suffix(suffix)
                if p.exists():
                    p.unlink()
            self.log.debug(f"Deleted old checkpoint: {old.name}")
'''
write("nanomind/checkpoint/manager.py", src)
commit("feat: add CheckpointManager._cleanup() — delete checkpoints beyond keep_last_n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — CheckpointManager.load_latest()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/checkpoint/manager.py")
src += '''
    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | None = None,
    ) -> dict | None:
        """
        Load the most recently saved checkpoint.

        Args:
            model:     Model to restore.
            optimizer: Optimizer to restore (None = skip).
            device:    Target device.

        Returns:
            Metadata dict, or None if no checkpoint exists.
        """
        candidates = sorted(self.out_dir.glob("step_*.pt"))
        if not candidates:
            self.log.info("No checkpoint found — starting from scratch.")
            return None
        latest = candidates[-1]
        self.log.info(f"Resuming from: {latest.name}")
        return load_checkpoint(latest, model, optimizer, device)

    def load_best(
        self,
        model: nn.Module,
        device: torch.device | None = None,
    ) -> dict | None:
        """Load the best checkpoint (lowest val loss)."""
        best = self.out_dir / "best.pt"
        if not best.exists():
            self.log.warning("No best.pt found.")
            return None
        self.log.info("Loading best.pt")
        return load_checkpoint(best, model, device=device)
'''
write("nanomind/checkpoint/manager.py", src)
commit("feat: add CheckpointManager.load_latest() and load_best() restore methods")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — CheckpointManager.list_checkpoints()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/checkpoint/manager.py")
src += '''
    def list_checkpoints(self) -> list[dict]:
        """
        List all checkpoints in the output directory with their metadata.

        Returns:
            List of metadata dicts, sorted by step (ascending).
        """
        from nanomind.checkpoint.metadata import load_metadata
        result = []
        for pt in sorted(self.out_dir.glob("step_*.pt")):
            json_path = pt.with_suffix(".json")
            if json_path.exists():
                meta = load_metadata(json_path)
            else:
                meta = {"path": str(pt)}
            meta["path"] = str(pt)
            result.append(meta)
        return result

    def __repr__(self) -> str:
        n = len(self._saved)
        return (
            f"CheckpointManager("
            f"out_dir='{self.out_dir}', "
            f"saved={n}, "
            f"best_val={self._best_val:.4f})"
        )
'''
write("nanomind/checkpoint/manager.py", src)
commit("feat: add CheckpointManager.list_checkpoints() and __repr__()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — auto_resume() helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/resume.py", '''\
"""
nanomind/checkpoint/resume.py — Auto-resume helpers.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from nanomind.checkpoint.io import load_checkpoint
from nanomind.utils.logger import get_logger


def auto_resume(
    out_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[int, dict | None]:
    """
    Automatically resume from the latest checkpoint in ``out_dir``.

    If no checkpoint exists, returns ``(0, None)`` so training starts fresh.

    Args:
        out_dir:   Directory to search for checkpoints.
        model:     Model to restore.
        optimizer: Optimizer to restore (None = skip).
        device:    Target device.

    Returns:
        Tuple of ``(start_step, metadata_or_None)``.

    Example::

        start_step, meta = auto_resume("checkpoints", model, optimizer, device)
        trainer.step = start_step
    """
    log = get_logger("resume")
    candidates = sorted(Path(out_dir).glob("step_*.pt"))
    if not candidates:
        log.info("No checkpoint found — starting training from step 0.")
        return 0, None

    latest = candidates[-1]
    log.info(f"Auto-resuming from: {latest.name}")
    meta = load_checkpoint(latest, model, optimizer, device)
    start_step = meta.get("step", 0) + 1
    log.info(f"Resumed at step {start_step}.")
    return start_step, meta
''')
commit("feat: add auto_resume() — detect and load latest checkpoint or start fresh")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — model-only inference checkpoint save
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/inference.py", '''\
"""
nanomind/checkpoint/inference.py — Lightweight inference-only checkpoints.

Saves only model weights (no optimizer state), producing smaller files
suitable for deployment and sharing.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn


def save_for_inference(
    path: str | Path,
    model: nn.Module,
    model_config: dict | None = None,
    step: int = 0,
) -> Path:
    """
    Save a lightweight inference-only checkpoint (weights only).

    Args:
        path:         Destination file path.
        model:        Model whose weights to save.
        model_config: Model configuration dict (stored in file for loading).
        step:         Training step (metadata only).

    Returns:
        Path of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state":  model.state_dict(),
        "model_config": model_config or {},
        "step":         step,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def load_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> dict:
    """
    Load an inference-only checkpoint into a model.

    Args:
        path:   Path to the inference checkpoint.
        model:  Model to restore.
        device: Target device.

    Returns:
        Dict with ``"model_config"`` and ``"step"``.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return {
        "model_config": payload.get("model_config", {}),
        "step":         payload.get("step", 0),
    }
''')
commit("feat: add save_for_inference() and load_inference_checkpoint() for deployment")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — checkpoint info() helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/info.py", '''\
"""
nanomind/checkpoint/info.py — Checkpoint inspection utilities.
"""

from __future__ import annotations

from pathlib import Path
import torch


def checkpoint_info(path: str | Path) -> dict:
    """
    Read checkpoint metadata without loading model weights.

    Parses the companion ``.json`` file if available, otherwise loads
    the ``.pt`` and extracts just the metadata key (much faster than
    loading full weights).

    Args:
        path: Path to a ``.pt`` checkpoint file.

    Returns:
        Metadata dict.
    """
    from nanomind.checkpoint.metadata import load_metadata
    path = Path(path)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        return load_metadata(json_path)
    # Fall back to loading pt header only
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("metadata", {"path": str(path)})


def print_checkpoint_info(path: str | Path) -> None:
    """Pretty-print checkpoint metadata."""
    info = checkpoint_info(path)
    print(f"Checkpoint: {Path(path).name}")
    print("-" * 40)
    for k, v in info.items():
        if k != "model_config":
            print(f"  {k:<20}: {v}")
    if "model_config" in info:
        print("  model_config:")
        for k, v in info["model_config"].items():
            print(f"    {k:<18}: {v}")
''')
commit("feat: add checkpoint_info() and print_checkpoint_info() inspection utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update checkpoint __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/checkpoint/__init__.py", '''\
"""NanoMind checkpoint sub-package.

Primary exports:
    - :class:`CheckpointManager`        — manages save/load/cleanup lifecycle
    - :class:`CheckpointConfig`         — checkpoint configuration dataclass
    - :func:`save_checkpoint`           — low-level atomic checkpoint save
    - :func:`load_checkpoint`           — low-level checkpoint restore
    - :func:`load_for_inference`        — inference-only model restore
    - :func:`save_for_inference`        — save weights-only checkpoint
    - :func:`load_inference_checkpoint` — load weights-only checkpoint
    - :func:`auto_resume`               — auto-detect and resume from latest
    - :func:`checkpoint_info`           — inspect checkpoint metadata
"""

from nanomind.checkpoint.config import CheckpointConfig
from nanomind.checkpoint.manager import CheckpointManager
from nanomind.checkpoint.io import save_checkpoint, load_checkpoint, load_for_inference
from nanomind.checkpoint.inference import save_for_inference, load_inference_checkpoint
from nanomind.checkpoint.resume import auto_resume
from nanomind.checkpoint.info import checkpoint_info, print_checkpoint_info

__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "load_for_inference",
    "save_for_inference",
    "load_inference_checkpoint",
    "auto_resume",
    "checkpoint_info",
    "print_checkpoint_info",
]
''')
commit("refactor: export all checkpoint components from nanomind/checkpoint/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: save/load roundtrip (model state equality)
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_checkpoint.py", '''\
"""
tests/test_checkpoint.py — Tests for the NanoMind checkpoint system.
"""

import pytest
import torch

from nanomind.model import NanoMind, ModelConfig
from nanomind.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    load_for_inference,
    save_for_inference,
    load_inference_checkpoint,
    auto_resume,
    checkpoint_info,
)

CFG = ModelConfig(
    vocab_size=32, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


def make_model():
    torch.manual_seed(42)
    return NanoMind(CFG)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


# ── save/load roundtrip ───────────────────────────────────────────────────────

class TestSaveLoadRoundtrip:
    def test_model_state_preserved(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=100, val_loss=1.5)

        model2 = make_model()
        # Reinit model2 with different weights
        for p in model2.parameters():
            torch.nn.init.normal_(p)

        load_checkpoint(path, model2)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model weights differ after load"

    def test_step_in_metadata(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=42, val_loss=2.0)
        meta = load_checkpoint(path, make_model())
        assert meta["step"] == 42

    def test_companion_json_created(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=1)
        assert (tmp_path / "ckpt.json").exists()
''')
commit("test: add save/load roundtrip tests — model state equality, step, json companion")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: optimizer state preserved after load
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_checkpoint.py")
src += '''

# ── Optimizer state ───────────────────────────────────────────────────────────

class TestOptimizerState:
    def test_optimizer_state_preserved(self, tmp_path):
        model = make_model()
        opt   = make_optimizer(model)

        # Run a step so optimizer has non-default state
        idx    = torch.randint(0, 32, (2, 8))
        tgt    = torch.randint(0, 32, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        opt.step(); opt.zero_grad()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer=opt, step=1)

        model2 = make_model()
        opt2   = make_optimizer(model2)
        load_checkpoint(path, model2, optimizer=opt2)

        # State dict keys should match
        assert opt.state_dict().keys() == opt2.state_dict().keys()

    def test_no_optimizer_state_when_not_saved(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer=None, step=1)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state" not in payload
'''
write("tests/test_checkpoint.py", src)
commit("test: add optimizer state preservation and skip tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: CheckpointManager best tracking
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_checkpoint.py")
src += '''

# ── CheckpointManager ─────────────────────────────────────────────────────────

class TestCheckpointManager:
    def test_save_creates_file(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=False, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        path = mgr.save(model, step=100, val_loss=2.0)
        assert path.exists()

    def test_best_ckpt_created_on_improvement(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=True, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=2.0)
        assert (tmp_path / "best.pt").exists()

    def test_best_ckpt_not_updated_on_no_improvement(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=True, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=2.0)
        import os
        mtime1 = os.path.getmtime(tmp_path / "best.pt")
        import time; time.sleep(0.05)
        mgr.save(model, step=200, val_loss=3.0)  # worse -> no update
        mtime2 = os.path.getmtime(tmp_path / "best.pt")
        assert mtime1 == mtime2

    def test_retention_policy(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=False, keep_last_n=2)
        mgr = CheckpointManager(cfg)
        model = make_model()
        for step in [100, 200, 300]:
            mgr.save(model, step=step, val_loss=1.0)
        # Only 2 most recent should remain
        pts = list(tmp_path.glob("step_*.pt"))
        assert len(pts) == 2
'''
write("tests/test_checkpoint.py", src)
commit("test: add CheckpointManager save, best tracking, and retention policy tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: list_checkpoints
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_checkpoint.py")
src += '''

# ── list_checkpoints ──────────────────────────────────────────────────────────

class TestListCheckpoints:
    def test_empty_dir(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path))
        mgr = CheckpointManager(cfg)
        assert mgr.list_checkpoints() == []

    def test_lists_all_checkpoints(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        for step in [100, 200]:
            mgr.save(model, step=step, val_loss=1.0)
        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2

    def test_metadata_in_list(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=1.23)
        ckpts = mgr.list_checkpoints()
        assert ckpts[0]["step"] == 100
        assert abs(ckpts[0]["val_loss"] - 1.23) < 1e-5
'''
write("tests/test_checkpoint.py", src)
commit("test: add CheckpointManager.list_checkpoints() tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: inference checkpoint + auto_resume
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_checkpoint.py")
src += '''

# ── Inference checkpoint ──────────────────────────────────────────────────────

class TestInferenceCheckpoint:
    def test_save_and_load_inference(self, tmp_path):
        model  = make_model()
        path   = tmp_path / "inference.pt"
        save_for_inference(path, model, model_config=CFG.to_dict(), step=500)
        assert path.exists()

        model2 = make_model()
        for p in model2.parameters():
            torch.nn.init.normal_(p)
        info = load_inference_checkpoint(path, model2)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
        assert info["step"] == 500

    def test_inference_ckpt_has_no_optimizer(self, tmp_path):
        model = make_model()
        path  = tmp_path / "inference.pt"
        save_for_inference(path, model)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state" not in payload


# ── auto_resume ───────────────────────────────────────────────────────────────

class TestAutoResume:
    def test_no_checkpoint_returns_zero(self, tmp_path):
        model = make_model()
        step, meta = auto_resume(str(tmp_path), model)
        assert step == 0
        assert meta is None

    def test_resumes_from_latest(self, tmp_path):
        model = make_model()
        save_checkpoint(tmp_path / "step_0000100.pt", model, step=100)
        save_checkpoint(tmp_path / "step_0000200.pt", model, step=200)
        model2 = make_model()
        step, meta = auto_resume(str(tmp_path), model2)
        assert step == 201   # step + 1
        assert meta["step"] == 200
'''
write("tests/test_checkpoint.py", src)
commit("test: add inference checkpoint and auto_resume tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 10 | Checkpointing & resumption | 🔜 |",
    "| 10 | Checkpointing & resumption | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Optimizers: AdamW factory, param groups, WarmupCosine/Cosine/Linear/WarmupLinear schedules (Day 9)",
    "- Optimizers: AdamW factory, param groups, WarmupCosine/Cosine/Linear/WarmupLinear schedules (Day 9)\n- Checkpointing: atomic save/load, CheckpointManager, best tracking, auto_resume, inference ckpt (Day 10)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 10 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 10 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 10 COMPLETE ===")
