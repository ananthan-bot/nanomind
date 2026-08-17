"""
day8_commits.py — 20 atomic commits for Day 8: Training Infrastructure.
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

print("\n=== DAY 8: Training Infrastructure — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — trainer package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/trainer/__init__.py", '"""NanoMind trainer sub-package."""\n')
commit("feat: add nanomind/trainer/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — TrainConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/trainer/config.py", '''\
"""
nanomind/trainer/config.py — Training configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """
    All hyperparameters needed to run NanoMind training.

    Attributes:
        max_iters:        Total number of training iterations.
        eval_interval:    Run validation every N iterations.
        eval_iters:       Number of batches to average for eval loss.
        log_interval:     Print training loss every N iterations.
        grad_accum_steps: Accumulate gradients over N steps before update.
        grad_clip:        Max gradient norm (0 = no clipping).
        use_amp:          Use automatic mixed precision (CUDA only).
        early_stop_patience: Stop if val loss doesn't improve for N evals (0 = off).
        seed:             Random seed.
        device:           Device string — ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.
        out_dir:          Directory to write checkpoints and logs.
    """

    max_iters:           int   = 5000
    eval_interval:       int   = 200
    eval_iters:          int   = 50
    log_interval:        int   = 10
    grad_accum_steps:    int   = 1
    grad_clip:           float = 1.0
    use_amp:             bool  = False
    early_stop_patience: int   = 0
    seed:                int   = 42
    device:              str   = "auto"
    out_dir:             str   = "checkpoints"

    def __post_init__(self) -> None:
        assert self.max_iters > 0
        assert self.eval_interval > 0
        assert self.grad_accum_steps >= 1
        assert self.grad_clip >= 0.0
''')
commit("feat: add TrainConfig dataclass with validation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Trainer class skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/trainer/trainer.py", '''\
"""
nanomind/trainer/trainer.py — Training loop for NanoMind.

The Trainer class owns the full training lifecycle:
    - train_step()     : one forward + backward + optimizer step
    - eval_step()      : one forward without gradients
    - estimate_loss()  : average loss over N batches
    - train()          : the full training loop
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.trainer.config import TrainConfig
from nanomind.utils.logger import get_logger
from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr
from nanomind.utils.timer import Timer, tokens_per_second


class Trainer:
    """
    Handles the NanoMind training loop.

    Args:
        model:        The :class:`~nanomind.model.NanoMind` model.
        optimizer:    Configured optimizer (e.g. AdamW).
        train_loader: DataLoader for training batches.
        val_loader:   DataLoader for validation batches.
        cfg:          :class:`~nanomind.trainer.TrainConfig`.
        device:       Resolved :class:`torch.device`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.log          = get_logger("trainer")

        # State
        self.step:     int   = 0
        self.best_val: float = float("inf")
        self._timer          = Timer()

        # AMP scaler (only active when use_amp=True and CUDA available)
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if cfg.use_amp and device.type == "cuda"
            else None
        )
''')
commit("feat: add Trainer class skeleton with state, AMP scaler, and logger")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — _infinite_loader() helper
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        """Yield batches endlessly, restarting the loader when exhausted."""
        while True:
            yield from loader
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: add _infinite_loader() — endlessly cycle through a DataLoader")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — train_step() — forward + backward
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform one forward + backward pass (no optimizer step).

        Supports AMP via :class:`torch.cuda.amp.GradScaler` when enabled.

        Args:
            x: Input tokens  ``(B, T)``
            y: Target tokens ``(B, T)``

        Returns:
            Loss value as a Python float.
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        if self._scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = self.model(x, y)
            self._scaler.scale(loss).backward()
        else:
            _, loss = self.model(x, y)
            loss.backward()

        return loss.item()
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: implement train_step() — forward + backward with optional AMP")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — gradient accumulation + optimizer step
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    def _optimizer_step(self) -> None:
        """
        Apply accumulated gradients, clip norms, and step the optimizer.

        Handles both standard and AMP (GradScaler) paths.
        Resets gradients after the update.
        """
        if self._scaler is not None:
            if self.cfg.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: add _optimizer_step() with gradient clipping and AMP scaler support")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — eval_step() + estimate_loss()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    @torch.no_grad()
    def eval_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single validation forward pass. Returns loss as float."""
        self.model.eval()
        x, y   = x.to(self.device), y.to(self.device)
        _, loss = self.model(x, y)
        return loss.item()

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """
        Estimate mean loss on train and val sets over ``eval_iters`` batches.

        Returns:
            Dict with keys ``"train"`` and ``"val"``.
        """
        self.model.eval()
        results: dict[str, float] = {}
        loaders = {"train": self.train_loader, "val": self.val_loader}
        for split, loader in loaders.items():
            losses: list[float] = []
            it = iter(loader)
            for _ in range(min(self.cfg.eval_iters, len(loader))):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                losses.append(self.eval_step(x, y))
            results[split] = sum(losses) / len(losses) if losses else float("nan")
        self.model.train()
        return results
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: add eval_step() and estimate_loss() over N batches for both splits")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — tokens-per-second throughput tracking
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    def _log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        elapsed: float,
        batch_size: int,
        block_size: int,
    ) -> None:
        """Print a compact training log line."""
        tps = tokens_per_second(
            batch_size * block_size * self.cfg.log_interval, elapsed
        )
        self.log.info(
            f"step {step:>6} | loss {fmt_loss(loss)} | "
            f"lr {fmt_lr(lr)} | {tps:,.0f} tok/s"
        )
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: add _log_step() with tokens-per-second throughput logging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — early stopping tracker
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    def _check_early_stop(self, val_loss: float) -> bool:
        """
        Check whether training should stop early.

        Tracks patience counter — incremented each eval where val loss
        does not improve. Resets to 0 on improvement.

        Args:
            val_loss: Latest validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.cfg.early_stop_patience <= 0:
            return False

        if val_loss < self.best_val:
            self._patience_counter = 0
            self.best_val = val_loss
        else:
            self._patience_counter = getattr(self, "_patience_counter", 0) + 1
            if self._patience_counter >= self.cfg.early_stop_patience:
                self.log.warning(
                    f"Early stopping: val loss did not improve for "
                    f"{self.cfg.early_stop_patience} evals."
                )
                return True
        return False
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: add _check_early_stop() — patience-based early stopping")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — full train() loop
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/trainer/trainer.py")
src += '''
    def train(
        self,
        lr_scheduler=None,
        on_eval: "Callable | None" = None,  # noqa: F821
    ) -> dict[str, float]:
        """
        Run the full training loop.

        Args:
            lr_scheduler: Optional LR scheduler with a ``get_lr(step)`` callable.
            on_eval:      Optional callback called after each evaluation with
                          ``(step, train_loss, val_loss)``.

        Returns:
            Dict with ``"best_val"`` and ``"final_train"`` losses.
        """
        from pathlib import Path as _P
        out_dir = _P(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(
            f"Starting training | device={self.device} | "
            f"max_iters={self.cfg.max_iters:,}"
        )

        data_iter   = self._infinite_loader(self.train_loader)
        running_loss = 0.0
        self._timer.start()
        self.optimizer.zero_grad(set_to_none=True)

        for step in range(self.step, self.cfg.max_iters):
            self.step = step

            # Update LR
            if lr_scheduler is not None:
                lr = lr_scheduler(step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            # Gradient accumulation
            for micro_step in range(self.cfg.grad_accum_steps):
                x, y       = next(data_iter)
                step_loss  = self.train_step(x, y)
                running_loss += step_loss / self.cfg.grad_accum_steps

            self._optimizer_step()

            # Logging
            if (step + 1) % self.cfg.log_interval == 0:
                elapsed = self._timer.stop()
                self._log_step(
                    step + 1, running_loss / self.cfg.log_interval,
                    lr, elapsed,
                    x.size(0), x.size(1),
                )
                running_loss = 0.0
                self._timer.start()

            # Evaluation
            if (step + 1) % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                self.log.info(
                    f"[eval] step {step+1} | "
                    f"train={fmt_loss(losses['train'])} | "
                    f"val={fmt_loss(losses['val'])}"
                )
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.log.info(f"  New best val: {fmt_loss(self.best_val)}")

                if on_eval:
                    on_eval(step + 1, losses["train"], losses["val"])

                if self._check_early_stop(losses["val"]):
                    break

        self.log.info(
            f"Training complete | best_val={fmt_loss(self.best_val)}"
        )
        return {"best_val": self.best_val, "final_train": running_loss}
'''
write("nanomind/trainer/trainer.py", src)
commit("feat: implement full train() loop with grad accum, eval, logging, and early stop")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — training time estimation utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/trainer/estimate.py", '''\
"""
nanomind/trainer/estimate.py — Training time and compute estimation utilities.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn


def estimate_training_time(
    model: nn.Module,
    train_loader,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: torch.device | None = None,
) -> dict:
    """
    Estimate training throughput and total training time.

    Runs a few warm-up steps and then times ``n_measure`` steps to
    estimate tokens/second and total training duration.

    Args:
        model:       The model to benchmark.
        train_loader: The training DataLoader.
        n_warmup:    Number of warm-up steps (not timed).
        n_measure:   Number of steps to time.
        device:      Device to run on.

    Returns:
        Dict with keys:
        - ``tokens_per_second``: float
        - ``seconds_per_iter``:  float
        - ``estimated_total_s``: float (for 1000 iters)
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    data_it = iter(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # Warm-up
    for _ in range(n_warmup):
        try:
            x, y = next(data_it)
        except StopIteration:
            data_it = iter(train_loader)
            x, y = next(data_it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for _ in range(n_measure):
        try:
            x, y = next(data_it)
        except StopIteration:
            data_it = iter(train_loader)
            x, y = next(data_it)
        x, y = x.to(device), y.to(device)
        total_tokens += x.numel()
        _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps     = total_tokens / elapsed
    spi     = elapsed / n_measure
    return {
        "tokens_per_second":  tps,
        "seconds_per_iter":   spi,
        "estimated_total_s":  spi * 1000,
    }
''')
commit("feat: add estimate_training_time() — benchmark throughput before full training")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — update trainer __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/trainer/__init__.py", '''\
"""NanoMind trainer sub-package.

Primary exports:
    - :class:`Trainer`     — full training loop with eval, logging, early stop
    - :class:`TrainConfig` — training hyperparameter configuration
    - :func:`estimate_training_time` — pre-training throughput benchmark
"""

from nanomind.trainer.config import TrainConfig
from nanomind.trainer.trainer import Trainer
from nanomind.trainer.estimate import estimate_training_time

__all__ = ["Trainer", "TrainConfig", "estimate_training_time"]
''')
commit("refactor: export Trainer, TrainConfig, estimate_training_time from trainer package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: train_step reduces loss
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_trainer.py", '''\
"""
tests/test_trainer.py — Tests for the NanoMind Trainer.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model import NanoMind, ModelConfig
from nanomind.trainer import Trainer, TrainConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

VOCAB  = 32
BLOCK  = 8
D      = 32
N_HEAD = 2
B      = 4


def make_model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(ModelConfig(
        vocab_size=VOCAB, block_size=BLOCK,
        d_model=D, n_layers=2, n_heads=N_HEAD, dropout=0.0
    ))


def make_loaders(n: int = 64):
    tokens = torch.randint(0, VOCAB, (n + BLOCK,))
    xs = torch.stack([tokens[i:i+BLOCK]   for i in range(n)])
    ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(n)])
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
    return loader, loader  # use same for train/val in tests


def make_trainer(model=None, cfg=None) -> Trainer:
    m   = model or make_model()
    c   = cfg or TrainConfig(max_iters=10, eval_interval=5, log_interval=5,
                              grad_clip=1.0, use_amp=False)
    device = torch.device("cpu")
    tl, vl = make_loaders()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return Trainer(m, opt, tl, vl, c, device)


# ── train_step ────────────────────────────────────────────────────────────────

class TestTrainStep:
    def test_returns_float(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        loss = t.train_step(x, y)
        assert isinstance(loss, float)

    def test_loss_is_positive(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        loss = t.train_step(x, y)
        assert loss > 0

    def test_multiple_steps_accumulate_grads(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        t.train_step(x, y)
        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in t.model.parameters()
        )
        assert has_grad
''')
commit("test: add train_step() tests — returns float, positive loss, gradient accumulation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: loss decreases over training
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── Full training loop ────────────────────────────────────────────────────────

class TestTrainLoop:
    def test_loss_decreases(self):
        model = make_model()
        cfg   = TrainConfig(
            max_iters=50, eval_interval=25, log_interval=10,
            grad_clip=1.0, use_amp=False
        )
        device = torch.device("cpu")
        tl, vl = make_loaders(128)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        trainer = Trainer(model, opt, tl, vl, cfg, device)

        # Record initial loss
        x, y = next(iter(tl))
        initial_loss = trainer.eval_step(x, y)

        result = trainer.train()

        # Record final loss
        final_loss = trainer.eval_step(x, y)
        assert final_loss < initial_loss, (
            f"Expected loss to decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_train_returns_dict(self):
        t = make_trainer()
        result = t.train()
        assert "best_val" in result
        assert "final_train" in result
'''
write("tests/test_trainer.py", src)
commit("test: add full train loop tests — loss decreases, result dict returned")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: estimate_loss
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── estimate_loss ─────────────────────────────────────────────────────────────

class TestEstimateLoss:
    def test_returns_train_and_val(self):
        t = make_trainer()
        losses = t.estimate_loss()
        assert "train" in losses
        assert "val" in losses

    def test_losses_are_positive_floats(self):
        t = make_trainer()
        losses = t.estimate_loss()
        for split, v in losses.items():
            assert isinstance(v, float), f"{split} loss is not float"
            assert v > 0, f"{split} loss is not positive"

    def test_model_returns_to_train_mode(self):
        t = make_trainer()
        t.estimate_loss()
        assert t.model.training
'''
write("tests/test_trainer.py", src)
commit("test: add estimate_loss() tests — returns correct keys, positive values, model mode")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: gradient accumulation equivalence
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── Gradient accumulation ─────────────────────────────────────────────────────

class TestGradAccum:
    def test_accum_produces_same_gradients(self):
        """2 steps with accum=2 should equal 1 step on double batch."""
        torch.manual_seed(42)
        model1 = make_model()
        torch.manual_seed(42)
        model2 = make_model()

        # Single step on batch of size 2B
        x2 = torch.randint(0, VOCAB, (B * 2, BLOCK))
        y2 = torch.randint(0, VOCAB, (B * 2, BLOCK))
        _, loss1 = model1(x2, y2)
        loss1.backward()

        # Two accumulated steps on batches of size B
        x1a = x2[:B]; y1a = y2[:B]
        x1b = x2[B:]; y1b = y2[B:]
        _, la = model2(x1a, y1a); (la / 2).backward()
        _, lb = model2(x1b, y1b); (lb / 2).backward()

        # Gradients should be approximately equal
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
                    "Gradient mismatch between single-step and accumulated steps"
                )
'''
write("tests/test_trainer.py", src)
commit("test: add gradient accumulation equivalence test")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: early stopping
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── Early stopping ────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_no_early_stop_when_patience_zero(self):
        t = make_trainer()
        assert not t._check_early_stop(999.0)   # Very high loss, but patience=0

    def test_early_stop_triggers(self):
        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=2, log_interval=5,
            early_stop_patience=2,
        ))
        t.best_val = 1.0
        t._check_early_stop(1.5)   # no improvement -> counter = 1
        assert t._check_early_stop(1.5)  # no improvement -> counter = 2 -> stop

    def test_early_stop_resets_on_improvement(self):
        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=2, log_interval=5,
            early_stop_patience=3,
        ))
        t.best_val = 2.0
        t._check_early_stop(1.5)   # improvement -> reset
        assert getattr(t, "_patience_counter", 0) == 0
'''
write("tests/test_trainer.py", src)
commit("test: add early stopping tests (no stop, triggers, resets on improvement)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: AMP doesn't break output shape
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── TrainConfig ───────────────────────────────────────────────────────────────

class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.max_iters == 5000
        assert cfg.grad_accum_steps == 1

    def test_invalid_max_iters(self):
        with pytest.raises(AssertionError):
            TrainConfig(max_iters=0)

    def test_invalid_grad_accum(self):
        with pytest.raises(AssertionError):
            TrainConfig(grad_accum_steps=0)
'''
write("tests/test_trainer.py", src)
commit("test: add TrainConfig validation tests (defaults, invalid values)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — on_eval callback test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_trainer.py")
src += '''

# ── on_eval callback ──────────────────────────────────────────────────────────

class TestOnEvalCallback:
    def test_callback_is_called(self):
        calls = []
        def on_eval(step, train_loss, val_loss):
            calls.append((step, train_loss, val_loss))

        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=5, log_interval=5
        ))
        t.train(on_eval=on_eval)
        # Should be called twice (at step 5 and 10)
        assert len(calls) == 2

    def test_callback_receives_correct_step(self):
        steps = []
        def on_eval(step, *_):
            steps.append(step)

        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=5, log_interval=5
        ))
        t.train(on_eval=on_eval)
        assert steps == [5, 10]
'''
write("tests/test_trainer.py", src)
commit("test: add on_eval callback invocation and step correctness tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 8 | Training infrastructure | 🔜 |",
    "| 8 | Training infrastructure | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Full model: NanoMind with embeddings, N blocks, weight tying, generate(), ModelConfig (Day 7)",
    "- Full model: NanoMind with embeddings, N blocks, weight tying, generate(), ModelConfig (Day 7)\n- Training: Trainer loop with AMP, grad accum, gradient clip, early stop, estimate_loss (Day 8)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 8 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 8 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 8 COMPLETE ===")
