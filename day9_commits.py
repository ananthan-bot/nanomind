"""
day9_commits.py — 20 atomic commits for Day 9: Optimizers & LR Scheduling.
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

print("\n=== DAY 9: Optimizers & LR Scheduling — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — optim package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/__init__.py", '"""NanoMind optimizer and LR scheduler sub-package."""\n')
commit("feat: add nanomind/optim/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — param group builder (split weight decay)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/param_groups.py", '''\
"""
nanomind/optim/param_groups.py — Parameter group utilities for AdamW.

AdamW applies weight decay to all parameters by default, but it should NOT
be applied to biases, LayerNorm/RMSNorm weights, or embedding weights —
decaying these degrades performance.

This module splits parameters into two groups:
  - ``decay``:   all weight matrices (2D+)
  - ``no_decay``: biases, 1D params (norms), embeddings
"""

from __future__ import annotations

import torch.nn as nn


def get_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> list[dict]:
    """
    Split model parameters into decay and no-decay groups for AdamW.

    Rules:
    - 2D+ parameters (weight matrices) get weight decay.
    - 1D parameters (biases, norm weights) get NO weight decay.
    - Embedding weights get NO weight decay.

    Args:
        model:        The model whose parameters to group.
        weight_decay: Weight decay coefficient for the decay group.

    Returns:
        List of two param group dicts, ready to pass to an optimizer.

    Example::

        groups = get_param_groups(model, weight_decay=0.1)
        optimizer = torch.optim.AdamW(groups, lr=3e-4)
    """
    decay_params    : list = []
    no_decay_params : list = []

    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        # No decay: 1-D params (biases, norms), embedding matrices
        if param.dim() < 2 or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def count_param_groups(groups: list[dict]) -> dict[str, int]:
    """Return parameter counts per group."""
    return {
        "decay":    sum(p.numel() for p in groups[0]["params"]),
        "no_decay": sum(p.numel() for p in groups[1]["params"]),
    }
''')
commit("feat: add get_param_groups() — split params into decay/no-decay for AdamW")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — get_optimizer() factory
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/optimizer.py", '''\
"""
nanomind/optim/optimizer.py — Optimizer factory for NanoMind.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.optim.param_groups import get_param_groups


def get_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Build an optimizer for NanoMind with proper weight decay grouping.

    Automatically separates parameters into decay and no-decay groups
    so that biases, norms, and embeddings are not penalized.

    Args:
        model:          The model to optimize.
        lr:             Peak learning rate.
        weight_decay:   Coefficient applied to decayed parameters.
        betas:          AdamW beta coefficients ``(beta1, beta2)``.
        eps:            Numerical stability epsilon.
        optimizer_type: ``"adamw"`` (default) or ``"sgd"``.

    Returns:
        Configured :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If an unsupported optimizer type is requested.
    """
    groups = get_param_groups(model, weight_decay=weight_decay)

    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)

    raise ValueError(
        f"Unknown optimizer '{optimizer_type}'. Choose 'adamw' or 'sgd'."
    )
''')
commit("feat: add get_optimizer() factory — AdamW/SGD with proper weight decay grouping")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — BaseLRSchedule abstract class
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/schedules.py", '''\
"""
nanomind/optim/schedules.py — Learning rate schedule functions for NanoMind.

All schedules are implemented as plain callables: ``lr = schedule(step)``.
This keeps them independent of any optimizer or framework scheduler,
making them easy to test and compose.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRSchedule(ABC):
    """
    Abstract base class for all NanoMind LR schedules.

    Subclasses implement ``__call__(step) -> float``.
    """

    @abstractmethod
    def __call__(self, step: int) -> float:
        """Return the learning rate for the given training step."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
''')
commit("feat: add LRSchedule abstract base class — callable learning rate interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — ConstantLR
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class ConstantLR(LRSchedule):
    """
    Constant learning rate — returns the same LR for every step.

    Args:
        lr: The fixed learning rate.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, step: int) -> float:
        return self.lr

    def __repr__(self) -> str:
        return f"ConstantLR(lr={self.lr})"
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add ConstantLR schedule — fixed learning rate throughout training")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — LinearWarmup
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class LinearWarmup(LRSchedule):
    """
    Linear warmup from 0 to ``max_lr`` over ``warmup_steps`` steps.

    After warmup, delegates to ``post_warmup_schedule`` (or holds ``max_lr``).

    Args:
        max_lr:                Peak learning rate after warmup.
        warmup_steps:          Number of warmup steps.
        post_warmup_schedule:  Optional schedule called after warmup.
    """

    def __init__(
        self,
        max_lr: float,
        warmup_steps: int,
        post_warmup_schedule: LRSchedule | None = None,
    ) -> None:
        self.max_lr               = max_lr
        self.warmup_steps         = warmup_steps
        self.post_warmup_schedule = post_warmup_schedule

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear ramp: 0 -> max_lr
            return self.max_lr * (step + 1) / self.warmup_steps
        if self.post_warmup_schedule is not None:
            return self.post_warmup_schedule(step)
        return self.max_lr

    def __repr__(self) -> str:
        return (
            f"LinearWarmup(max_lr={self.max_lr}, "
            f"warmup_steps={self.warmup_steps})"
        )
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add LinearWarmup schedule — linear ramp from 0 to max_lr over warmup steps")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — CosineDecay
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class CosineDecay(LRSchedule):
    """
    Cosine annealing from ``max_lr`` down to ``min_lr`` over ``total_steps``.

    Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t/T))

    Args:
        max_lr:       Starting (peak) learning rate.
        min_lr:       Minimum learning rate (floor).
        total_steps:  Total number of steps for the decay.
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        total_steps: int,
    ) -> None:
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"CosineDecay(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"total_steps={self.total_steps})"
        )
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add CosineDecay schedule — cosine annealing from max_lr to min_lr")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — WarmupCosine (combined)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class WarmupCosine(LRSchedule):
    """
    Linear warmup followed by cosine decay — the default NanoMind schedule.

    This is equivalent to composing :class:`LinearWarmup` with
    :class:`CosineDecay`, but provided as a single convenience class.

    Args:
        max_lr:       Peak learning rate (after warmup, before decay).
        min_lr:       Minimum learning rate at end of cosine decay.
        warmup_steps: Number of linear warmup steps.
        total_steps:  Total training steps (including warmup).
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self._cosine      = CosineDecay(max_lr, min_lr, total_steps - warmup_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        return self._cosine(step - self.warmup_steps)

    def __repr__(self) -> str:
        return (
            f"WarmupCosine(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup={self.warmup_steps}, total={self.total_steps})"
        )
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add WarmupCosine schedule — linear warmup + cosine decay (default schedule)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — LinearDecay schedule
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class LinearDecay(LRSchedule):
    """
    Linear decay from ``max_lr`` to ``min_lr`` over ``total_steps``.

    Args:
        max_lr:      Starting learning rate.
        min_lr:      Ending learning rate.
        total_steps: Total number of decay steps.
    """

    def __init__(self, max_lr: float, min_lr: float, total_steps: int) -> None:
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return self.max_lr - progress * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"LinearDecay(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"total_steps={self.total_steps})"
        )
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add LinearDecay schedule — linear interpolation from max_lr to min_lr")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — WarmupLinear schedule
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

class WarmupLinear(LRSchedule):
    """
    Linear warmup followed by linear decay.

    Args:
        max_lr:       Peak learning rate after warmup.
        min_lr:       Minimum LR at end of decay.
        warmup_steps: Number of warmup steps.
        total_steps:  Total training steps (warmup + decay).
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        decay_step  = step - self.warmup_steps
        if decay_step >= decay_steps:
            return self.min_lr
        progress = decay_step / decay_steps
        return self.max_lr - progress * (self.max_lr - self.min_lr)

    def __repr__(self) -> str:
        return (
            f"WarmupLinear(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup={self.warmup_steps}, total={self.total_steps})"
        )
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add WarmupLinear schedule — linear warmup then linear decay")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — get_lr_scheduler() factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/optim/schedules.py")
src += '''

# ── Schedule factory ──────────────────────────────────────────────────────────

_SCHEDULE_REGISTRY: dict[str, type[LRSchedule]] = {
    "constant":      ConstantLR,
    "cosine":        CosineDecay,
    "warmup_cosine": WarmupCosine,
    "linear":        LinearDecay,
    "warmup_linear": WarmupLinear,
}


def get_lr_scheduler(name: str, **kwargs) -> LRSchedule:
    """
    Build an LR schedule by name.

    Args:
        name:    Schedule name. One of:
                 ``"constant"``, ``"cosine"``, ``"warmup_cosine"``,
                 ``"linear"``, ``"warmup_linear"``.
        **kwargs: Arguments forwarded to the schedule constructor.

    Returns:
        A callable :class:`LRSchedule` instance.

    Raises:
        ValueError: If the name is not recognised.

    Example::

        sched = get_lr_scheduler(
            "warmup_cosine",
            max_lr=3e-4, min_lr=3e-5,
            warmup_steps=100, total_steps=5000,
        )
        lr = sched(step=250)
    """
    key = name.lower().replace("-", "_")
    if key not in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. Available: {sorted(_SCHEDULE_REGISTRY)}"
        )
    return _SCHEDULE_REGISTRY[key](**kwargs)


def list_schedules() -> list[str]:
    """Return sorted list of all registered LR schedule names."""
    return sorted(_SCHEDULE_REGISTRY)
'''
write("nanomind/optim/schedules.py", src)
commit("feat: add get_lr_scheduler() factory and schedule registry")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — gradient norm tracking utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/grad_utils.py", '''\
"""
nanomind/optim/grad_utils.py — Gradient analysis utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the global gradient norm across all parameters.

    Equivalent to :func:`torch.nn.utils.clip_grad_norm_` but without clipping.
    Useful for monitoring gradient health during training.

    Args:
        model:     The model whose gradients to measure.
        norm_type: The norm type (default: L2).

    Returns:
        Total gradient norm as a float. Returns 0.0 if no gradients exist.
    """
    params_with_grad = [
        p for p in model.parameters()
        if p.grad is not None
    ]
    if not params_with_grad:
        return 0.0

    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type)
            for p in params_with_grad
        ]),
        norm_type,
    )
    return total_norm.item()


def get_grad_stats(model: nn.Module) -> dict[str, float]:
    """
    Compute gradient statistics for debugging.

    Returns:
        Dict with ``"max"``, ``"min"``, ``"mean"``, ``"l2_norm"`` values.
    """
    grads = [
        p.grad.detach().abs()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not grads:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "l2_norm": 0.0}

    all_grads = torch.cat([g.flatten() for g in grads])
    return {
        "max":     all_grads.max().item(),
        "min":     all_grads.min().item(),
        "mean":    all_grads.mean().item(),
        "l2_norm": compute_grad_norm(model),
    }
''')
commit("feat: add compute_grad_norm() and get_grad_stats() gradient analysis utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — optimizer state summary
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/summary.py", '''\
"""
nanomind/optim/summary.py — Optimizer and schedule summary utilities.
"""

from __future__ import annotations

import torch


def optimizer_summary(optimizer: torch.optim.Optimizer) -> str:
    """
    Return a human-readable summary of the optimizer and its param groups.

    Args:
        optimizer: Any :class:`torch.optim.Optimizer`.

    Returns:
        Multi-line string summary.
    """
    lines = [f"Optimizer: {type(optimizer).__name__}"]
    for i, pg in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        lr       = pg.get("lr", "?")
        wd       = pg.get("weight_decay", "?")
        lines.append(
            f"  Group {i}: {n_params:>10,} params | "
            f"lr={lr} | wd={wd}"
        )
    total = sum(
        sum(p.numel() for p in pg["params"])
        for pg in optimizer.param_groups
    )
    lines.append(f"  Total: {total:,} params")
    return "\n".join(lines)


def schedule_preview(
    schedule,
    total_steps: int,
    n_points: int = 10,
) -> list[tuple[int, float]]:
    """
    Preview an LR schedule at evenly spaced steps.

    Args:
        schedule:    A callable ``schedule(step) -> float``.
        total_steps: Total number of training steps.
        n_points:    Number of steps to sample.

    Returns:
        List of ``(step, lr)`` tuples.
    """
    steps = [int(i * total_steps / (n_points - 1)) for i in range(n_points)]
    return [(s, schedule(s)) for s in steps]
''')
commit("feat: add optimizer_summary() and schedule_preview() diagnostic utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update optim __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/optim/__init__.py", '''\
"""NanoMind optimizer and LR scheduler sub-package.

Optimizers:
    - :func:`get_optimizer`   — build AdamW/SGD with proper weight decay grouping
    - :func:`get_param_groups`— split params into decay / no-decay

LR Schedules (all callable: ``lr = schedule(step)``):
    - :class:`ConstantLR`     — fixed LR
    - :class:`LinearWarmup`   — linear ramp to max_lr
    - :class:`CosineDecay`    — cosine annealing
    - :class:`WarmupCosine`   — warmup + cosine (default)
    - :class:`LinearDecay`    — linear annealing
    - :class:`WarmupLinear`   — warmup + linear decay
    - :func:`get_lr_scheduler`— build a schedule by name
    - :func:`list_schedules`  — list all registered schedule names

Utilities:
    - :func:`compute_grad_norm` — global gradient L2 norm
    - :func:`get_grad_stats`    — gradient min/max/mean/norm
    - :func:`optimizer_summary` — human-readable optimizer info
    - :func:`schedule_preview`  — preview LR values at N steps
"""

from nanomind.optim.optimizer import get_optimizer
from nanomind.optim.param_groups import get_param_groups, count_param_groups
from nanomind.optim.schedules import (
    LRSchedule,
    ConstantLR,
    LinearWarmup,
    CosineDecay,
    WarmupCosine,
    LinearDecay,
    WarmupLinear,
    get_lr_scheduler,
    list_schedules,
)
from nanomind.optim.grad_utils import compute_grad_norm, get_grad_stats
from nanomind.optim.summary import optimizer_summary, schedule_preview

__all__ = [
    "get_optimizer",
    "get_param_groups",
    "count_param_groups",
    "LRSchedule",
    "ConstantLR",
    "LinearWarmup",
    "CosineDecay",
    "WarmupCosine",
    "LinearDecay",
    "WarmupLinear",
    "get_lr_scheduler",
    "list_schedules",
    "compute_grad_norm",
    "get_grad_stats",
    "optimizer_summary",
    "schedule_preview",
]
''')
commit("refactor: export all optim components from nanomind/optim/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: param group splitting
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_optim.py", '''\
"""
tests/test_optim.py — Tests for NanoMind optimizers and LR schedules.
"""

import pytest
import math
import torch

from nanomind.model import NanoMind, ModelConfig
from nanomind.optim import (
    get_optimizer,
    get_param_groups,
    count_param_groups,
    get_lr_scheduler,
    list_schedules,
    WarmupCosine,
    CosineDecay,
    LinearWarmup,
    LinearDecay,
    WarmupLinear,
    ConstantLR,
    compute_grad_norm,
    get_grad_stats,
    optimizer_summary,
    schedule_preview,
)

CFG = ModelConfig(
    vocab_size=32, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


@pytest.fixture
def model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(CFG)


# ── Param groups ──────────────────────────────────────────────────────────────

class TestParamGroups:
    def test_returns_two_groups(self, model):
        groups = get_param_groups(model)
        assert len(groups) == 2

    def test_decay_group_has_weight_decay(self, model):
        groups = get_param_groups(model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1

    def test_no_decay_group_has_zero_wd(self, model):
        groups = get_param_groups(model)
        assert groups[1]["weight_decay"] == 0.0

    def test_all_params_covered(self, model):
        groups  = get_param_groups(model)
        counts  = count_param_groups(groups)
        total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grouped = counts["decay"] + counts["no_decay"]
        assert grouped == total

    def test_no_duplicates(self, model):
        groups = get_param_groups(model)
        all_ids = [id(p) for g in groups for p in g["params"]]
        assert len(all_ids) == len(set(all_ids))
''')
commit("test: add param group splitting tests (two groups, WD, coverage, no duplicates)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: cosine decay schedule
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_optim.py")
src += '''

# ── CosineDecay ───────────────────────────────────────────────────────────────

class TestCosineDecay:
    def test_starts_at_max_lr(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert abs(s(0) - 1e-3) < 1e-9

    def test_ends_at_min_lr(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert abs(s(100) - 1e-5) < 1e-9

    def test_monotonically_decreasing(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        lrs = [s(i) for i in range(101)]
        assert all(lrs[i] >= lrs[i+1] for i in range(100))

    def test_midpoint_near_average(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        mid = s(50)
        avg = (1e-3 + 1e-5) / 2
        assert abs(mid - avg) < 1e-5
'''
write("tests/test_optim.py", src)
commit("test: add CosineDecay schedule tests (start, end, monotone, midpoint)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: linear warmup schedule
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_optim.py")
src += '''

# ── LinearWarmup ──────────────────────────────────────────────────────────────

class TestLinearWarmup:
    def test_starts_near_zero(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert s(0) < 1e-3

    def test_reaches_max_at_warmup_end(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert abs(s(9) - 1e-3) < 1e-9

    def test_monotonically_increasing_during_warmup(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        lrs = [s(i) for i in range(10)]
        assert all(lrs[i] < lrs[i+1] for i in range(9))

    def test_constant_after_warmup_no_schedule(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert s(10) == 1e-3
        assert s(100) == 1e-3


# ── WarmupCosine ──────────────────────────────────────────────────────────────

class TestWarmupCosine:
    def test_starts_near_zero(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert s(0) < 1e-3

    def test_reaches_max_at_warmup_end(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert abs(s(9) - 1e-3) < 1e-9

    def test_ends_at_min_lr(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert abs(s(100) - 1e-5) < 1e-9

    def test_overall_shape(self):
        s    = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        lrs  = [s(i) for i in range(101)]
        # Warmup phase: increasing
        assert lrs[0] < lrs[9]
        # Post-warmup: decreasing
        assert lrs[10] > lrs[100]
'''
write("tests/test_optim.py", src)
commit("test: add LinearWarmup and WarmupCosine schedule tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: factory + other schedules
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_optim.py")
src += '''

# ── Factory ───────────────────────────────────────────────────────────────────

class TestScheduleFactory:
    def test_get_constant(self):
        s = get_lr_scheduler("constant", lr=1e-3)
        assert isinstance(s, ConstantLR)
        assert s(0) == 1e-3
        assert s(999) == 1e-3

    def test_get_cosine(self):
        s = get_lr_scheduler("cosine", max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert isinstance(s, CosineDecay)

    def test_get_warmup_cosine(self):
        s = get_lr_scheduler("warmup_cosine", max_lr=1e-3, min_lr=1e-5,
                             warmup_steps=10, total_steps=100)
        assert isinstance(s, WarmupCosine)

    def test_get_linear(self):
        s = get_lr_scheduler("linear", max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert isinstance(s, LinearDecay)

    def test_get_warmup_linear(self):
        s = get_lr_scheduler("warmup_linear", max_lr=1e-3, min_lr=1e-5,
                             warmup_steps=10, total_steps=100)
        assert isinstance(s, WarmupLinear)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_lr_scheduler("nosuchthing", lr=1e-3)

    def test_list_schedules(self):
        names = list_schedules()
        assert "warmup_cosine" in names
        assert "cosine" in names
'''
write("tests/test_optim.py", src)
commit("test: add get_lr_scheduler factory and list_schedules tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: grad utils + optimizer + summary
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_optim.py")
src += '''

# ── Grad utilities ────────────────────────────────────────────────────────────

class TestGradUtils:
    def test_grad_norm_zero_before_backward(self, model):
        assert compute_grad_norm(model) == 0.0

    def test_grad_norm_positive_after_backward(self, model):
        idx  = torch.randint(0, CFG.vocab_size, (2, 8))
        tgt  = torch.randint(0, CFG.vocab_size, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        assert compute_grad_norm(model) > 0.0

    def test_grad_stats_keys(self, model):
        idx  = torch.randint(0, CFG.vocab_size, (2, 8))
        tgt  = torch.randint(0, CFG.vocab_size, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        stats = get_grad_stats(model)
        assert "max" in stats and "l2_norm" in stats


# ── get_optimizer ─────────────────────────────────────────────────────────────

class TestGetOptimizer:
    def test_returns_adamw(self, model):
        opt = get_optimizer(model, optimizer_type="adamw")
        assert isinstance(opt, torch.optim.AdamW)

    def test_returns_sgd(self, model):
        opt = get_optimizer(model, optimizer_type="sgd")
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self, model):
        with pytest.raises(ValueError):
            get_optimizer(model, optimizer_type="rmsprop")

    def test_two_param_groups(self, model):
        opt = get_optimizer(model)
        assert len(opt.param_groups) == 2


# ── Summary utilities ─────────────────────────────────────────────────────────

class TestSummaryUtils:
    def test_optimizer_summary_string(self, model):
        opt = get_optimizer(model)
        s   = optimizer_summary(opt)
        assert "AdamW" in s
        assert "Group" in s

    def test_schedule_preview_length(self):
        s    = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        pts  = schedule_preview(s, total_steps=100, n_points=5)
        assert len(pts) == 5
        assert pts[0][0] == 0
        assert pts[-1][0] == 100
'''
write("tests/test_optim.py", src)
commit("test: add grad_utils, get_optimizer, optimizer_summary, schedule_preview tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 9 | Optimizers & LR scheduling | 🔜 |",
    "| 9 | Optimizers & LR scheduling | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Training: Trainer loop with AMP, grad accum, gradient clip, early stop, estimate_loss (Day 8)",
    "- Training: Trainer loop with AMP, grad accum, gradient clip, early stop, estimate_loss (Day 8)\n- Optimizers: AdamW factory, param groups, WarmupCosine/Cosine/Linear/WarmupLinear schedules (Day 9)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 9 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 9 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 9 COMPLETE ===")
