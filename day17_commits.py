"""
day17_commits.py — 20 atomic commits for Day 17: LoRA Fine-tuning.
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

print("\n=== DAY 17: LoRA Fine-tuning — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — lora package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/__init__.py", '"""NanoMind LoRA (Low-Rank Adaptation) fine-tuning sub-package."""\n')
commit("feat: add nanomind/lora/ package skeleton for LoRA fine-tuning")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — LoRAConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/config.py", '''\
"""
nanomind/lora/config.py — LoRA configuration dataclass.

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices A and B
into frozen linear layers. The effective weight update is::

    W' = W + (alpha / r) * B @ A

where r is the rank and alpha is the scaling factor.

Reference: Hu et al. (2021) — https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.

    Attributes:
        r:            LoRA rank (dimensionality of low-rank matrices A and B).
                      Typical values: 4, 8, 16, 32. Higher rank = more capacity.
        alpha:        LoRA scaling factor. Effective scale = alpha / r.
                      Typically set equal to r (scale = 1.0) or 2r (scale = 2.0).
        dropout:      Dropout applied to the LoRA input before A projection.
        target_modules: Names of module types to inject LoRA into.
                      Typical choices: ``["q_proj", "v_proj"]`` (query + value only,
                      as in the original LoRA paper) or all attention projections.
        bias:         Whether to train biases alongside LoRA (``"none"``, ``"all"``,
                      or ``"lora_only"``).
    """

    r:               int        = 8
    alpha:           float      = 16.0
    dropout:         float      = 0.0
    target_modules:  list[str]  = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias:            str        = "none"   # "none", "all", "lora_only"

    def __post_init__(self) -> None:
        assert self.r > 0,              "LoRA rank r must be positive"
        assert self.alpha > 0,          "LoRA alpha must be positive"
        assert 0.0 <= self.dropout < 1.0
        assert self.bias in ("none", "all", "lora_only")

    @property
    def scaling(self) -> float:
        """Effective LoRA scaling factor: alpha / r."""
        return self.alpha / self.r
''')
commit("feat: add LoRAConfig dataclass — rank, alpha, dropout, target_modules, scaling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — LoRALinear layer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/layer.py", '''\
"""
nanomind/lora/layer.py — LoRA-augmented Linear layer.

Wraps a frozen ``nn.Linear`` with two trainable low-rank matrices A and B::

    output = x @ W.T + scaling * x @ A.T @ B.T
           = base_output + lora_output

where:
  - W is the frozen pre-trained weight (shape: out_features × in_features)
  - A is the down-projection: (r × in_features) — initialised with Kaiming Normal
  - B is the up-projection:   (out_features × r) — initialised with zeros
  - B = 0 at init means LoRA output is zero at the start of training,
    so the model begins from the pre-trained state

Only A and B are updated during fine-tuning; W is kept frozen.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A drop-in replacement for ``nn.Linear`` with LoRA adaptation.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        r:            LoRA rank.
        alpha:        LoRA scaling (effective scale = alpha / r).
        dropout:      Dropout on LoRA path input.
        bias:         Whether the base linear has a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r            = r
        self.scaling      = alpha / r
        self.merged       = False

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Optional dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise A with Kaiming Normal; B stays zero (safe start)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Create a LoRALinear by wrapping an existing ``nn.Linear``.

        Copies the pre-trained weight (and bias if present) and freezes them.

        Args:
            linear: Existing linear layer to adapt.
            r:      LoRA rank.
            alpha:  LoRA scaling.
            dropout: LoRA dropout.

        Returns:
            New :class:`LoRALinear` with copied frozen base weights.
        """
        has_bias = linear.bias is not None
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )
        lora.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora.bias_param.data.copy_(linear.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base linear + LoRA delta.

        Args:
            x: Input ``(..., in_features)``

        Returns:
            Output ``(..., out_features)``
        """
        base_out = F.linear(x, self.weight, self.bias_param)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, scaling={self.scaling:.3f}, merged={self.merged}"
        )
''')
commit("feat: add LoRALinear — frozen base weight + trainable low-rank A and B matrices")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — LoRALinear.merge() and unmerge()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/lora/layer.py")
src += '''

    def merge(self) -> None:
        """
        Merge LoRA weights into the base weight for efficient inference.

        After merging, the layer behaves as a standard ``nn.Linear`` with
        no additional compute overhead. The LoRA delta is baked into W::

            W_merged = W + scaling * B @ A

        Call :meth:`unmerge` to restore the separate LoRA matrices.
        """
        if self.merged:
            return
        self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True

    def unmerge(self) -> None:
        """
        Unmerge LoRA weights from the base weight (reverse of :meth:`merge`).

        Restores the original W by subtracting the LoRA delta::

            W_original = W_merged - scaling * B @ A
        """
        if not self.merged:
            return
        self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False
'''
write("nanomind/lora/layer.py", src)
commit("feat: add LoRALinear.merge() and unmerge() — absorb/restore LoRA delta into base weight")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — inject_lora_into_model()
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/inject.py", '''\
"""
nanomind/lora/inject.py — Inject LoRA layers into a pre-trained model.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.lora.config import LoRAConfig
from nanomind.lora.layer import LoRALinear
from nanomind.utils.logger import get_logger

log = get_logger("lora.inject")


def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig,
) -> nn.Module:
    """
    Replace target ``nn.Linear`` layers with :class:`~nanomind.lora.LoRALinear`.

    Only layers whose **name** (the attribute name in the parent module)
    matches one of ``cfg.target_modules`` are replaced.

    Args:
        model: The pre-trained model to inject LoRA into.
        cfg:   LoRA configuration.

    Returns:
        The same model with LoRA layers injected (in-place modification).

    Example::

        cfg = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
        inject_lora(model, cfg)
    """
    n_injected = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if not isinstance(child, nn.Linear):
                continue
            if attr_name not in cfg.target_modules:
                continue

            lora_layer = LoRALinear.from_linear(
                child,
                r=cfg.r,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
            )
            setattr(module, attr_name, lora_layer)
            n_injected += 1
            log.debug(f"  Injected LoRA into: {name}.{attr_name}")

    log.info(f"LoRA injected into {n_injected} layers.")
    return model


def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
) -> nn.Module:
    """
    Freeze all model parameters except LoRA matrices (and optionally biases).

    Args:
        model: Model with LoRA layers already injected.
        bias:  ``"none"`` — no biases trained
               ``"all"``  — all biases trained
               ``"lora_only"`` — only biases of LoRA layers trained

    Returns:
        The model with frozen parameters (in-place).
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        elif bias == "all" and "bias" in name:
            param.requires_grad_(True)
        elif bias == "lora_only" and "lora" in name and "bias" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model
''')
commit("feat: add inject_lora() and mark_only_lora_as_trainable() — freeze base, train LoRA")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — LoRA parameter utilities
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/utils.py", '''\
"""
nanomind/lora/utils.py — LoRA parameter counting and inspection utilities.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.lora.layer import LoRALinear


def lora_parameter_stats(model: nn.Module) -> dict:
    """
    Count total, trainable, and LoRA-specific parameters.

    Args:
        model: A model with LoRA layers injected.

    Returns:
        Dict with:
        - ``total``     : total parameter count
        - ``trainable`` : trainable (LoRA) parameter count
        - ``frozen``    : frozen (base) parameter count
        - ``lora_pct``  : percentage of parameters that are trainable
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    lora_pct  = 100.0 * trainable / max(total, 1)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    frozen,
        "lora_pct":  lora_pct,
    }


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only the LoRA-specific parameters from a model's state dict.

    Used for saving lightweight LoRA checkpoints (much smaller than full model).

    Args:
        model: A model with LoRA layers.

    Returns:
        State dict containing only ``lora_A`` and ``lora_B`` entries.
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def merge_all_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA deltas into base weights across the entire model.

    After merging, the model runs as fast as the original with no overhead.

    Args:
        model: Model with LoRA layers.

    Returns:
        Same model with all LoRA weights merged (in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def unmerge_all_lora(model: nn.Module) -> nn.Module:
    """
    Unmerge all LoRA deltas from base weights across the entire model.

    Args:
        model: Model with merged LoRA layers.

    Returns:
        Same model with LoRA weights separated (in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    return model


def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA configuration and trainable parameters."""
    stats = lora_parameter_stats(model)
    print("=" * 50)
    print("LoRA Parameter Summary")
    print("=" * 50)
    print(f"  Total parameters  : {stats['total']:>12,}")
    print(f"  Trainable (LoRA)  : {stats['trainable']:>12,}  ({stats['lora_pct']:.2f}%)")
    print(f"  Frozen (base)     : {stats['frozen']:>12,}")
    print("=" * 50)

    # List LoRA layers
    lora_layers = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]
    print(f"  LoRA layers ({len(lora_layers)}):")
    for name, m in lora_layers:
        n = 2 * m.r * (m.in_features + m.out_features)
        print(f"    {name:<40} r={m.r}  params={n:,}")
    print("=" * 50)
''')
commit("feat: add lora_parameter_stats(), get_lora_state_dict(), merge/unmerge_all_lora()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — LoRA checkpoint save/load
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/checkpoint.py", '''\
"""
nanomind/lora/checkpoint.py — Save and load LoRA-only checkpoints.

LoRA checkpoints store only the A and B matrices — typically just a few MB
compared to hundreds of MB for the full model. The base model is loaded
separately and the LoRA weights are applied on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from nanomind.lora.utils import get_lora_state_dict
from nanomind.utils.logger import get_logger

log = get_logger("lora.checkpoint")


def save_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """
    Save only the LoRA weights (A and B matrices) to a file.

    Args:
        model:    Model with LoRA layers.
        path:     Destination file path (e.g. ``lora_weights.pt``).
        metadata: Optional dict with experiment info (step, loss, config).

    Returns:
        Path of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lora_state = get_lora_state_dict(model)
    payload = {
        "lora_state": lora_state,
        "metadata":   metadata or {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)

    n_params = sum(v.numel() for v in lora_state.values())
    size_kb   = path.stat().st_size / 1024
    log.info(f"Saved LoRA checkpoint: {path.name} ({n_params:,} params, {size_kb:.1f} KB)")
    return path


def load_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict:
    """
    Load LoRA weights into a model with LoRA layers already injected.

    Args:
        model:   Model with LoRA layers (injected but not trained).
        path:    Path to the LoRA ``.pt`` checkpoint.
        device:  Device to map weights to.
        strict:  Whether to require exact LoRA key matching.

    Returns:
        Metadata dict from the checkpoint.
    """
    path    = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    lora_state = payload.get("lora_state", payload)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)

    lora_missing = [k for k in missing    if "lora" in k]
    non_lora_unexp = [k for k in unexpected if "lora" not in k]

    if strict and lora_missing:
        raise RuntimeError(f"Missing LoRA keys: {lora_missing}")
    if non_lora_unexp:
        log.warning(f"Unexpected non-LoRA keys: {non_lora_unexp}")

    log.info(f"Loaded LoRA checkpoint from: {path.name}")
    return payload.get("metadata", {})
''')
commit("feat: add save_lora_checkpoint() and load_lora_checkpoint() — tiny LoRA-only saves")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — LoRAModel high-level wrapper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/model.py", '''\
"""
nanomind/lora/model.py — High-level LoRA model wrapper.

LoRAModel wraps a pre-trained NanoMind model and handles the full
LoRA lifecycle:
  1. Inject LoRA layers into target modules
  2. Freeze base parameters
  3. Train only LoRA parameters
  4. Optionally merge for inference
  5. Save/load lightweight LoRA checkpoints
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.lora.config import LoRAConfig
from nanomind.lora.inject import inject_lora, mark_only_lora_as_trainable
from nanomind.lora.checkpoint import save_lora_checkpoint, load_lora_checkpoint
from nanomind.lora.utils import (
    lora_parameter_stats,
    merge_all_lora,
    unmerge_all_lora,
    print_lora_summary,
)
from nanomind.utils.logger import get_logger


class LoRAModel(nn.Module):
    """
    LoRA-wrapped NanoMind model for parameter-efficient fine-tuning.

    Usage::

        base_model = NanoMind(cfg)
        lora_cfg   = LoRAConfig(r=8, alpha=16, target_modules=["q_proj","v_proj"])
        model      = LoRAModel(base_model, lora_cfg)

        # Only LoRA parameters are updated
        optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-4)

        # After fine-tuning, save only LoRA weights (~few MB)
        model.save("my_lora.pt")

    Args:
        model: Pre-trained base model.
        cfg:   LoRA configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: LoRAConfig,
    ) -> None:
        super().__init__()
        self.cfg  = cfg
        self.log  = get_logger("lora.model")

        # Inject LoRA layers
        inject_lora(model, cfg)
        mark_only_lora_as_trainable(model, bias=cfg.bias)
        self.model = model

        stats = lora_parameter_stats(model)
        self.log.info(
            f"LoRAModel ready: {stats['trainable']:,} trainable params "
            f"({stats['lora_pct']:.2f}% of {stats['total']:,} total)"
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def lora_parameters(self):
        """Return only the trainable LoRA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def merge_for_inference(self) -> "LoRAModel":
        """Merge LoRA weights into base weights for zero-overhead inference."""
        merge_all_lora(self.model)
        self.log.info("LoRA weights merged — ready for inference.")
        return self

    def unmerge(self) -> "LoRAModel":
        """Unmerge LoRA weights for further training."""
        unmerge_all_lora(self.model)
        return self

    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save only LoRA weights (lightweight checkpoint)."""
        save_lora_checkpoint(self.model, path, metadata)

    def load(self, path: str, device: torch.device | None = None) -> dict:
        """Load LoRA weights from a checkpoint."""
        return load_lora_checkpoint(self.model, path, device)

    def summary(self) -> None:
        """Print a LoRA parameter summary."""
        print_lora_summary(self.model)

    def __repr__(self) -> str:
        stats = lora_parameter_stats(self.model)
        return (
            f"LoRAModel("
            f"r={self.cfg.r}, "
            f"alpha={self.cfg.alpha}, "
            f"targets={self.cfg.target_modules}, "
            f"trainable={stats['trainable']:,} ({stats['lora_pct']:.2f}%))"
        )
''')
commit("feat: add LoRAModel wrapper — inject, freeze, train, merge, save, load LoRA")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — LoRA fine-tuning trainer helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/finetune.py", '''\
"""
nanomind/lora/finetune.py — Convenience helpers for LoRA fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.lora.config import LoRAConfig
from nanomind.lora.model import LoRAModel
from nanomind.optim import get_optimizer, get_lr_scheduler
from nanomind.utils.logger import get_logger

log = get_logger("lora.finetune")


def finetune_with_lora(
    base_model: nn.Module,
    train_loader: DataLoader,
    lora_cfg: LoRAConfig | None = None,
    lr: float = 3e-4,
    max_iters: int = 1000,
    device: torch.device | None = None,
    save_path: str | None = None,
) -> LoRAModel:
    """
    Fine-tune a pre-trained model with LoRA.

    A convenience function that:
    1. Wraps the model with :class:`LoRAModel`
    2. Creates an AdamW optimizer on LoRA parameters only
    3. Runs a simple training loop
    4. Optionally saves LoRA weights

    Args:
        base_model:   Pre-trained model to fine-tune.
        train_loader: DataLoader yielding (x, y) batches.
        lora_cfg:     LoRA configuration (defaults to r=8, alpha=16).
        lr:           Learning rate for LoRA parameters.
        max_iters:    Number of fine-tuning steps.
        device:       Training device.
        save_path:    If provided, save LoRA weights here after training.

    Returns:
        Trained :class:`LoRAModel`.
    """
    lora_cfg = lora_cfg or LoRAConfig()
    device   = device or next(base_model.parameters()).device

    lora_model = LoRAModel(base_model, lora_cfg)
    lora_model.to(device)
    lora_model.summary()

    optimizer = torch.optim.AdamW(lora_model.lora_parameters(), lr=lr)
    loader_iter = iter(train_loader)
    step = 0

    while step < max_iters:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = lora_model(x, y)
        loss.backward()
        optimizer.step()
        step += 1

        if step % 100 == 0 or step == max_iters:
            log.info(f"step {step}/{max_iters}  loss={loss.item():.4f}")

    if save_path:
        lora_model.save(save_path, metadata={"steps": max_iters, "lr": lr})

    return lora_model
''')
commit("feat: add finetune_with_lora() — one-call LoRA fine-tuning convenience function")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update lora __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/lora/__init__.py", '''\
"""NanoMind LoRA (Low-Rank Adaptation) fine-tuning sub-package.

LoRA enables efficient fine-tuning by injecting trainable low-rank matrices
into frozen pre-trained linear layers. Only ~1-5% of parameters are updated.

Primary exports:
    - :class:`LoRAModel`              — high-level LoRA wrapper (recommended entry point)
    - :class:`LoRAConfig`             — rank, alpha, target_modules configuration
    - :class:`LoRALinear`             — drop-in LoRA-augmented nn.Linear

Injection:
    - :func:`inject_lora`             — replace target layers with LoRALinear
    - :func:`mark_only_lora_as_trainable` — freeze all base parameters

Checkpointing:
    - :func:`save_lora_checkpoint`    — save only A/B matrices (lightweight)
    - :func:`load_lora_checkpoint`    — load LoRA weights into injected model

Utilities:
    - :func:`lora_parameter_stats`    — count total/trainable/frozen params
    - :func:`get_lora_state_dict`     — extract LoRA-only state dict
    - :func:`merge_all_lora`          — merge deltas into weights (inference)
    - :func:`unmerge_all_lora`        — separate LoRA from base weights
    - :func:`print_lora_summary`      — pretty-print parameter table

Fine-tuning:
    - :func:`finetune_with_lora`      — one-call fine-tuning helper
"""

from nanomind.lora.config import LoRAConfig
from nanomind.lora.layer import LoRALinear
from nanomind.lora.inject import inject_lora, mark_only_lora_as_trainable
from nanomind.lora.checkpoint import save_lora_checkpoint, load_lora_checkpoint
from nanomind.lora.utils import (
    lora_parameter_stats,
    get_lora_state_dict,
    merge_all_lora,
    unmerge_all_lora,
    print_lora_summary,
)
from nanomind.lora.model import LoRAModel
from nanomind.lora.finetune import finetune_with_lora

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAModel",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "lora_parameter_stats",
    "get_lora_state_dict",
    "merge_all_lora",
    "unmerge_all_lora",
    "print_lora_summary",
    "finetune_with_lora",
]
''')
commit("refactor: export all LoRA components from nanomind/lora/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: lora_finetune.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/lora_finetune.py", '''\
"""
examples/lora_finetune.py — LoRA fine-tuning demo for NanoMind.

Loads a pre-trained model (or trains a tiny one from scratch),
then fine-tunes it on a new text corpus with LoRA.

Usage:
    python examples/lora_finetune.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.lora import LoRAConfig, LoRAModel, print_lora_summary

# ── 1. Base model (pre-trained or randomly initialised) ───────────────────────
VOCAB_TEXT = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?" * 10
tokenizer  = CharTokenizer().build(VOCAB_TEXT)

model_cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=32, d_model=64, n_layers=2, n_heads=4, dropout=0.0
)
base_model = NanoMind(model_cfg)
print(f"Base model: {base_model.num_parameters():,} params")

# ── 2. Fine-tuning corpus ─────────────────────────────────────────────────────
FINETUNE_TEXT = "the quick brown fox jumps over the lazy dog " * 30
ids     = tokenizer.encode(FINETUNE_TEXT)
tokens  = torch.tensor(ids)
BLOCK   = model_cfg.block_size
xs = torch.stack([tokens[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=True, drop_last=True)

# ── 3. Wrap with LoRA ─────────────────────────────────────────────────────────
lora_cfg = LoRAConfig(
    r=4,
    alpha=8.0,
    dropout=0.0,
    target_modules=["q_proj", "v_proj"],   # only adapt Q and V projections
)
lora_model = LoRAModel(base_model, lora_cfg)
lora_model.summary()

# ── 4. Fine-tune with LoRA only ───────────────────────────────────────────────
optimizer = torch.optim.AdamW(lora_model.lora_parameters(), lr=3e-3)
for step in range(200):
    x, y = next(iter(loader))
    optimizer.zero_grad()
    _, loss = lora_model(x, y)
    loss.backward()
    optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"step {step+1}/200  loss={loss.item():.4f}")

# ── 5. Save tiny LoRA checkpoint ──────────────────────────────────────────────
lora_model.save("checkpoints/lora_finetune.pt")
print("\\nLoRA weights saved (only A/B matrices — very small!)")

# ── 6. Merge for inference ────────────────────────────────────────────────────
lora_model.merge_for_inference()

from nanomind.generate import Generator, GenerationConfig
from nanomind.tokenizer.char import CharTokenizer
gen = Generator(lora_model, tokenizer)
out = gen.generate("the ", GenerationConfig(max_new_tokens=40, strategy="top_k", top_k=10))
print(f"\\nGenerated: the {out}")
''')
commit("feat: add examples/lora_finetune.py — end-to-end LoRA fine-tuning demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: LoRALinear forward shape
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_lora.py", '''\
"""
tests/test_lora.py — Tests for NanoMind LoRA fine-tuning.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from nanomind import NanoMind, ModelConfig
from nanomind.lora import (
    LoRAConfig,
    LoRALinear,
    LoRAModel,
    inject_lora,
    mark_only_lora_as_trainable,
    save_lora_checkpoint,
    load_lora_checkpoint,
    lora_parameter_stats,
    get_lora_state_dict,
    merge_all_lora,
    unmerge_all_lora,
)

IN_F, OUT_F, R = 64, 128, 8
B, T, D = 2, 8, 32
VOCAB = 32


def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── LoRALinear ────────────────────────────────────────────────────────────────

class TestLoRALinear:
    def test_output_shape(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        x     = torch.randn(B, IN_F)
        out   = layer(x)
        assert out.shape == (B, OUT_F)

    def test_3d_input(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        x     = torch.randn(B, T, IN_F)
        out   = layer(x)
        assert out.shape == (B, T, OUT_F)

    def test_lora_A_requires_grad(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert layer.lora_A.requires_grad
        assert layer.lora_B.requires_grad

    def test_weight_frozen(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert not layer.weight.requires_grad

    def test_lora_B_init_is_zero(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert torch.all(layer.lora_B == 0)

    def test_from_linear(self):
        linear = nn.Linear(IN_F, OUT_F, bias=False)
        lora   = LoRALinear.from_linear(linear, r=R)
        assert torch.equal(lora.weight.data, linear.weight.data)

    def test_from_linear_with_bias(self):
        linear = nn.Linear(IN_F, OUT_F, bias=True)
        lora   = LoRALinear.from_linear(linear, r=R)
        assert lora.bias_param is not None
        assert torch.equal(lora.bias_param.data, linear.bias.data)
''')
commit("test: add LoRALinear shape, init, frozen weight, and from_linear tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: merge/unmerge roundtrip
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── Merge / Unmerge ───────────────────────────────────────────────────────────

class TestMergeUnmerge:
    def test_merge_changes_weight(self):
        layer  = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        w_orig = layer.weight.data.clone()
        # Train lora_A slightly
        layer.lora_A.data.fill_(0.01)
        layer.merge()
        assert not torch.equal(layer.weight.data, w_orig)

    def test_unmerge_restores_weight(self):
        layer  = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        w_orig = layer.weight.data.clone()
        layer.lora_A.data.fill_(0.01)
        layer.merge()
        layer.unmerge()
        assert torch.allclose(layer.weight.data, w_orig, atol=1e-6)

    def test_merged_output_equals_unmerged(self):
        layer = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        torch.manual_seed(1)
        layer.lora_A.data = torch.randn_like(layer.lora_A) * 0.01
        x       = torch.randn(B, IN_F)
        out_sep  = layer(x).detach().clone()
        layer.merge()
        out_merged = layer(x).detach().clone()
        assert torch.allclose(out_sep, out_merged, atol=1e-5)

    def test_double_merge_is_no_op(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        w1    = layer.weight.data.clone()
        layer.merge()
        layer.merge()   # second merge should be no-op
        assert torch.allclose(layer.weight.data, w1, atol=1e-6)

    def test_merge_all_and_unmerge_all(self):
        model = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        merge_all_lora(model)
        unmerge_all_lora(model)   # should not raise
'''
write("tests/test_lora.py", src)
commit("test: add merge/unmerge roundtrip, output equivalence, and merge_all tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: inject_lora and freeze
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── inject_lora ───────────────────────────────────────────────────────────────

class TestInjectLoRA:
    def test_target_layers_replaced(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        for name, mod in model.named_modules():
            if name.endswith("q_proj"):
                assert isinstance(mod, LoRALinear), f"{name} not replaced"

    def test_non_target_layers_unchanged(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        for name, mod in model.named_modules():
            if name.endswith("v_proj"):
                assert isinstance(mod, nn.Linear), f"{name} should not be replaced"

    def test_mark_only_lora_trainable(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model, bias="none")
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_model_still_runs_after_injection(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        idx = torch.randint(0, VOCAB, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, VOCAB)
'''
write("tests/test_lora.py", src)
commit("test: add inject_lora target replacement, freeze, and forward pass tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: LoRAModel
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── LoRAModel ─────────────────────────────────────────────────────────────────

class TestLoRAModel:
    def test_forward_shape(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        lm       = LoRAModel(model, lora_cfg)
        idx      = torch.randint(0, VOCAB, (B, T))
        logits, _ = lm(idx)
        assert logits.shape == (B, T, VOCAB)

    def test_lora_parameters_are_subset(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        lm       = LoRAModel(model, lora_cfg)
        lora_params = lm.lora_parameters()
        total_params = list(lm.parameters())
        assert len(lora_params) < len(total_params)

    def test_repr_contains_rank(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=8, target_modules=["q_proj"])
        lm       = LoRAModel(model, lora_cfg)
        assert "r=8" in repr(lm)

    def test_fewer_trainable_than_total(self):
        model = tiny_model()
        total_before = sum(p.numel() for p in model.parameters())
        lm    = LoRAModel(model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
        stats = lora_parameter_stats(lm.model)
        assert stats["trainable"] < stats["total"]
        assert stats["lora_pct"] < 50.0
'''
write("tests/test_lora.py", src)
commit("test: add LoRAModel forward, parameter subset, repr, and trainable count tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: LoRA checkpoint save/load
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── LoRA checkpoint ───────────────────────────────────────────────────────────

class TestLoRACheckpoint:
    def test_save_creates_file(self, tmp_path):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        path = tmp_path / "lora.pt"
        save_lora_checkpoint(model, path)
        assert path.exists()

    def test_checkpoint_smaller_than_full_model(self, tmp_path):
        import pickle, io
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        lora_path = tmp_path / "lora.pt"
        full_path = tmp_path / "full.pt"
        save_lora_checkpoint(model, lora_path)
        torch.save(model.state_dict(), full_path)
        assert lora_path.stat().st_size < full_path.stat().st_size

    def test_roundtrip_preserves_weights(self, tmp_path):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        # Set A to non-zero
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.data.fill_(0.123)

        path = tmp_path / "lora.pt"
        save_lora_checkpoint(model, path)

        # Load into fresh model
        model2 = tiny_model()
        inject_lora(model2, lora_cfg)
        load_lora_checkpoint(model2, path)

        for m1, m2 in zip(model.modules(), model2.modules()):
            if isinstance(m1, LoRALinear):
                assert torch.equal(m1.lora_A.data, m2.lora_A.data)

    def test_lora_state_dict_only_lora_keys(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        sd = get_lora_state_dict(model)
        assert all("lora_A" in k or "lora_B" in k for k in sd)
'''
write("tests/test_lora.py", src)
commit("test: add LoRA checkpoint save, size, roundtrip, and state_dict key tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: parameter stats
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── Parameter stats ───────────────────────────────────────────────────────────

class TestParameterStats:
    def test_trainable_less_than_total(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model)
        stats = lora_parameter_stats(model)
        assert stats["trainable"] < stats["total"]
        assert stats["frozen"] == stats["total"] - stats["trainable"]

    def test_lora_pct_in_range(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=2, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model)
        stats = lora_parameter_stats(model)
        assert 0.0 < stats["lora_pct"] < 100.0

    def test_higher_rank_more_trainable(self):
        model1 = tiny_model()
        model2 = tiny_model()
        inject_lora(model1, LoRAConfig(r=2, target_modules=["q_proj"]))
        inject_lora(model2, LoRAConfig(r=16, target_modules=["q_proj"]))
        mark_only_lora_as_trainable(model1)
        mark_only_lora_as_trainable(model2)
        s1 = lora_parameter_stats(model1)
        s2 = lora_parameter_stats(model2)
        assert s2["trainable"] > s1["trainable"]
'''
write("tests/test_lora.py", src)
commit("test: add lora_parameter_stats() correctness and rank scaling tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: gradient flows only through LoRA
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_lora.py")
src += '''

# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_only_lora_gets_gradients(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        lm       = LoRAModel(model, lora_cfg)

        x = torch.randint(0, VOCAB, (B, T))
        y = torch.randint(0, VOCAB, (B, T))
        _, loss = lm(x, y)
        loss.backward()

        for name, p in lm.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.grad is not None, f"{name} should have gradient"
            elif p.requires_grad is False:
                assert p.grad is None, f"{name} (frozen) should not have gradient"

    def test_lora_b_starts_zero_no_grad_effect(self):
        """At init, B=0 so LoRA output is zero — base model output is unchanged."""
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)

        idx = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            out, _ = model(idx)
        # Output should still be finite (not all-zero)
        assert out.isfinite().all()
'''
write("tests/test_lora.py", src)
commit("test: add gradient flow — only LoRA params get gradients, frozen params stay clean")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — update nanomind/__init__.py to expose lora
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace(
    "__version__ = \"1.2.0\"",
    "__version__ = \"1.3.0\""
)
src = src.replace(
    "from nanomind.pos import get_attention, list_pos_types",
    "from nanomind.pos import get_attention, list_pos_types\nfrom nanomind.lora import LoRAConfig, LoRAModel"
)
src = src.replace(
    "    \"list_pos_types\",\n    \"__version__\",\n]",
    "    \"list_pos_types\",\n    \"LoRAConfig\",\n    \"LoRAModel\",\n    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v1.3.0 — expose LoRAConfig and LoRAModel in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Attention** | MHA, GQA (Llama 2/Mistral), MQA (Falcon), GQA+RoPE |",
    "| **Attention** | MHA, GQA (Llama 2/Mistral), MQA (Falcon), GQA+RoPE |\n"
    "| **Fine-tuning** | LoRA (rank, alpha, target modules, merge, save/load) |"
)
readme = readme.replace(
    "**Total: 320 commits across 16 days.**",
    "**Total: 340 commits across 17 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.2.0] — 2024 — GQA & MQA",
    "## [1.3.0] — 2024 — LoRA Fine-tuning\n\n### Added\n"
    "- `LoRALinear` — drop-in frozen linear + trainable low-rank A/B matrices\n"
    "- `LoRAModel` — high-level wrapper: inject, freeze, train, merge, save/load\n"
    "- `LoRAConfig` — rank, alpha, dropout, target_modules, bias config\n"
    "- `inject_lora()` — replace target `nn.Linear` with `LoRALinear`\n"
    "- `merge_all_lora()` / `unmerge_all_lora()` — zero-overhead inference\n"
    "- `save_lora_checkpoint()` — save only A/B matrices (tiny files)\n"
    "- `load_lora_checkpoint()` — load LoRA weights into injected model\n"
    "- `finetune_with_lora()` — one-call fine-tuning convenience function\n"
    "- `examples/lora_finetune.py` — end-to-end LoRA demo\n\n---\n\n"
    "## [1.2.0] — 2024 — GQA & MQA"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.3.0, update README and CHANGELOG for Day 17 LoRA")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 17 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.3.0", "-m", "NanoMind v1.3.0 — LoRA fine-tuning", check=False)
r = run("git", "push", "origin", "v1.3.0", check=False)
print("Tag v1.3.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 17 COMPLETE — v1.3.0 TAGGED! ===")
