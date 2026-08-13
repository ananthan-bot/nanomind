"""
day6_commits.py — 20 atomic commits for Day 6: Transformer Blocks.
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

print("\n=== DAY 6: Transformer Blocks — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — blocks package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/__init__.py", '"""NanoMind transformer blocks sub-package."""\n')
commit("feat: add nanomind/blocks/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — LayerNorm wrapper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/norms.py", '''\
"""
nanomind/blocks/norms.py — Normalization layers for NanoMind.

Provides LayerNorm and RMSNorm, along with a registry so the
normalization type can be swapped via config without code changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (Ba et al., 2016).

    Thin wrapper around :class:`torch.nn.LayerNorm` that matches
    the RMSNorm interface so they can be used interchangeably.

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
        bias:    If True, learn an additive bias parameter.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=True)
        if not bias:
            # Zero-out and freeze the bias
            nn.init.zeros_(self.norm.bias)
            self.norm.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over the last dimension."""
        return self.norm(x)

    def extra_repr(self) -> str:
        return f"d_model={self.norm.normalized_shape[0]}"
''')
commit("feat: add LayerNorm wrapper with optional bias and consistent interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — RMSNorm implementation
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/norms.py")
src += '''

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler than LayerNorm — no mean subtraction, only RMS scaling.
    Used in LLaMA, Mistral, and other modern LLMs.

    Formula: x / RMS(x) * weight,  where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the last dimension of ``x``."""
        # Compute RMS along the last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"
'''
write("nanomind/blocks/norms.py", src)
commit("feat: add RMSNorm — simpler LLaMA-style normalization without mean subtraction")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — normalization registry
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/norms.py")
src += '''

# ── Norm registry ─────────────────────────────────────────────────────────────

_NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "layernorm": LayerNorm,
    "rmsnorm":   RMSNorm,
}


def get_norm(name: str, d_model: int, **kwargs) -> nn.Module:
    """
    Instantiate a normalization layer by name.

    Args:
        name:    Norm type — ``"layernorm"`` or ``"rmsnorm"``.
        d_model: Feature dimension.
        **kwargs: Extra arguments forwarded to the norm constructor.

    Returns:
        An :class:`nn.Module` normalization layer.

    Raises:
        ValueError: If the name is not recognised.

    Example::

        norm = get_norm("rmsnorm", d_model=256)
        out  = norm(x)
    """
    key = name.lower().replace("_", "")
    if key not in _NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm '{name}'. Available: {sorted(_NORM_REGISTRY)}"
        )
    return _NORM_REGISTRY[key](d_model, **kwargs)


def list_norms() -> list[str]:
    """Return a sorted list of all registered normalization names."""
    return sorted(_NORM_REGISTRY)
'''
write("nanomind/blocks/norms.py", src)
commit("feat: add get_norm() registry — instantiate LayerNorm or RMSNorm by name")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — FeedForward with GELU
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/feedforward.py", '''\
"""
nanomind/blocks/feedforward.py — Position-wise feed-forward network (FFN).

The FFN is applied identically to each position in the sequence.
It consists of two linear transformations with a non-linearity in between:

    FFN(x) = Linear_2(activation(Linear_1(x)))

The hidden dimension is typically 4x the model dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model:    Input/output dimension.
        d_ff:       Hidden dimension (default: 4 * d_model).
        dropout:    Dropout probability applied after activation.
        activation: Activation function name — ``"gelu"`` or ``"swiglu"``.
        bias:       Whether to use bias in linear layers.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.d_ff       = d_ff or 4 * d_model
        self.activation = activation.lower()
        self.dropout    = dropout

        if self.activation == "swiglu":
            # SwiGLU needs two parallel projections for the gating mechanism
            self.fc1_gate = nn.Linear(d_model, self.d_ff, bias=bias)
            self.fc1_up   = nn.Linear(d_model, self.d_ff, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)

        self.fc2     = nn.Linear(self.d_ff, d_model, bias=bias)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation to each position."""
        if self.activation == "gelu":
            return self._forward_gelu(x)
        if self.activation == "swiglu":
            return self._forward_swiglu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _forward_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Standard two-layer FFN with GELU activation."""
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

    def _forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: gate(x) * sigmoid(gate(x)) * up(x)."""
        raise NotImplementedError  # implemented in next commit

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"activation={self.activation}, dropout={self.dropout}"
        )
''')
commit("feat: add FeedForward class with GELU activation and configurable hidden dim")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — SwiGLU activation
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/feedforward.py")
src = src.replace(
    "        raise NotImplementedError  # implemented in next commit",
    """\
        # SwiGLU: element-wise product of SiLU(gate) and up-projection
        # Reference: Shazeer (2020) — https://arxiv.org/abs/2002.05202
        gate = F.silu(self.fc1_gate(x))   # SiLU = x * sigmoid(x) = Swish
        up   = self.fc1_up(x)
        return self.fc2(self.drop(gate * up))"""
)
write("nanomind/blocks/feedforward.py", src)
commit("feat: implement SwiGLU activation in FeedForward (gate * SiLU(up) projection)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — FFN registry
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/feedforward.py")
src += '''

def get_ffn(d_model: int, activation: str = "gelu", **kwargs) -> FeedForward:
    """
    Convenience factory for :class:`FeedForward`.

    Args:
        d_model:    Model dimension.
        activation: ``"gelu"`` (default) or ``"swiglu"``.
        **kwargs:   Forwarded to :class:`FeedForward` constructor.

    Returns:
        Configured :class:`FeedForward` module.
    """
    supported = ("gelu", "swiglu")
    if activation not in supported:
        raise ValueError(f"Unknown activation '{activation}'. Choose from {supported}")
    return FeedForward(d_model=d_model, activation=activation, **kwargs)
'''
write("nanomind/blocks/feedforward.py", src)
commit("feat: add get_ffn() factory — instantiate FeedForward by activation name")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — TransformerBlock with Pre-LN (skeleton)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/block.py", '''\
"""
nanomind/blocks/block.py — Transformer block (Pre-Norm and Post-Norm variants).

Each block contains:
1. A multi-head causal self-attention sub-layer
2. A position-wise feed-forward sub-layer
Both with residual connections and normalization.

Pre-Norm (used here by default):
    out = x + Attention(Norm(x))
    out = out + FFN(Norm(out))

Post-Norm (original Transformer, Vaswani 2017):
    out = Norm(x + Attention(x))
    out = Norm(out + FFN(out))

Pre-Norm trains more stably at scale without learning rate warmup tricks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.blocks.feedforward import FeedForward
from nanomind.blocks.norms import get_norm


class TransformerBlock(nn.Module):
    """
    A single Pre-Norm transformer block.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        d_ff:       FFN hidden dimension (default: 4 * d_model).
        dropout:    Dropout probability.
        norm_type:  ``"layernorm"`` or ``"rmsnorm"``.
        activation: FFN activation — ``"gelu"`` or ``"swiglu"``.
        norm_placement: ``"pre"`` (default) or ``"post"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        norm_placement: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_placement = norm_placement.lower()
        assert self.norm_placement in ("pre", "post"), (
            "norm_placement must be 'pre' or 'post'"
        )

        self.attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads,
            block_size=block_size, dropout=dropout,
        )
        self.ffn  = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
''')
commit("feat: add TransformerBlock skeleton with attention, FFN, and norm sub-layers")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — implement Pre-LN forward()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src = src.replace(
    "        raise NotImplementedError",
    """\
        \"\"\"
        Apply one transformer block.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV-cache for fast inference.

        Returns:
            Tuple of ``(output, attention_weights)``.
        \"\"\"
        if self.norm_placement == "pre":
            return self._forward_pre_norm(x, kv_cache)
        return self._forward_post_norm(x, kv_cache)

    def _forward_pre_norm(
        self, x: torch.Tensor, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Pre-Norm: Norm -> Sub-layer -> Add residual.\"\"\"
        attn_out, weights = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights

    def _forward_post_norm(
        self, x: torch.Tensor, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Post-Norm: Sub-layer -> Add residual -> Norm.\"\"\"
        attn_out, weights = self.attn(x, kv_cache)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, weights"""
)
write("nanomind/blocks/block.py", src)
commit("feat: implement Pre-LN and Post-LN forward() in TransformerBlock")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — residual scaling (DeepNet-style init)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src = src.replace(
    "        self.norm1 = get_norm(norm_type, d_model)\n"
    "        self.norm2 = get_norm(norm_type, d_model)",
    """\
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        # Residual scaling factor alpha (set to 1.0 by default, < 1 for deep nets)
        self.residual_scale: float = 1.0"""
)
src = src.replace(
    "        x = x + attn_out\n        x = x + self.ffn(self.norm2(x))",
    """\
        x = x + self.residual_scale * attn_out
        x = x + self.residual_scale * self.ffn(self.norm2(x))"""
)
write("nanomind/blocks/block.py", src)
commit("feat: add residual_scale to TransformerBlock for DeepNet-style initialization")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — extra_repr and __repr__
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src += '''
    def extra_repr(self) -> str:
        return (
            f"norm_placement={self.norm_placement}, "
            f"residual_scale={self.residual_scale}"
        )
'''
write("nanomind/blocks/block.py", src)
commit("feat: add extra_repr() to TransformerBlock for descriptive string representation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — BlockConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/config.py", '''\
"""
nanomind/blocks/config.py — Configuration for a single Transformer block.
"""

from dataclasses import dataclass


@dataclass
class BlockConfig:
    """
    Configuration for one :class:`~nanomind.blocks.TransformerBlock`.

    Attributes:
        d_model:        Embedding dimension.
        n_heads:        Number of attention heads.
        block_size:     Maximum sequence length.
        d_ff:           FFN hidden dimension. Defaults to 4 * d_model if None.
        dropout:        Dropout probability.
        norm_type:      ``"layernorm"`` (default) or ``"rmsnorm"``.
        activation:     FFN activation — ``"gelu"`` (default) or ``"swiglu"``.
        norm_placement: ``"pre"`` (Pre-LN, default) or ``"post"`` (Post-LN).
    """
    d_model: int        = 128
    n_heads: int        = 4
    block_size: int     = 128
    d_ff: int | None    = None
    dropout: float      = 0.1
    norm_type: str      = "layernorm"
    activation: str     = "gelu"
    norm_placement: str = "pre"

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.dropout >= 0.0
        assert self.norm_type in ("layernorm", "rmsnorm")
        assert self.activation in ("gelu", "swiglu")
        assert self.norm_placement in ("pre", "post")

    @property
    def effective_d_ff(self) -> int:
        """The FFN hidden dimension (resolves None to 4 * d_model)."""
        return self.d_ff or 4 * self.d_model
''')
commit("feat: add BlockConfig dataclass with validation and effective_d_ff property")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — block_from_config() factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src += '''

def block_from_config(cfg: "BlockConfig") -> TransformerBlock:  # noqa: F821
    """
    Instantiate a :class:`TransformerBlock` from a :class:`BlockConfig`.

    Args:
        cfg: Block configuration dataclass.

    Returns:
        A configured :class:`TransformerBlock`.
    """
    return TransformerBlock(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        block_size=cfg.block_size,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        norm_type=cfg.norm_type,
        activation=cfg.activation,
        norm_placement=cfg.norm_placement,
    )
'''
write("nanomind/blocks/block.py", src)
commit("feat: add block_from_config() factory — build TransformerBlock from BlockConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update blocks __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/blocks/__init__.py", '''\
"""NanoMind transformer blocks sub-package.

Core components:
    - :class:`TransformerBlock`  — Pre/Post-Norm transformer block
    - :class:`FeedForward`       — Position-wise FFN (GELU or SwiGLU)
    - :class:`LayerNorm`         — Standard layer normalization
    - :class:`RMSNorm`           — RMS normalization (LLaMA-style)
    - :class:`BlockConfig`       — Block configuration dataclass
    - :func:`get_norm`           — Norm factory by name
    - :func:`get_ffn`            — FFN factory by activation name
    - :func:`block_from_config`  — Block factory from config
"""

from nanomind.blocks.norms import LayerNorm, RMSNorm, get_norm, list_norms
from nanomind.blocks.feedforward import FeedForward, get_ffn
from nanomind.blocks.block import TransformerBlock, block_from_config
from nanomind.blocks.config import BlockConfig

__all__ = [
    "LayerNorm",
    "RMSNorm",
    "get_norm",
    "list_norms",
    "FeedForward",
    "get_ffn",
    "TransformerBlock",
    "block_from_config",
    "BlockConfig",
]
''')
commit("refactor: export all block components from nanomind/blocks/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: TransformerBlock output shapes
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_blocks.py", '''\
"""
tests/test_blocks.py — Tests for NanoMind transformer blocks.
"""

import pytest
import torch

from nanomind.blocks import (
    TransformerBlock,
    FeedForward,
    LayerNorm,
    RMSNorm,
    BlockConfig,
    get_norm,
    get_ffn,
    block_from_config,
)

B, T, D, H = 2, 16, 64, 4


@pytest.fixture
def block() -> TransformerBlock:
    return TransformerBlock(d_model=D, n_heads=H, block_size=T, dropout=0.0)


# ── TransformerBlock output shape ─────────────────────────────────────────────

class TestTransformerBlockShape:
    def test_output_shape(self, block):
        x = torch.randn(B, T, D)
        out, weights = block(x)
        assert out.shape == (B, T, D)

    def test_weights_shape(self, block):
        x = torch.randn(B, T, D)
        _, weights = block(x)
        assert weights.shape == (B, H, T, T)

    def test_single_token(self, block):
        x = torch.randn(B, 1, D)
        out, _ = block(x)
        assert out.shape == (B, 1, D)

    def test_batch_independence(self, block):
        # Each sample in a batch should be independent
        x = torch.randn(B, T, D)
        out_full, _ = block(x)
        out_single, _ = block(x[:1])
        assert torch.allclose(out_full[:1], out_single, atol=1e-5)
''')
commit("test: add TransformerBlock output shape, weight shape, and batch independence tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: residual connection correctness
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_blocks.py")
src += '''

# ── Residual connections ──────────────────────────────────────────────────────

class TestResidualConnections:
    def test_residual_preserves_scale(self, block):
        # With very small weights (near zero init), output ~ input
        # We check the block doesnt explode or collapse
        x = torch.randn(B, T, D)
        out, _ = block(x)
        # Output should be in a reasonable range
        assert out.abs().max().item() < 1000.0

    def test_residual_scale_default_one(self, block):
        assert block.residual_scale == 1.0

    def test_residual_scale_can_be_changed(self, block):
        block.residual_scale = 0.5
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)
'''
write("tests/test_blocks.py", src)
commit("test: add residual connection scale tests for TransformerBlock")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: Pre-LN vs Post-LN shape equivalence
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_blocks.py")
src += '''

# ── Pre-LN vs Post-LN ────────────────────────────────────────────────────────

class TestNormPlacement:
    def test_pre_norm_output_shape(self):
        block = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                 dropout=0.0, norm_placement="pre")
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)

    def test_post_norm_output_shape(self):
        block = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                 dropout=0.0, norm_placement="post")
        x = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)

    def test_pre_post_outputs_differ(self):
        torch.manual_seed(0)
        pre  = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                dropout=0.0, norm_placement="pre")
        torch.manual_seed(0)
        post = TransformerBlock(d_model=D, n_heads=H, block_size=T,
                                dropout=0.0, norm_placement="post")
        x = torch.randn(B, T, D)
        out_pre,  _ = pre(x)
        out_post, _ = post(x)
        # Different placement -> different outputs
        assert not torch.allclose(out_pre, out_post)
'''
write("tests/test_blocks.py", src)
commit("test: add Pre-LN vs Post-LN output shape and difference tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: SwiGLU shape
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_blocks.py")
src += '''

# ── FeedForward ───────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_gelu_output_shape(self):
        ffn = FeedForward(d_model=D, dropout=0.0, activation="gelu")
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_swiglu_output_shape(self):
        ffn = FeedForward(d_model=D, dropout=0.0, activation="swiglu")
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_custom_d_ff(self):
        ffn = FeedForward(d_model=D, d_ff=D * 8, dropout=0.0)
        x   = torch.randn(B, T, D)
        assert ffn(x).shape == (B, T, D)

    def test_get_ffn_factory(self):
        ffn = get_ffn(D, activation="gelu")
        assert isinstance(ffn, FeedForward)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            get_ffn(D, activation="relu")
'''
write("tests/test_blocks.py", src)
commit("test: add FeedForward GELU, SwiGLU output shape, and factory tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: LayerNorm, RMSNorm, get_norm
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_blocks.py")
src += '''

# ── Norms ─────────────────────────────────────────────────────────────────────

class TestNorms:
    def test_layernorm_output_shape(self):
        norm = LayerNorm(D)
        x    = torch.randn(B, T, D)
        assert norm(x).shape == (B, T, D)

    def test_rmsnorm_output_shape(self):
        norm = RMSNorm(D)
        x    = torch.randn(B, T, D)
        assert norm(x).shape == (B, T, D)

    def test_rmsnorm_unit_rms(self):
        norm = RMSNorm(D)
        x    = torch.randn(1, 1, D)
        out  = norm(x)
        # After RMSNorm (with weight=1), rms(out) ~ 1
        rms  = out.pow(2).mean().sqrt()
        assert abs(rms.item() - 1.0) < 0.1

    def test_get_norm_layernorm(self):
        norm = get_norm("layernorm", D)
        assert isinstance(norm, LayerNorm)

    def test_get_norm_rmsnorm(self):
        norm = get_norm("rmsnorm", D)
        assert isinstance(norm, RMSNorm)

    def test_get_norm_unknown_raises(self):
        with pytest.raises(ValueError):
            get_norm("batchnorm", D)


# ── BlockConfig ───────────────────────────────────────────────────────────────

class TestBlockConfig:
    def test_defaults(self):
        cfg = BlockConfig()
        assert cfg.d_model == 128

    def test_effective_d_ff_default(self):
        cfg = BlockConfig(d_model=64)
        assert cfg.effective_d_ff == 256

    def test_effective_d_ff_custom(self):
        cfg = BlockConfig(d_model=64, d_ff=512)
        assert cfg.effective_d_ff == 512

    def test_invalid_d_model_n_heads(self):
        with pytest.raises(AssertionError):
            BlockConfig(d_model=65, n_heads=4)

    def test_block_from_config(self):
        cfg   = BlockConfig(d_model=D, n_heads=H, block_size=T, dropout=0.0)
        block = block_from_config(cfg)
        x     = torch.randn(B, T, D)
        out, _ = block(x)
        assert out.shape == (B, T, D)
'''
write("tests/test_blocks.py", src)
commit("test: add LayerNorm, RMSNorm, get_norm, and BlockConfig tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 6 | Transformer blocks | 🔜 |",
    "| 6 | Transformer blocks | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Attention: CausalSelfAttention, KVCache, causal mask, Flash Attention dispatch (Day 5)",
    "- Attention: CausalSelfAttention, KVCache, causal mask, Flash Attention dispatch (Day 5)\n- Blocks: TransformerBlock (Pre/Post-LN), FeedForward (GELU/SwiGLU), RMSNorm, LayerNorm (Day 6)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 6 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 6 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 6 COMPLETE ===")
