"""
day15_commits.py — 20 atomic commits for Day 15: RoPE & ALiBi Position Embeddings.
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

print("\n=== DAY 15: RoPE & ALiBi Position Embeddings — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — pos package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/__init__.py", '"""NanoMind positional embedding sub-package."""\n')
commit("feat: add nanomind/pos/ package skeleton for positional embedding strategies")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — rotate_half() helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/rope.py", '''\
"""
nanomind/pos/rope.py — Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating query and key vectors in complex space.
Unlike learned or sinusoidal embeddings, RoPE is applied *inside* attention
directly to Q and K — position information is baked into the dot product.

Key properties:
  - Relative position awareness: the dot product <Rq, Rk> depends only on (m-n)
  - Extrapolation: generalises better to longer sequences than learned embeddings
  - No additional parameters
  - Used in: LLaMA, Mistral, PaLM 2, Falcon, GPT-NeoX

Reference: Su et al. (2021) — https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of ``x`` by 90 degrees (half-dimension rotation).

    Splits the last dimension into two halves and returns::

        [-x2, x1]  where x = [x1, x2]

    This implements the complex-number rotation step in RoPE.

    Args:
        x: Input tensor of shape ``(..., d)`` where d is even.

    Returns:
        Rotated tensor of the same shape.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine frequency matrices for RoPE.

    The frequencies follow the formula::

        theta_i = 1 / base^(2i / head_dim),  i in [0, head_dim/2)

    Args:
        head_dim:    Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        base:        RoPE base frequency (default: 10000, as in original paper).
        device:      Target device.

    Returns:
        Tuple of ``(cos, sin)`` tensors of shape ``(max_seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Inverse frequencies: shape (head_dim/2,)
    theta = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # Positions: shape (max_seq_len,)
    t = torch.arange(max_seq_len, device=device).float()

    # Outer product: (max_seq_len, head_dim/2)
    freqs = torch.outer(t, theta)

    # Duplicate to match head_dim: (max_seq_len, head_dim)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs.cos(), freqs.sin()
''')
commit("feat: add rotate_half() and precompute_rope_freqs() — RoPE frequency utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — apply_rotary_emb()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/rope.py")
src += '''

def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Rotates Q and K in-place using precomputed cos/sin frequencies::

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    Args:
        q:   Query tensor ``(B, n_heads, T, head_dim)``
        k:   Key tensor   ``(B, n_heads, T, head_dim)``
        cos: Cosine freq  ``(T, head_dim)``
        sin: Sine freq    ``(T, head_dim)``

    Returns:
        Tuple of rotated ``(q_rot, k_rot)`` with same shapes as inputs.
    """
    # Broadcast cos/sin over batch and head dims
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
'''
write("nanomind/pos/rope.py", src)
commit("feat: add apply_rotary_emb() — rotate Q and K with precomputed RoPE frequencies")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — RotaryEmbedding module
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/rope.py")
src += '''

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.

    Precomputes and caches the cosine and sine frequency tables for the
    maximum sequence length. During forward pass, slices them to the
    current sequence length.

    Args:
        head_dim:    Dimension per attention head.
        max_seq_len: Maximum supported sequence length.
        base:        RoPE frequency base (default: 10000).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.base        = base

        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, base)
        # Register as buffers so they move with .to(device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: ``(B, n_heads, T, head_dim)``
            k: ``(B, n_heads, T, head_dim)``

        Returns:
            Rotated ``(q_rot, k_rot)``
        """
        T = q.shape[2]
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        return apply_rotary_emb(q, k, cos, sin)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"base={self.base}"
        )
'''
write("nanomind/pos/rope.py", src)
commit("feat: add RotaryEmbedding nn.Module with cached cos/sin buffers")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — RoPECausalSelfAttention
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/rope_attention.py", '''\
"""
nanomind/pos/rope_attention.py — Causal self-attention with Rotary Position Embeddings.

Extends the base CausalSelfAttention by replacing the learned positional
embedding with RoPE applied inside the attention head computation.

Key difference from base attention:
  - No positional embedding at the input level
  - RoPE rotates Q and K *after* projection, *before* dot-product
  - Relative position is encoded implicitly in the QK scores
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.pos.rope import RotaryEmbedding


class RoPECausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length (used to build causal mask).
        dropout:     Attention dropout probability.
        rope_base:   RoPE frequency base (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.dropout   = dropout

        # Linear projections (no bias — matches LLaMA style)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE module
        self.rope = RotaryEmbedding(self.head_dim, block_size, rope_base)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with RoPE-rotated Q and K.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV cache (not used in training).

        Returns:
            Tuple of ``(output, attention_weights)``
        """
        B, T, _ = x.shape

        # Project and reshape to heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention with causal mask
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)                                 # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}"
        )
''')
commit("feat: add RoPECausalSelfAttention — causal attention with RoPE Q/K rotation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — ALiBi bias computation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/alibi.py", '''\
"""
nanomind/pos/alibi.py — Attention with Linear Biases (ALiBi).

ALiBi replaces positional embeddings with a simple linear position bias
added to attention scores *before* softmax::

    scores = QK^T / sqrt(d) - m * |i - j|

where ``m`` is a head-specific slope and |i-j| is the distance between positions.

Key properties:
  - No position embedding parameters
  - Excellent length extrapolation (models trained on short sequences
    can generate longer sequences at inference time)
  - Used in: BLOOM, MPT

Reference: Press et al. (2021) — https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute the per-head ALiBi slope values.

    Slopes are the geometric sequence: 2^(-8/n) for n_heads heads.

    Args:
        n_heads: Number of attention heads.

    Returns:
        Slope tensor ``(n_heads,)``
    """

    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_slopes_power_of_2(n_heads))

    # For non-power-of-2 heads, interpolate
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    base_slopes  = _slopes_power_of_2(closest_pow2)
    extra_slopes = _slopes_power_of_2(2 * closest_pow2)[0::2]
    slopes = base_slopes + extra_slopes[: n_heads - closest_pow2]
    return torch.tensor(slopes)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build the ALiBi position bias tensor.

    The bias for position pair (i, j) is ``-slope * |i - j|``.

    Args:
        n_heads: Number of attention heads.
        seq_len: Current sequence length.
        device:  Target device.

    Returns:
        Bias tensor ``(1, n_heads, seq_len, seq_len)``
    """
    slopes = get_alibi_slopes(n_heads).to(device)  # (n_heads,)

    # Build distance matrix: |i - j| for i, j in [0, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)
    distances = (positions - positions.T).abs().float()              # (T, T)

    # Outer product of slopes and distances: (n_heads, T, T)
    bias = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
    return bias.unsqueeze(0)   # (1, n_heads, T, T)
''')
commit("feat: add ALiBi — get_alibi_slopes() and build_alibi_bias() position bias utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — ALiBiCausalSelfAttention
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/alibi_attention.py", '''\
"""
nanomind/pos/alibi_attention.py — Causal self-attention with ALiBi position bias.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.pos.alibi import build_alibi_bias


class ALiBiCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with ALiBi position bias.

    Instead of adding a positional embedding to the input, ALiBi adds
    a fixed linear bias to the attention scores based on token distance.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ALiBi position bias applied to attention scores.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV cache (unused in training).

        Returns:
            Tuple of ``(output, attention_weights)``
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add ALiBi bias (built on the fly for current T)
        alibi  = build_alibi_bias(self.n_heads, T, device=x.device)
        scores = scores + alibi

        # Causal mask
        scores  = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}"
''')
commit("feat: add ALiBiCausalSelfAttention — causal attention with ALiBi position bias")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — get_attention() factory
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/factory.py", '''\
"""
nanomind/pos/factory.py — Attention factory keyed by positional embedding type.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

_ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "learned": CausalSelfAttention,
    "rope":    RoPECausalSelfAttention,
    "alibi":   ALiBiCausalSelfAttention,
}


def get_attention(
    pos_type: str,
    d_model: int,
    n_heads: int,
    block_size: int,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a causal self-attention module by positional embedding type.

    Args:
        pos_type:   ``"learned"`` (default), ``"rope"``, or ``"alibi"``.
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        **kwargs:   Extra arguments forwarded to the attention constructor.

    Returns:
        Configured attention :class:`nn.Module`.

    Raises:
        ValueError: If ``pos_type`` is not recognised.
    """
    key = pos_type.lower()
    if key not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown pos_type '{pos_type}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY)}"
        )
    return _ATTENTION_REGISTRY[key](
        d_model=d_model, n_heads=n_heads,
        block_size=block_size, dropout=dropout,
        **kwargs,
    )


def list_pos_types() -> list[str]:
    """Return a sorted list of all registered positional embedding types."""
    return sorted(_ATTENTION_REGISTRY)
''')
commit("feat: add get_attention() factory — select CausalSelfAttention by pos_type")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — update BlockConfig with pos_type field
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/config.py")
src = src.replace(
    "    norm_placement: str = \"pre\"",
    "    norm_placement: str = \"pre\"\n    pos_type:       str = \"learned\"  # \"learned\", \"rope\", or \"alibi\""
)
src = src.replace(
    "        assert self.norm_placement in (\"pre\", \"post\")",
    "        assert self.norm_placement in (\"pre\", \"post\")\n        assert self.pos_type in (\"learned\", \"rope\", \"alibi\")"
)
write("nanomind/blocks/config.py", src)
commit("feat: add pos_type field to BlockConfig — supports learned, rope, alibi")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update TransformerBlock to use get_attention()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src = src.replace(
    "from nanomind.attention import CausalSelfAttention",
    "from nanomind.pos.factory import get_attention"
)
src = src.replace(
    "        norm_placement: str = \"pre\",\n    ) -> None:",
    "        norm_placement: str = \"pre\",\n        pos_type: str = \"learned\",\n    ) -> None:"
)
src = src.replace(
    "        self.attn = CausalSelfAttention(\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "        )",
    "        self.attn = get_attention(\n"
    "            pos_type=pos_type,\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "        )"
)
write("nanomind/blocks/block.py", src)
commit("feat: update TransformerBlock to use get_attention() factory with pos_type")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — add pos_type to ModelConfig
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src = src.replace(
    "    weight_tying:   bool      = True",
    "    weight_tying:   bool      = True\n    pos_type:       str       = \"learned\"  # \"learned\", \"rope\", \"alibi\""
)
src = src.replace(
    "        assert self.norm_type in (\"layernorm\", \"rmsnorm\")",
    "        assert self.pos_type in (\"learned\", \"rope\", \"alibi\")\n        assert self.norm_type in (\"layernorm\", \"rmsnorm\")"
)
write("nanomind/model/config.py", src)
commit("feat: add pos_type field to ModelConfig — model-level positional embedding config")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — update NanoMind to wire pos_type through blocks
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
# Update block construction to pass pos_type
src = src.replace(
    "                norm_placement=cfg.norm_placement,\n            )\n            for _ in range(cfg.n_layers)",
    "                norm_placement=cfg.norm_placement,\n                pos_type=cfg.pos_type,\n            )\n            for _ in range(cfg.n_layers)"
)
# For rope/alibi, the model-level pos_emb is not needed
src = src.replace(
    "        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.d_model)",
    "        # Learned pos embedding used only when pos_type='learned'\n"
    "        self.pos_emb = (\n"
    "            nn.Embedding(cfg.block_size, cfg.d_model)\n"
    "            if cfg.pos_type == \"learned\"\n"
    "            else None\n"
    "        )"
)
# Update forward to conditionally add pos_emb
src = src.replace(
    "        tok  = self.token_emb(idx)                                    # (B, T, d_model)\n"
    "        pos  = self.pos_emb(torch.arange(T, device=idx.device))      # (T, d_model)\n"
    "        x    = self.emb_drop(tok + pos)",
    "        tok = self.token_emb(idx)   # (B, T, d_model)\n"
    "        if self.pos_emb is not None:\n"
    "            pos = self.pos_emb(torch.arange(T, device=idx.device))\n"
    "            x   = self.emb_drop(tok + pos)\n"
    "        else:\n"
    "            x = self.emb_drop(tok)   # RoPE/ALiBi: no explicit pos emb"
)
write("nanomind/model/model.py", src)
commit("feat: update NanoMind to skip learned pos_emb when using rope or alibi")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — update pos __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/__init__.py", '''\
"""NanoMind positional embedding sub-package.

Positional Embedding Types:
    - ``"learned"`` : Learned absolute positional embeddings (original Transformer / GPT)
    - ``"rope"``    : Rotary Position Embeddings — LLaMA, Mistral, PaLM 2
    - ``"alibi"``   : Attention with Linear Biases — BLOOM, MPT

Primary exports:
    - :func:`get_attention`          — factory: return attention module by pos_type
    - :func:`list_pos_types`         — list all registered positional types
    - :class:`RotaryEmbedding`       — RoPE module
    - :func:`apply_rotary_emb`       — apply RoPE to Q, K tensors
    - :func:`precompute_rope_freqs`  — precompute cos/sin frequency tables
    - :func:`rotate_half`            — 90-degree rotation helper
    - :func:`build_alibi_bias`       — build ALiBi position bias tensor
    - :func:`get_alibi_slopes`       — compute per-head ALiBi slopes
    - :class:`RoPECausalSelfAttention`   — attention with RoPE
    - :class:`ALiBiCausalSelfAttention`  — attention with ALiBi
"""

from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.pos.rope import (
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_rope_freqs,
    rotate_half,
)
from nanomind.pos.alibi import build_alibi_bias, get_alibi_slopes
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

__all__ = [
    "get_attention",
    "list_pos_types",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "precompute_rope_freqs",
    "rotate_half",
    "build_alibi_bias",
    "get_alibi_slopes",
    "RoPECausalSelfAttention",
    "ALiBiCausalSelfAttention",
]
''')
commit("refactor: export all pos components from nanomind/pos/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — add rope/alibi to configs
# ══════════════════════════════════════════════════════════════════════════════
write("configs/rope.yaml", '''\
# NanoMind with RoPE position embeddings (LLaMA-style)
run_name: nanomind_rope

model:
  vocab_size: 256
  block_size: 256
  d_model: 256
  n_layers: 6
  n_heads: 8
  dropout: 0.1
  norm_type: rmsnorm     # RMSNorm used alongside RoPE in LLaMA
  activation: swiglu     # SwiGLU used in LLaMA
  norm_placement: pre
  pos_type: rope          # <-- Rotary Position Embeddings
  weight_tying: false

train:
  max_iters: 10000
  eval_interval: 500
  grad_clip: 1.0
  device: auto
  out_dir: checkpoints/rope
''')
commit("feat: add configs/rope.yaml — LLaMA-style config with RoPE + RMSNorm + SwiGLU")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: rotate_half and apply_rotary_emb
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_pos.py", '''\
"""
tests/test_pos.py — Tests for positional embedding strategies.
"""

import math
import pytest
import torch

from nanomind.pos import (
    rotate_half,
    precompute_rope_freqs,
    apply_rotary_emb,
    RotaryEmbedding,
    build_alibi_bias,
    get_alibi_slopes,
    RoPECausalSelfAttention,
    ALiBiCausalSelfAttention,
    get_attention,
    list_pos_types,
)

B, T, D, H = 2, 16, 64, 4
HEAD_DIM = D // H


# ── rotate_half ───────────────────────────────────────────────────────────────

class TestRotateHalf:
    def test_output_shape(self):
        x = torch.randn(B, H, T, HEAD_DIM)
        assert rotate_half(x).shape == x.shape

    def test_double_rotation_is_neg_identity(self):
        x = torch.randn(4, 8)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)

    def test_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = rotate_half(x)
        # [-x2, x1] = [-3, -4, 1, 2]
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(out, expected)


# ── precompute_rope_freqs ─────────────────────────────────────────────────────

class TestPrecomputeRopeFreqs:
    def test_output_shapes(self):
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        assert cos.shape == (T, HEAD_DIM)
        assert sin.shape == (T, HEAD_DIM)

    def test_first_position_cos_is_one(self):
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        # At position 0, freqs = 0, so cos(0) = 1, sin(0) = 0
        assert torch.allclose(cos[0], torch.ones(HEAD_DIM), atol=1e-5)
        assert torch.allclose(sin[0], torch.zeros(HEAD_DIM), atol=1e-5)


# ── apply_rotary_emb ──────────────────────────────────────────────────────────

class TestApplyRotaryEmb:
    def test_output_shapes(self):
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_magnitude_preserved(self):
        """RoPE is a rotation so it preserves vector magnitude."""
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        cos, sin = precompute_rope_freqs(HEAD_DIM, T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)
''')
commit("test: add rotate_half(), precompute_rope_freqs(), apply_rotary_emb() tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: RotaryEmbedding module
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_pos.py")
src += '''

# ── RotaryEmbedding ───────────────────────────────────────────────────────────

class TestRotaryEmbedding:
    def test_output_shapes(self):
        rope = RotaryEmbedding(HEAD_DIM, max_seq_len=T)
        q = torch.randn(B, H, T, HEAD_DIM)
        k = torch.randn(B, H, T, HEAD_DIM)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_buffers_registered(self):
        rope = RotaryEmbedding(HEAD_DIM, max_seq_len=T)
        assert hasattr(rope, "cos_cached")
        assert hasattr(rope, "sin_cached")

    def test_shorter_seq_works(self):
        rope = RotaryEmbedding(HEAD_DIM, max_seq_len=T)
        q = torch.randn(B, H, T // 2, HEAD_DIM)
        k = torch.randn(B, H, T // 2, HEAD_DIM)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


# ── RoPECausalSelfAttention ───────────────────────────────────────────────────

class TestRoPEAttention:
    def test_output_shape(self):
        attn = RoPECausalSelfAttention(D, H, T, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, w = attn(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, H, T, T)

    def test_single_token(self):
        attn = RoPECausalSelfAttention(D, H, T, dropout=0.0)
        x    = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)

    def test_causal_mask_respected(self):
        """Attention from position i should not see positions > i."""
        attn = RoPECausalSelfAttention(D, H, T, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)   # change only last token
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        # All positions except the last should be identical
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
'''
write("tests/test_pos.py", src)
commit("test: add RotaryEmbedding and RoPECausalSelfAttention shape and causality tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: ALiBi
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_pos.py")
src += '''

# ── ALiBi ─────────────────────────────────────────────────────────────────────

class TestALiBi:
    def test_slopes_length(self):
        slopes = get_alibi_slopes(H)
        assert len(slopes) == H

    def test_slopes_positive(self):
        slopes = get_alibi_slopes(H)
        assert (slopes > 0).all()

    def test_slopes_decreasing(self):
        slopes = get_alibi_slopes(H)
        # Slopes should be monotonically decreasing
        assert (slopes[:-1] >= slopes[1:]).all()

    def test_bias_shape(self):
        bias = build_alibi_bias(H, T)
        assert bias.shape == (1, H, T, T)

    def test_bias_diagonal_is_zero(self):
        bias = build_alibi_bias(H, T)
        # Distance from a position to itself is 0
        for h in range(H):
            diag = bias[0, h].diagonal()
            assert torch.allclose(diag, torch.zeros(T), atol=1e-6)

    def test_bias_non_positive(self):
        # All biases should be <= 0 (penalizing distant positions)
        bias = build_alibi_bias(H, T)
        assert (bias <= 0).all()


class TestALiBiAttention:
    def test_output_shape(self):
        attn = ALiBiCausalSelfAttention(D, H, T, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, w = attn(x)
        assert out.shape == (B, T, D)

    def test_single_token(self):
        attn = ALiBiCausalSelfAttention(D, H, T, dropout=0.0)
        x    = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)
'''
write("tests/test_pos.py", src)
commit("test: add ALiBi slopes, bias shape, sign, and ALiBiAttention output tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: get_attention factory + list_pos_types
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_pos.py")
src += '''

# ── get_attention factory ─────────────────────────────────────────────────────

class TestGetAttentionFactory:
    def test_learned_returns_causal_attn(self):
        from nanomind.attention import CausalSelfAttention
        attn = get_attention("learned", D, H, T, dropout=0.0)
        assert isinstance(attn, CausalSelfAttention)

    def test_rope_returns_rope_attn(self):
        attn = get_attention("rope", D, H, T, dropout=0.0)
        assert isinstance(attn, RoPECausalSelfAttention)

    def test_alibi_returns_alibi_attn(self):
        attn = get_attention("alibi", D, H, T, dropout=0.0)
        assert isinstance(attn, ALiBiCausalSelfAttention)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_attention("sinusoidal", D, H, T)

    def test_list_pos_types(self):
        types = list_pos_types()
        assert "learned" in types
        assert "rope" in types
        assert "alibi" in types


# ── NanoMind with RoPE ────────────────────────────────────────────────────────

class TestNanoMindWithRoPE:
    def test_rope_model_forward(self):
        from nanomind import NanoMind, ModelConfig
        cfg   = ModelConfig(vocab_size=32, block_size=T, d_model=D,
                            n_layers=2, n_heads=H, dropout=0.0, pos_type="rope")
        model = NanoMind(cfg)
        idx   = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_alibi_model_forward(self):
        from nanomind import NanoMind, ModelConfig
        cfg   = ModelConfig(vocab_size=32, block_size=T, d_model=D,
                            n_layers=2, n_heads=H, dropout=0.0, pos_type="alibi")
        model = NanoMind(cfg)
        idx   = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_rope_has_no_pos_emb(self):
        from nanomind import NanoMind, ModelConfig
        cfg   = ModelConfig(vocab_size=32, block_size=T, d_model=D,
                            n_layers=1, n_heads=H, dropout=0.0, pos_type="rope")
        model = NanoMind(cfg)
        assert model.pos_emb is None
'''
write("tests/test_pos.py", src)
commit("test: add get_attention factory, list_pos_types, and NanoMind with RoPE/ALiBi tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — update nanomind/__init__.py to expose pos package
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace(
    "__version__ = \"1.0.0\"",
    "__version__ = \"1.1.0\""
)
src = src.replace(
    "from nanomind.model import NanoMind, ModelConfig\nfrom nanomind.config import NanoMindConfig",
    "from nanomind.model import NanoMind, ModelConfig\nfrom nanomind.config import NanoMindConfig\nfrom nanomind.pos import get_attention, list_pos_types"
)
src = src.replace(
    "__all__ = [\n    \"NanoMind\",\n    \"ModelConfig\",\n    \"NanoMindConfig\",\n    \"__version__\",\n]",
    "__all__ = [\n    \"NanoMind\",\n    \"ModelConfig\",\n    \"NanoMindConfig\",\n    \"get_attention\",\n    \"list_pos_types\",\n    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump version to 1.1.0 and expose pos package in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Attention** | SDPA, CausalSelfAttention, KV-Cache, Flash Attention |",
    "| **Attention** | SDPA, CausalSelfAttention, KV-Cache, Flash Attention |"
    "\n| **Position** | Learned, RoPE (LLaMA-style), ALiBi (BLOOM-style) |"
)
readme = readme.replace(
    "**Total: 280 commits across 14 days.**",
    "**Total: 300 commits across 15 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.0.0] — 2024 — Initial Release 🎉",
    "## [1.1.0] — 2024 — RoPE & ALiBi\n\n### Added\n- RoPE (Rotary Position Embeddings) — `pos_type=\"rope\"` in ModelConfig\n- ALiBi (Attention with Linear Biases) — `pos_type=\"alibi\"` in ModelConfig\n- `get_attention()` factory to swap positional embedding type via config\n- `configs/rope.yaml` — LLaMA-style config with RoPE + RMSNorm + SwiGLU\n- NanoMind now skips learned pos_emb when pos_type is rope or alibi\n\n---\n\n## [1.0.0] — 2024 — Initial Release 🎉"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 15 complete — RoPE & ALiBi, version bumped to 1.1.0")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 15 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

# Tag v1.1.0
run("git", "tag", "-a", "v1.1.0", "-m", "NanoMind v1.1.0 — RoPE & ALiBi", check=False)
r = run("git", "push", "origin", "v1.1.0", check=False)
print("Tag v1.1.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 15 COMPLETE — v1.1.0 TAGGED! ===")
