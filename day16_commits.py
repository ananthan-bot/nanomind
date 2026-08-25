"""
day16_commits.py — 20 atomic commits for Day 16: Grouped-Query Attention (GQA) & MQA.
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

print("\n=== DAY 16: Grouped-Query Attention (GQA) & MQA — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — repeat_kv() utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/gqa.py", '''\
"""
nanomind/attention/gqa.py — Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

Standard multi-head attention (MHA) uses n_heads query, key, and value heads.
GQA and MQA reduce memory pressure by sharing fewer KV heads across query heads:

  ┌─────────────────────────────────────────────────────┐
  │  MHA  (Multi-Head Attention)  → n_kv = n_heads      │
  │  GQA  (Grouped-Query)        → 1 < n_kv < n_heads   │
  │  MQA  (Multi-Query)          → n_kv = 1             │
  └─────────────────────────────────────────────────────┘

Benefits:
  - GQA/MQA dramatically reduce KV-cache size at inference time
  - KV-cache grows as: n_kv_heads × head_dim × seq_len × 2 (K+V)
  - Mistral 7B uses 8 KV heads for 32 query heads (4× cache reduction)
  - Llama 2 70B uses 8 KV heads for 64 query heads (8× cache reduction)

References:
  - GQA: Ainslie et al. (2023) — https://arxiv.org/abs/2305.13245
  - MQA: Shazeer (2019) — https://arxiv.org/abs/1911.02150
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of query heads for GQA.

    Each KV head is repeated ``n_rep`` times along the head dimension.
    This is equivalent to broadcasting each KV group across its query heads.

    Args:
        x:     KV tensor ``(B, n_kv_heads, T, head_dim)``
        n_rep: Number of times to repeat each KV head (= n_heads // n_kv_heads).

    Returns:
        Expanded tensor ``(B, n_heads, T, head_dim)``
    """
    if n_rep == 1:
        return x
    B, n_kv, T, head_dim = x.shape
    return (
        x.unsqueeze(2)                           # (B, n_kv, 1, T, head_dim)
         .expand(B, n_kv, n_rep, T, head_dim)   # (B, n_kv, n_rep, T, head_dim)
         .reshape(B, n_kv * n_rep, T, head_dim) # (B, n_heads, T, head_dim)
    )
''')
commit("feat: add repeat_kv() — expand KV heads to match query heads for GQA computation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — GroupedQueryAttention skeleton
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/gqa.py")
src += '''

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Uses ``n_heads`` query heads but only ``n_kv_heads`` key/value heads.
    Each group of ``n_heads // n_kv_heads`` query heads shares one KV head.

    Special cases:
      - ``n_kv_heads == n_heads``  → standard Multi-Head Attention (MHA)
      - ``n_kv_heads == 1``        → Multi-Query Attention (MQA)

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        n_kv_heads: Number of key/value heads (must divide n_heads evenly).
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        bias:       Whether to add bias to projection layers.

    Raises:
        AssertionError: If ``n_heads % n_kv_heads != 0``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repetitions per KV head
        self.head_dim   = d_model // n_heads
        self.dropout    = dropout

        # Query projection: full n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        # Key / Value projections: only n_kv_heads
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
'''
write("nanomind/attention/gqa.py", src)
commit("feat: add GroupedQueryAttention class skeleton with n_kv_heads projections")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — GQA forward pass
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/gqa.py")
src += '''
    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GQA forward pass.

        Steps:
        1. Project input to Q (n_heads), K (n_kv_heads), V (n_kv_heads)
        2. Expand K and V by repeating each n_rep times → (n_heads, T, head_dim)
        3. Compute scaled dot-product attention with causal mask
        4. Project output

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Optional KV cache (unused during training).

        Returns:
            Tuple of ``(output, attention_weights)``.
        """
        B, T, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match query heads
        k = repeat_kv(k, self.n_rep)   # (B, n_heads, T, head_dim)
        v = repeat_kv(v, self.n_rep)   # (B, n_heads, T, head_dim)

        # Scaled dot-product attention
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"n_rep={self.n_rep}, "
            f"head_dim={self.head_dim}"
        )
'''
write("nanomind/attention/gqa.py", src)
commit("feat: implement GroupedQueryAttention.forward() with repeat_kv expansion")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — MultiQueryAttention convenience class
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/gqa.py")
src += '''

class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-Query Attention (MQA) — special case of GQA where n_kv_heads = 1.

    All query heads share a single key and value head.
    This provides the maximum KV-cache memory reduction.

    Used in: Falcon 7B, early PaLM models.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        bias:       Whether to add bias to projection layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,              # single shared KV head
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads=1 (MQA)"
        )
'''
write("nanomind/attention/gqa.py", src)
commit("feat: add MultiQueryAttention (MQA) — n_kv_heads=1 special case of GQA")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — GQA with RoPE
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/gqa_rope.py", '''\
"""
nanomind/attention/gqa_rope.py — Grouped-Query Attention with RoPE.

Combines GQA\'s memory-efficient KV sharing with RoPE\'s relative position
encoding. This is the exact attention mechanism used in Llama 2 and Mistral.

Architecture:
  - Q: n_heads projections with RoPE applied
  - K: n_kv_heads projections with RoPE applied
  - V: n_kv_heads projections (no positional encoding needed)
  - K, V expanded via repeat_kv before attention
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.gqa import repeat_kv
from nanomind.pos.rope import RotaryEmbedding


class GQARoPEAttention(nn.Module):
    """
    Grouped-Query Attention with Rotary Position Embeddings.

    This is the standard attention in Llama 2 / Mistral:
      - ``n_kv_heads`` KV heads (far fewer than query heads)
      - RoPE applied to both Q and K *after* projection

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        n_kv_heads: Number of KV heads (must divide n_heads).
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        rope_base:  RoPE frequency base (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_size: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert d_model % n_heads == 0

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj   = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope      = RotaryEmbedding(self.head_dim, block_size, rope_base)
        self.attn_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (must be done before repeat_kv)
        q, k = self.rope(q, k)

        # Expand K and V to match query head count
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scale   = 1.0 / math.sqrt(self.head_dim)
        scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores  = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads} (GQA+RoPE)"
        )
''')
commit("feat: add GQARoPEAttention — Llama 2 / Mistral style GQA with RoPE")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — add n_kv_heads to ModelConfig
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src = src.replace(
    "    pos_type:       str       = \"learned\"  # \"learned\", \"rope\", \"alibi\"",
    "    pos_type:       str       = \"learned\"  # \"learned\", \"rope\", \"alibi\"\n"
    "    n_kv_heads:     int | None = None        # None = same as n_heads (standard MHA)"
)
src = src.replace(
    "        assert self.pos_type in (\"learned\", \"rope\", \"alibi\")",
    "        assert self.pos_type in (\"learned\", \"rope\", \"alibi\")\n"
    "        if self.n_kv_heads is not None:\n"
    "            assert self.n_heads % self.n_kv_heads == 0, (\n"
    "                f\"n_heads ({self.n_heads}) must be divisible by \"\n"
    "                f\"n_kv_heads ({self.n_kv_heads})\"\n"
    "            )"
)
write("nanomind/model/config.py", src)
commit("feat: add n_kv_heads to ModelConfig — None defaults to MHA (n_kv_heads == n_heads)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — update get_attention() factory with gqa / mqa / gqa_rope
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/pos/factory.py", '''\
"""
nanomind/pos/factory.py — Attention factory keyed by positional embedding type.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.attention import CausalSelfAttention
from nanomind.attention.gqa import GroupedQueryAttention, MultiQueryAttention
from nanomind.attention.gqa_rope import GQARoPEAttention
from nanomind.pos.rope_attention import RoPECausalSelfAttention
from nanomind.pos.alibi_attention import ALiBiCausalSelfAttention

_ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "learned":  CausalSelfAttention,
    "rope":     RoPECausalSelfAttention,
    "alibi":    ALiBiCausalSelfAttention,
    "gqa":      GroupedQueryAttention,
    "mqa":      MultiQueryAttention,
    "gqa_rope": GQARoPEAttention,
}


def get_attention(
    pos_type: str,
    d_model: int,
    n_heads: int,
    block_size: int,
    dropout: float = 0.1,
    n_kv_heads: int | None = None,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a causal self-attention module by type.

    Args:
        pos_type:   ``"learned"``, ``"rope"``, ``"alibi"``,
                    ``"gqa"``, ``"mqa"``, or ``"gqa_rope"``.
        d_model:    Model embedding dimension.
        n_heads:    Number of query heads.
        block_size: Maximum sequence length.
        dropout:    Attention dropout probability.
        n_kv_heads: KV head count for GQA/MQA (ignored for MHA/MQA).
        **kwargs:   Extra arguments forwarded to the attention constructor.

    Returns:
        Configured attention :class:`nn.Module`.
    """
    key = pos_type.lower()
    if key not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown pos_type '{pos_type}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY)}"
        )

    # GQA variants need n_kv_heads
    if key in ("gqa", "gqa_rope") and n_kv_heads is not None:
        return _ATTENTION_REGISTRY[key](
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            block_size=block_size, dropout=dropout, **kwargs,
        )

    return _ATTENTION_REGISTRY[key](
        d_model=d_model, n_heads=n_heads,
        block_size=block_size, dropout=dropout, **kwargs,
    )


def list_pos_types() -> list[str]:
    """Return sorted list of all registered attention/positional types."""
    return sorted(_ATTENTION_REGISTRY)
''')
commit("feat: extend get_attention() factory with gqa, mqa, gqa_rope variants")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — update BlockConfig with n_kv_heads
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/config.py")
src = src.replace(
    "    pos_type:       str = \"learned\"  # \"learned\", \"rope\", or \"alibi\"",
    "    pos_type:       str       = \"learned\"  # \"learned\", \"rope\", \"alibi\", \"gqa\", \"mqa\", \"gqa_rope\"\n"
    "    n_kv_heads:     int | None = None        # for GQA/MQA; None = MHA"
)
write("nanomind/blocks/config.py", src)
commit("feat: add n_kv_heads to BlockConfig for GQA/MQA support")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — update TransformerBlock to pass n_kv_heads
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src = src.replace(
    "        norm_placement: str = \"pre\",\n        pos_type: str = \"learned\",\n    ) -> None:",
    "        norm_placement: str = \"pre\",\n        pos_type: str = \"learned\",\n"
    "        n_kv_heads: int | None = None,\n    ) -> None:"
)
src = src.replace(
    "        self.attn = get_attention(\n"
    "            pos_type=pos_type,\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "        )",
    "        self.attn = get_attention(\n"
    "            pos_type=pos_type,\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "            n_kv_heads=n_kv_heads,\n"
    "        )"
)
write("nanomind/blocks/block.py", src)
commit("feat: update TransformerBlock to pass n_kv_heads to get_attention() factory")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update NanoMind to wire n_kv_heads through blocks
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "                norm_placement=cfg.norm_placement,\n                pos_type=cfg.pos_type,\n            )\n            for _ in range(cfg.n_layers)",
    "                norm_placement=cfg.norm_placement,\n"
    "                pos_type=cfg.pos_type,\n"
    "                n_kv_heads=cfg.n_kv_heads,\n"
    "            )\n            for _ in range(cfg.n_layers)"
)
write("nanomind/model/model.py", src)
commit("feat: update NanoMind to pass n_kv_heads from ModelConfig through to blocks")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — update attention __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/__init__.py")
# Add GQA exports
new_imports = (
    "from nanomind.attention.gqa import (\n"
    "    GroupedQueryAttention,\n"
    "    MultiQueryAttention,\n"
    "    repeat_kv,\n"
    ")\n"
    "from nanomind.attention.gqa_rope import GQARoPEAttention\n"
)
src = src.rstrip() + "\n\n" + new_imports
write("nanomind/attention/__init__.py", src)
commit("refactor: export GQA, MQA, GQARoPEAttention, and repeat_kv from attention package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — update pos __init__ exports with new factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/__init__.py")
src = src.replace(
    "    - ``\"alibi\"``   : Attention with Linear Biases — BLOOM, MPT",
    "    - ``\"alibi\"``   : Attention with Linear Biases — BLOOM, MPT\n"
    "    - ``\"gqa\"``     : Grouped-Query Attention — Llama 2, Mistral\n"
    "    - ``\"mqa\"``     : Multi-Query Attention — Falcon 7B\n"
    "    - ``\"gqa_rope\"``: GQA + RoPE — Llama 2 70B, Mistral 7B exact"
)
write("nanomind/pos/__init__.py", src)
commit("docs: update pos/__init__.py docstring with gqa, mqa, gqa_rope types")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — add configs/mistral_style.yaml
# ══════════════════════════════════════════════════════════════════════════════
write("configs/mistral_style.yaml", '''\
# NanoMind Mistral-style configuration
# GQA + RoPE + SwiGLU + RMSNorm — mirrors Mistral 7B architecture

run_name: nanomind_mistral_style

model:
  vocab_size: 32000
  block_size: 1024
  d_model: 512
  n_layers: 8
  n_heads: 16         # 16 query heads
  n_kv_heads: 4       # 4 KV heads (4:1 GQA ratio — 4x KV cache reduction)
  d_ff: 2048
  dropout: 0.0
  norm_type: rmsnorm
  activation: swiglu
  norm_placement: pre
  pos_type: gqa_rope   # GQA with Rotary Position Embeddings
  weight_tying: false
  bias: false

train:
  max_iters: 50000
  eval_interval: 1000
  grad_clip: 1.0
  use_amp: true
  grad_accum_steps: 8
  device: auto
  out_dir: checkpoints/mistral_style

checkpoint:
  save_interval: 2000
  keep_last_n: 3
  save_best: true

generate:
  max_new_tokens: 256
  strategy: top_p
  temperature: 0.7
  top_p: 0.95
''')
commit("feat: add configs/mistral_style.yaml — GQA+RoPE+SwiGLU+RMSNorm configuration")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — add configs/llama2_style.yaml
# ══════════════════════════════════════════════════════════════════════════════
write("configs/llama2_style.yaml", '''\
# NanoMind Llama 2 style configuration
# GQA + RoPE + SwiGLU + RMSNorm (small-scale reproduction)

run_name: nanomind_llama2_style

model:
  vocab_size: 32000
  block_size: 512
  d_model: 256
  n_layers: 6
  n_heads: 8           # 8 query heads
  n_kv_heads: 2        # 2 KV heads (4:1 ratio like Llama 2 70B at scale)
  dropout: 0.0
  norm_type: rmsnorm
  activation: swiglu
  norm_placement: pre
  pos_type: gqa_rope
  weight_tying: false
  bias: false

train:
  max_iters: 20000
  eval_interval: 500
  grad_clip: 1.0
  use_amp: false
  device: auto
  out_dir: checkpoints/llama2_style

generate:
  max_new_tokens: 200
  strategy: top_k
  temperature: 0.8
  top_k: 40
''')
commit("feat: add configs/llama2_style.yaml — small-scale Llama 2 style config with GQA+RoPE")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: repeat_kv
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_gqa.py", '''\
"""
tests/test_gqa.py — Tests for Grouped-Query Attention (GQA) and MQA.
"""

import pytest
import torch

from nanomind.attention.gqa import GroupedQueryAttention, MultiQueryAttention, repeat_kv
from nanomind.attention.gqa_rope import GQARoPEAttention
from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.model import NanoMind, ModelConfig

B, T, D = 2, 16, 64
N_HEADS = 8
N_KV    = 2
HEAD_DIM = D // N_HEADS


# ── repeat_kv ─────────────────────────────────────────────────────────────────

class TestRepeatKV:
    def test_n_rep_1_is_identity(self):
        x = torch.randn(B, N_KV, T, HEAD_DIM)
        assert torch.equal(repeat_kv(x, 1), x)

    def test_output_shape(self):
        x = torch.randn(B, N_KV, T, HEAD_DIM)
        out = repeat_kv(x, N_HEADS // N_KV)
        assert out.shape == (B, N_HEADS, T, HEAD_DIM)

    def test_repeated_values_equal(self):
        x   = torch.randn(B, N_KV, T, HEAD_DIM)
        n_rep = N_HEADS // N_KV
        out = repeat_kv(x, n_rep)
        for i in range(N_KV):
            for r in range(n_rep):
                assert torch.equal(out[:, i * n_rep + r], x[:, i])

    def test_single_kv_head(self):
        x   = torch.randn(B, 1, T, HEAD_DIM)
        out = repeat_kv(x, N_HEADS)
        assert out.shape == (B, N_HEADS, T, HEAD_DIM)
        # All heads should be identical
        for h in range(N_HEADS):
            assert torch.equal(out[:, h], x[:, 0])
''')
commit("test: add repeat_kv() shape, identity, and value equality tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: GroupedQueryAttention shapes
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_gqa.py")
src += '''

# ── GroupedQueryAttention ─────────────────────────────────────────────────────

class TestGroupedQueryAttention:
    def test_output_shape(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x   = torch.randn(B, T, D)
        out, w = gqa(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, N_HEADS, T, T)

    def test_single_token(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x   = torch.randn(B, 1, D)
        out, _ = gqa(x)
        assert out.shape == (B, 1, D)

    def test_causal(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x1  = torch.randn(1, T, D)
        x2  = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = gqa(x1)
        out2, _ = gqa(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(D, N_HEADS, 3, T)   # 8 % 3 != 0

    def test_fewer_params_than_mha(self):
        gqa = GroupedQueryAttention(D, N_HEADS, N_KV, T)
        from nanomind.attention import CausalSelfAttention
        mha = CausalSelfAttention(D, N_HEADS, T)
        gqa_params = sum(p.numel() for p in gqa.parameters())
        mha_params = sum(p.numel() for p in mha.parameters())
        assert gqa_params < mha_params
'''
write("tests/test_gqa.py", src)
commit("test: add GroupedQueryAttention output shape, causality, and param count tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: MultiQueryAttention
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_gqa.py")
src += '''

# ── MultiQueryAttention ───────────────────────────────────────────────────────

class TestMultiQueryAttention:
    def test_output_shape(self):
        mqa = MultiQueryAttention(D, N_HEADS, T, dropout=0.0)
        x   = torch.randn(B, T, D)
        out, w = mqa(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, N_HEADS, T, T)

    def test_n_kv_heads_is_one(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert mqa.n_kv_heads == 1

    def test_kv_proj_size(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert mqa.k_proj.out_features == HEAD_DIM   # single KV head
        assert mqa.v_proj.out_features == HEAD_DIM

    def test_mqa_is_gqa_subclass(self):
        mqa = MultiQueryAttention(D, N_HEADS, T)
        assert isinstance(mqa, GroupedQueryAttention)
'''
write("tests/test_gqa.py", src)
commit("test: add MultiQueryAttention n_kv_heads, proj size, and subclass tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: GQARoPEAttention
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_gqa.py")
src += '''

# ── GQARoPEAttention ──────────────────────────────────────────────────────────

class TestGQARoPEAttention:
    def test_output_shape(self):
        attn = GQARoPEAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, w = attn(x)
        assert out.shape == (B, T, D)
        assert w.shape   == (B, N_HEADS, T, T)

    def test_has_rope_module(self):
        attn = GQARoPEAttention(D, N_HEADS, N_KV, T)
        from nanomind.pos.rope import RotaryEmbedding
        assert isinstance(attn.rope, RotaryEmbedding)

    def test_causal(self):
        attn = GQARoPEAttention(D, N_HEADS, N_KV, T, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
'''
write("tests/test_gqa.py", src)
commit("test: add GQARoPEAttention output shape, RoPE module, and causality tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: NanoMind with GQA and MQA
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_gqa.py")
src += '''

# ── NanoMind integration ──────────────────────────────────────────────────────

class TestNanoMindWithGQA:
    def _make(self, pos_type, n_kv_heads=None):
        cfg = ModelConfig(
            vocab_size=32, block_size=T, d_model=D,
            n_layers=2, n_heads=N_HEADS, dropout=0.0,
            pos_type=pos_type,
            n_kv_heads=n_kv_heads,
        )
        return NanoMind(cfg)

    def test_gqa_forward(self):
        model = self._make("gqa", n_kv_heads=N_KV)
        idx   = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_mqa_forward(self):
        model = self._make("mqa")
        idx   = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_gqa_rope_forward(self):
        model = self._make("gqa_rope", n_kv_heads=N_KV)
        idx   = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_gqa_fewer_params_than_mha(self):
        gqa_model = self._make("gqa", n_kv_heads=N_KV)
        mha_model = self._make("learned")
        assert gqa_model.num_parameters() < mha_model.num_parameters()

    def test_factory_lists_gqa_types(self):
        types = list_pos_types()
        assert "gqa" in types
        assert "mqa" in types
        assert "gqa_rope" in types
'''
write("tests/test_gqa.py", src)
commit("test: add NanoMind GQA/MQA/GQA+RoPE forward pass and param count tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG, bump to v1.2.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.1.0\"", "__version__ = \"1.2.0\"")
write("nanomind/__init__.py", src)

readme = read("README.md")
readme = readme.replace(
    "| **Position** | Learned, RoPE (LLaMA-style), ALiBi (BLOOM-style) |",
    "| **Position** | Learned, RoPE (LLaMA-style), ALiBi (BLOOM-style) |\n"
    "| **Attention** | MHA, GQA (Llama 2/Mistral), MQA (Falcon), GQA+RoPE |"
)
readme = readme.replace(
    "**Total: 300 commits across 15 days.**",
    "**Total: 320 commits across 16 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.1.0] — 2024 — RoPE & ALiBi",
    "## [1.2.0] — 2024 — GQA & MQA\n\n### Added\n"
    "- `GroupedQueryAttention` — GQA with configurable n_kv_heads\n"
    "- `MultiQueryAttention` — MQA (n_kv_heads=1)\n"
    "- `GQARoPEAttention` — Llama 2 / Mistral exact attention\n"
    "- `repeat_kv()` — expand KV heads to match query heads\n"
    "- `n_kv_heads` field in `ModelConfig` (None = standard MHA)\n"
    "- `pos_type`: ``\"gqa\"``, ``\"mqa\"``, ``\"gqa_rope\"`` in factory\n"
    "- `configs/mistral_style.yaml` and `configs/llama2_style.yaml`\n\n---\n\n"
    "## [1.1.0] — 2024 — RoPE & ALiBi"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.2.0, update README and CHANGELOG for Day 16 GQA/MQA")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 16 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.2.0", "-m", "NanoMind v1.2.0 — GQA, MQA, GQA+RoPE", check=False)
r = run("git", "push", "origin", "v1.2.0", check=False)
print("Tag v1.2.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 16 COMPLETE — v1.2.0 TAGGED! ===")
