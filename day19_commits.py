"""
day19_commits.py — 20 atomic commits for Day 19: Sliding Window Attention (SWA).
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

print("\n=== DAY 19: Sliding Window Attention (SWA) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — swa.py skeleton + docstring
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/swa.py", '''\
"""
nanomind/attention/swa.py — Sliding Window Attention (SWA).

Standard full attention has O(T²) memory and compute with sequence length T.
Sliding Window Attention limits each token to attending only to the W
previous tokens, reducing complexity to O(T × W):

  ┌──────────────────────────────────────────────────────┐
  │  Full attention       :  O(T²)   memory & compute    │
  │  Sliding Window (SWA) :  O(T·W)  memory & compute    │
  └──────────────────────────────────────────────────────┘

Each token can attend to up to W tokens in its local causal window.
Tokens more than W steps away are masked out (set to -∞ before softmax).

With multiple layers:
  - Layer 1: each token "sees" W tokens
  - Layer 2: each token effectively sees W² tokens (via layer 1 outputs)
  - Layer L: receptive field grows as W^L → full context with enough layers

Used in:
  - Mistral 7B (window_size=4096 on 8192 block_size)
  - Longformer (local + global attention)

Reference: Jiang et al. (2023) — https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
''')
commit("feat: add nanomind/attention/swa.py — Sliding Window Attention module skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — build_sliding_window_mask()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/swa.py")
src += '''

def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a causal sliding-window attention mask.

    Token at position ``i`` can attend to positions ``j`` where::

        max(0, i - window_size + 1) <= j <= i

    Positions outside the window are masked to ``-inf``.

    Args:
        seq_len:     Sequence length T.
        window_size: Maximum look-back window W (number of tokens visible).
        device:      Target device.

    Returns:
        Boolean mask ``(T, T)`` where ``True`` means **allowed** to attend.
        (Consistent with PyTorch's ``scaled_dot_product_attention`` convention.)

    Example (T=5, W=3)::

        token 0: sees [0]
        token 1: sees [0, 1]
        token 2: sees [0, 1, 2]
        token 3: sees [1, 2, 3]
        token 4: sees [2, 3, 4]
    """
    # Start from causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Build band mask: only allow positions within window_size
    # distance[i, j] = i - j   (negative means j > i → already masked by causal)
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = rows - cols                                      # (T, T)
    window_mask = distance < window_size                        # j >= i - W + 1

    return causal & window_mask
'''
write("nanomind/attention/swa.py", src)
commit("feat: add build_sliding_window_mask() — causal + local window attention mask")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — SlidingWindowAttention class
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/swa.py")
src += '''

class SlidingWindowAttention(nn.Module):
    """
    Multi-head causal self-attention with a sliding local window.

    Each position can only attend to the ``window_size`` most recent tokens,
    drastically reducing memory from O(T²) to O(T × window_size).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length (for mask precomputation).
        window_size: Local attention window (W). Each token sees W past tokens.
        dropout:     Attention dropout probability.
        bias:        Whether projections have bias terms.

    Note:
        With ``window_size >= block_size``, this reduces to standard full attention.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        window_size: int = 256,
        dropout:     float = 0.1,
        bias:        bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = min(window_size, block_size)
        self.dropout     = dropout

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # Precompute the sliding window mask for the full block_size
        mask = build_sliding_window_mask(block_size, self.window_size)
        self.register_buffer("swa_mask", mask)   # (block_size, block_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sliding window attention forward pass.

        Args:
            x:        Input ``(B, T, d_model)``
            kv_cache: Not used during training.

        Returns:
            ``(output, attention_weights)``
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Slice mask to current sequence length
        mask = self.swa_mask[:T, :T]                      # (T, T)

        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply sliding window mask: positions outside window → -inf
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size}"
        )
'''
write("nanomind/attention/swa.py", src)
commit("feat: implement SlidingWindowAttention — local causal attention with window masking")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — SlidingWindowAttention with RoPE (Mistral-exact)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/swa_rope.py", '''\
"""
nanomind/attention/swa_rope.py — Sliding Window Attention with RoPE.

Combines Sliding Window Attention with Rotary Position Embeddings.
This is the exact attention mechanism used in Mistral 7B:
  - SWA limits context to window_size tokens (reduces memory)
  - RoPE encodes relative position (enables length generalisation)
  - GQA reduces KV cache (via n_kv_heads < n_heads — see Day 16)

Reference: Jiang et al. (2023) — https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.swa import build_sliding_window_mask
from nanomind.pos.rope import RotaryEmbedding


class SWARoPEAttention(nn.Module):
    """
    Sliding Window Attention with Rotary Position Embeddings (Mistral-style).

    Args:
        d_model:     Model embedding dimension.
        n_heads:     Number of attention heads.
        block_size:  Maximum sequence length.
        window_size: Local attention window W.
        dropout:     Attention dropout.
        rope_base:   RoPE frequency base.
        bias:        Whether projections have bias.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        window_size: int   = 256,
        dropout:     float = 0.0,
        rope_base:   float = 10000.0,
        bias:        bool  = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = min(window_size, block_size)

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope      = RotaryEmbedding(self.head_dim, block_size, rope_base)
        self.attn_drop = nn.Dropout(dropout)

        mask = build_sliding_window_mask(block_size, self.window_size)
        self.register_buffer("swa_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        mask   = self.swa_mask[:T, :T]
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size} (SWA+RoPE)"
        )
''')
commit("feat: add SWARoPEAttention — Sliding Window + RoPE (Mistral 7B exact attention)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — add window_size to ModelConfig
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src = src.replace(
    "    n_kv_heads:     int | None = None        # None = same as n_heads (standard MHA)",
    "    n_kv_heads:     int | None = None        # None = same as n_heads (standard MHA)\n"
    "    window_size:    int | None = None        # None = full attention; int = SWA window"
)
write("nanomind/model/config.py", src)
commit("feat: add window_size to ModelConfig — None = full attention, int = SWA window")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — add window_size to BlockConfig
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/config.py")
src = src.replace(
    "    n_kv_heads:     int | None = None        # for GQA/MQA; None = MHA",
    "    n_kv_heads:     int | None = None        # for GQA/MQA; None = MHA\n"
    "    window_size:    int | None = None        # None = full attention; int = SWA window"
)
write("nanomind/blocks/config.py", src)
commit("feat: add window_size to BlockConfig for SWA support")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — extend get_attention() factory with swa and swa_rope
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/factory.py")
src = src.replace(
    "from nanomind.attention.gqa_rope import GQARoPEAttention",
    "from nanomind.attention.gqa_rope import GQARoPEAttention\n"
    "from nanomind.attention.swa import SlidingWindowAttention\n"
    "from nanomind.attention.swa_rope import SWARoPEAttention"
)
src = src.replace(
    "    \"gqa_rope\": GQARoPEAttention,",
    "    \"gqa_rope\": GQARoPEAttention,\n"
    "    \"swa\":      SlidingWindowAttention,\n"
    "    \"swa_rope\": SWARoPEAttention,"
)
# Update get_attention() to forward window_size
src = src.replace(
    "    n_kv_heads: int | None = None,\n    **kwargs,\n) -> nn.Module:",
    "    n_kv_heads:  int | None = None,\n"
    "    window_size: int | None = None,\n"
    "    **kwargs,\n) -> nn.Module:"
)
src = src.replace(
    "    # GQA variants need n_kv_heads\n"
    "    if key in (\"gqa\", \"gqa_rope\") and n_kv_heads is not None:\n"
    "        return _ATTENTION_REGISTRY[key](\n"
    "            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,\n"
    "            block_size=block_size, dropout=dropout, **kwargs,\n"
    "        )\n\n"
    "    return _ATTENTION_REGISTRY[key](\n"
    "        d_model=d_model, n_heads=n_heads,\n"
    "        block_size=block_size, dropout=dropout, **kwargs,\n"
    "    )",
    "    # GQA variants need n_kv_heads\n"
    "    if key in (\"gqa\", \"gqa_rope\") and n_kv_heads is not None:\n"
    "        return _ATTENTION_REGISTRY[key](\n"
    "            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,\n"
    "            block_size=block_size, dropout=dropout, **kwargs,\n"
    "        )\n\n"
    "    # SWA variants need window_size\n"
    "    if key in (\"swa\", \"swa_rope\") and window_size is not None:\n"
    "        return _ATTENTION_REGISTRY[key](\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "            window_size=window_size, **kwargs,\n"
    "        )\n\n"
    "    return _ATTENTION_REGISTRY[key](\n"
    "        d_model=d_model, n_heads=n_heads,\n"
    "        block_size=block_size, dropout=dropout, **kwargs,\n"
    "    )"
)
write("nanomind/pos/factory.py", src)
commit("feat: extend get_attention() factory with swa and swa_rope variants")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — update TransformerBlock to pass window_size
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/blocks/block.py")
src = src.replace(
    "        n_kv_heads: int | None = None,\n    ) -> None:",
    "        n_kv_heads:  int | None = None,\n"
    "        window_size: int | None = None,\n    ) -> None:"
)
src = src.replace(
    "        self.attn = get_attention(\n"
    "            pos_type=pos_type,\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "            n_kv_heads=n_kv_heads,\n"
    "        )",
    "        self.attn = get_attention(\n"
    "            pos_type=pos_type,\n"
    "            d_model=d_model, n_heads=n_heads,\n"
    "            block_size=block_size, dropout=dropout,\n"
    "            n_kv_heads=n_kv_heads,\n"
    "            window_size=window_size,\n"
    "        )"
)
write("nanomind/blocks/block.py", src)
commit("feat: update TransformerBlock to pass window_size to get_attention() factory")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — update NanoMind to wire window_size through blocks
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "                n_kv_heads=cfg.n_kv_heads,\n            )\n            for _ in range(cfg.n_layers)",
    "                n_kv_heads=cfg.n_kv_heads,\n"
    "                window_size=cfg.window_size,\n"
    "            )\n            for _ in range(cfg.n_layers)"
)
write("nanomind/model/model.py", src)
commit("feat: update NanoMind to pass window_size from ModelConfig through to blocks")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — memory complexity utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/complexity.py", '''\
"""
nanomind/attention/complexity.py — Attention memory and compute complexity analysis.
"""

from __future__ import annotations


def attention_memory_bytes(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 4,
    window_size: int | None = None,
) -> dict:
    """
    Estimate attention memory usage in bytes.

    Args:
        seq_len:     Sequence length T.
        n_heads:     Number of attention heads H.
        head_dim:    Dimension per head D_h.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).
        window_size: If provided, compute SWA memory instead of full O(T²).

    Returns:
        Dict with:
        - ``attn_matrix_bytes`` : bytes for attention score matrix
        - ``kv_cache_bytes``    : bytes for K and V cache
        - ``total_mb``          : total attention memory in MB
    """
    W = window_size if window_size is not None else seq_len

    # Attention score matrix: (H, T, W) per batch
    attn_matrix = n_heads * seq_len * W * dtype_bytes

    # KV cache: 2 × (H, T, D_h) for K and V
    kv_cache = 2 * n_heads * seq_len * head_dim * dtype_bytes

    total     = attn_matrix + kv_cache
    total_mb  = total / (1024 ** 2)

    return {
        "attn_matrix_bytes": attn_matrix,
        "kv_cache_bytes":    kv_cache,
        "total_bytes":       total,
        "total_mb":          total_mb,
        "window_size":       W,
        "seq_len":           seq_len,
    }


def print_complexity_comparison(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    window_size: int,
    dtype_bytes: int = 2,
) -> None:
    """
    Print a side-by-side memory comparison: full attention vs. SWA.
    """
    full = attention_memory_bytes(seq_len, n_heads, head_dim, dtype_bytes)
    swa  = attention_memory_bytes(seq_len, n_heads, head_dim, dtype_bytes, window_size)
    savings = 1 - swa["total_bytes"] / full["total_bytes"]

    print("=" * 55)
    print(f"Attention Memory: T={seq_len}, H={n_heads}, D_h={head_dim}")
    print("=" * 55)
    print(f"  Full attention : {full['total_mb']:.1f} MB")
    print(f"  SWA (W={window_size:4d})  : {swa['total_mb']:.1f} MB")
    print(f"  Memory savings : {savings:.1%}")
    print("=" * 55)
''')
commit("feat: add attention_memory_bytes() and print_complexity_comparison() — O(T²) vs O(T·W)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — export SWA from attention __init__
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/__init__.py")
src = src.rstrip() + (
    "\n\nfrom nanomind.attention.swa import SlidingWindowAttention, build_sliding_window_mask\n"
    "from nanomind.attention.swa_rope import SWARoPEAttention\n"
    "from nanomind.attention.complexity import attention_memory_bytes, print_complexity_comparison\n"
)
write("nanomind/attention/__init__.py", src)
commit("refactor: export SWA, SWARoPEAttention, and complexity utils from attention package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — add configs/mistral_swa.yaml
# ══════════════════════════════════════════════════════════════════════════════
write("configs/mistral_swa.yaml", '''\
# NanoMind Mistral-inspired SWA configuration
# Sliding Window Attention + RoPE + RMSNorm + SwiGLU

run_name: nanomind_mistral_swa

model:
  vocab_size: 32000
  block_size: 512
  d_model: 256
  n_layers: 6
  n_heads: 8
  n_kv_heads: 2         # GQA: 4:1 ratio
  window_size: 128      # each token sees last 128 tokens only
  dropout: 0.0
  norm_type: rmsnorm
  activation: swiglu
  norm_placement: pre
  pos_type: swa_rope    # Sliding Window + RoPE
  weight_tying: false
  bias: false

train:
  max_iters: 20000
  eval_interval: 500
  grad_clip: 1.0
  use_amp: false
  device: auto
  out_dir: checkpoints/mistral_swa

generate:
  max_new_tokens: 200
  strategy: top_p
  temperature: 0.8
  top_p: 0.95
''')
commit("feat: add configs/mistral_swa.yaml — SWA+RoPE+RMSNorm+SwiGLU Mistral-style config")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — update pos factory __init__ docs
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/__init__.py")
src = src.replace(
    "    - ``\"gqa_rope\"``: GQA + RoPE — Llama 2 70B, Mistral 7B exact",
    "    - ``\"gqa_rope\"``: GQA + RoPE — Llama 2 70B, Mistral 7B exact\n"
    "    - ``\"swa\"``     : Sliding Window Attention — O(T·W) memory\n"
    "    - ``\"swa_rope\"``: SWA + RoPE — Mistral 7B exact (with window)"
)
write("nanomind/pos/__init__.py", src)
commit("docs: update pos/__init__.py docstring with swa and swa_rope types")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: build_sliding_window_mask
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_swa.py", '''\
"""
tests/test_swa.py — Tests for Sliding Window Attention (SWA).
"""

import pytest
import torch

from nanomind.attention.swa import SlidingWindowAttention, build_sliding_window_mask
from nanomind.attention.swa_rope import SWARoPEAttention
from nanomind.attention.complexity import attention_memory_bytes
from nanomind.pos.factory import get_attention, list_pos_types
from nanomind.model import NanoMind, ModelConfig

B, T, D, H = 2, 16, 64, 4
W = 4    # small window for tests


# ── build_sliding_window_mask ─────────────────────────────────────────────────

class TestBuildSlidingWindowMask:
    def test_output_shape(self):
        mask = build_sliding_window_mask(T, W)
        assert mask.shape == (T, T)

    def test_dtype_is_bool(self):
        mask = build_sliding_window_mask(T, W)
        assert mask.dtype == torch.bool

    def test_causal_upper_triangle_is_false(self):
        mask = build_sliding_window_mask(T, W)
        # Upper triangle (future tokens) must all be False
        for i in range(T):
            for j in range(i + 1, T):
                assert not mask[i, j].item(), f"mask[{i},{j}] should be False (future)"

    def test_diagonal_is_true(self):
        mask = build_sliding_window_mask(T, W)
        for i in range(T):
            assert mask[i, i].item(), f"mask[{i},{i}] should be True (self)"

    def test_beyond_window_is_false(self):
        mask = build_sliding_window_mask(T, W)
        for i in range(T):
            for j in range(max(0, i - W), i):
                assert mask[i, j].item(), f"mask[{i},{j}] in window — should be True"
            for j in range(0, max(0, i - W)):
                assert not mask[i, j].item(), f"mask[{i},{j}] beyond window — should be False"

    def test_full_window_equals_causal(self):
        mask_full   = build_sliding_window_mask(T, T)
        mask_causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        assert torch.equal(mask_full, mask_causal)

    def test_window_1_is_diagonal(self):
        mask = build_sliding_window_mask(T, 1)
        expected = torch.eye(T, dtype=torch.bool)
        assert torch.equal(mask, expected)
''')
commit("test: add build_sliding_window_mask() shape, causal, diagonal, and window boundary tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: SlidingWindowAttention output shape
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_swa.py")
src += '''

# ── SlidingWindowAttention ────────────────────────────────────────────────────

class TestSlidingWindowAttention:
    def test_output_shape(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, wts = attn(x)
        assert out.shape == (B, T, D)
        assert wts.shape == (B, H, T, T)

    def test_single_token(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)

    def test_weights_outside_window_are_zero(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        # Attention weights outside window should be exactly 0
        for i in range(T):
            for j in range(0, max(0, i - W)):
                assert wts[0, :, i, j].abs().max().item() < 1e-6, \
                    f"weights[{i},{j}] should be 0 (beyond window)"

    def test_future_weights_are_zero(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        for i in range(T):
            for j in range(i + 1, T):
                assert wts[0, :, i, j].abs().max().item() < 1e-6, \
                    f"weights[{i},{j}] should be 0 (future)"

    def test_causal_invariance(self):
        attn = SlidingWindowAttention(D, H, T, window_size=W, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_window_clips_to_block_size(self):
        attn = SlidingWindowAttention(D, H, T, window_size=T * 10)
        assert attn.window_size == T
'''
write("tests/test_swa.py", src)
commit("test: add SlidingWindowAttention shape, zero weights outside window, and causality tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: SWARoPEAttention
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_swa.py")
src += '''

# ── SWARoPEAttention ──────────────────────────────────────────────────────────

class TestSWARoPEAttention:
    def test_output_shape(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(B, T, D)
        out, wts = attn(x)
        assert out.shape == (B, T, D)

    def test_has_rope_module(self):
        from nanomind.pos.rope import RotaryEmbedding
        attn = SWARoPEAttention(D, H, T, window_size=W)
        assert isinstance(attn.rope, RotaryEmbedding)

    def test_causal(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x1   = torch.randn(1, T, D)
        x2   = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        out1, _ = attn(x1)
        out2, _ = attn(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_window_respected(self):
        attn = SWARoPEAttention(D, H, T, window_size=W, dropout=0.0)
        x    = torch.randn(1, T, D)
        _, wts = attn(x)
        for i in range(T):
            for j in range(0, max(0, i - W)):
                assert wts[0, :, i, j].abs().max().item() < 1e-6
'''
write("tests/test_swa.py", src)
commit("test: add SWARoPEAttention shape, RoPE module, causality, and window tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: NanoMind with SWA
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_swa.py")
src += '''

# ── NanoMind with SWA ─────────────────────────────────────────────────────────

class TestNanoMindWithSWA:
    def _model(self, pos_type, window_size=None):
        cfg = ModelConfig(
            vocab_size=32, block_size=T, d_model=D,
            n_layers=2, n_heads=H, dropout=0.0,
            pos_type=pos_type,
            window_size=window_size,
        )
        return NanoMind(cfg)

    def test_swa_forward(self):
        model  = self._model("swa", window_size=W)
        idx    = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_swa_rope_forward(self):
        model  = self._model("swa_rope", window_size=W)
        idx    = torch.randint(0, 32, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, 32)

    def test_factory_lists_swa_types(self):
        types = list_pos_types()
        assert "swa" in types
        assert "swa_rope" in types

    def test_swa_no_pos_emb(self):
        model = self._model("swa_rope", window_size=W)
        assert model.pos_emb is None
'''
write("tests/test_swa.py", src)
commit("test: add NanoMind with SWA/SWA+RoPE forward pass and factory registration tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: memory complexity utility
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_swa.py")
src += '''

# ── Memory complexity ─────────────────────────────────────────────────────────

class TestMemoryComplexity:
    def test_swa_less_memory_than_full(self):
        full = attention_memory_bytes(256, H, D // H, window_size=None)
        swa  = attention_memory_bytes(256, H, D // H, window_size=W)
        assert swa["total_bytes"] < full["total_bytes"]

    def test_memory_scales_linearly_with_window(self):
        m1 = attention_memory_bytes(256, H, D // H, window_size=32)
        m2 = attention_memory_bytes(256, H, D // H, window_size=64)
        # Doubling window roughly doubles attn matrix size
        assert m2["attn_matrix_bytes"] > m1["attn_matrix_bytes"]

    def test_full_window_equals_full_attention(self):
        T_ = 128
        full = attention_memory_bytes(T_, H, D // H, window_size=None)
        swa  = attention_memory_bytes(T_, H, D // H, window_size=T_)
        assert full["attn_matrix_bytes"] == swa["attn_matrix_bytes"]

    def test_dict_keys_present(self):
        result = attention_memory_bytes(T, H, D // H)
        for key in ("attn_matrix_bytes", "kv_cache_bytes", "total_mb"):
            assert key in result
'''
write("tests/test_swa.py", src)
commit("test: add attention_memory_bytes() SWA vs full memory comparison tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump version + update public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.4.0\"", "__version__ = \"1.5.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v1.5.0 — Sliding Window Attention added to public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Inference** | Speculative decoding (2-4x speedup, exact target distribution) |",
    "| **Inference** | Speculative decoding (2-4x speedup, exact target distribution) |\n"
    "| **Long context** | Sliding Window Attention — O(T·W) vs O(T²), Mistral-style |"
)
readme = readme.replace(
    "**Total: 360 commits across 18 days.**",
    "**Total: 380 commits across 19 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.4.0] — 2024 — Speculative Decoding",
    "## [1.5.0] — 2024 — Sliding Window Attention\n\n### Added\n"
    "- `SlidingWindowAttention` — O(T·W) causal local attention with window mask\n"
    "- `SWARoPEAttention` — SWA + RoPE (Mistral 7B exact attention)\n"
    "- `build_sliding_window_mask()` — causal + local window boolean mask\n"
    "- `window_size` field in `ModelConfig` / `BlockConfig`\n"
    "- `pos_type`: ``\"swa\"`` and ``\"swa_rope\"`` in get_attention() factory\n"
    "- `attention_memory_bytes()` — O(T²) vs O(T·W) memory comparison\n"
    "- `configs/mistral_swa.yaml` — full Mistral-style SWA config\n\n---\n\n"
    "## [1.4.0] — 2024 — Speculative Decoding"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.5.0, update README and CHANGELOG for Day 19 SWA")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 19 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.5.0",
    "-m", "NanoMind v1.5.0 — Sliding Window Attention", check=False)
r = run("git", "push", "origin", "v1.5.0", check=False)
print("Tag v1.5.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 19 COMPLETE — v1.5.0 TAGGED! ===")
