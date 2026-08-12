"""
day5_commits.py — 20 atomic commits for Day 5: Attention Mechanism.
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

print("\n=== DAY 5: Attention Mechanism — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — attention package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/__init__.py", '"""NanoMind attention sub-package."""\n')
commit("feat: add nanomind/attention/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — scaled_dot_product_attention() standalone function
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/functional.py", '''\
"""
nanomind/attention/functional.py — Pure-function attention operations.

Implements the core mathematical operations of attention independently
of any nn.Module, so they can be tested in isolation.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Implements: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        q:         Query tensor  ``(B, n_heads, T, head_dim)``
        k:         Key tensor    ``(B, n_heads, T, head_dim)``
        v:         Value tensor  ``(B, n_heads, T, head_dim)``
        mask:      Boolean mask ``(1, 1, T, T)`` — True positions are MASKED OUT
        dropout_p: Attention dropout probability
        training:  Whether the model is in training mode

    Returns:
        Tuple of:
        - ``out``    : attended output ``(B, n_heads, T, head_dim)``
        - ``weights``: attention weights ``(B, n_heads, T, T)``
    """
    d_k = q.size(-1)
    scale = math.sqrt(d_k)

    # (B, n_heads, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    out = torch.matmul(weights, v)
    return out, weights
''')
commit("feat: add scaled_dot_product_attention() — core attention math as pure function")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — causal mask generation utility
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/functional.py")
src += '''

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention mask.

    A True value at position (i, j) means position i is NOT allowed
    to attend to position j (i.e., j > i is masked out).

    Args:
        seq_len: Sequence length T.
        device:  Target device for the mask tensor.

    Returns:
        Boolean tensor of shape ``(1, 1, T, T)``.
        Upper triangle (excluding diagonal) is True (masked).

    Example::

        mask = make_causal_mask(4, device)
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
    """
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(ones, diagonal=1)          # Upper triangle = True (masked)
    return mask.unsqueeze(0).unsqueeze(0)        # (1, 1, T, T)
'''
write("nanomind/attention/functional.py", src)
commit("feat: add make_causal_mask() — upper-triangular boolean mask for causal attention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — CausalSelfAttention class skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/attention.py", '''\
"""
nanomind/attention/attention.py — Causal multi-head self-attention module.

Implements the decoder-style attention block used in GPT:
- All positions attend only to current and past positions (causal mask)
- Fused QKV projection for efficiency
- Optional KV-cache for fast autoregressive inference
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.attention.functional import make_causal_mask, scaled_dot_product_attention


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Args:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length (used to pre-allocate the mask).
        dropout:    Attention and residual dropout probability.
        bias:       Whether to use bias in projections (default: False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.dropout    = dropout

        # Projections — stubs filled in next commits
        self.qkv_proj   = None  # fused QKV
        self.out_proj   = None  # output projection
        self.attn_drop  = None
        self.resid_drop = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
''')
commit("feat: add CausalSelfAttention class skeleton with constructor and forward stub")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — fused QKV projection + dropout layers
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
src = src.replace(
    "        # Projections — stubs filled in next commits\n"
    "        self.qkv_proj   = None  # fused QKV\n"
    "        self.out_proj   = None  # output projection\n"
    "        self.attn_drop  = None\n"
    "        self.resid_drop = None",
    """\
        # Fused QKV projection: one Linear produces Q, K, V concatenated
        self.qkv_proj   = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)"""
)
write("nanomind/attention/attention.py", src)
commit("feat: add fused QKV projection and dropout layers to CausalSelfAttention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — register causal mask as buffer
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
src = src.replace(
    "        self.attn_drop  = nn.Dropout(dropout)\n"
    "        self.resid_drop = nn.Dropout(dropout)",
    """\
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Pre-computed causal mask registered as a buffer (moves with the model)
        mask = make_causal_mask(block_size, device=torch.device("cpu"))
        self.register_buffer("causal_mask", mask)   # (1, 1, T, T)"""
)
write("nanomind/attention/attention.py", src)
commit("feat: register causal mask as buffer so it moves with model.to(device)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — split_heads() and merge_heads() helpers
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
src = src.replace(
    "    def forward(",
    """\
    # ── Head utilities ────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Reshape ``(B, T, d_model)`` -> ``(B, n_heads, T, head_dim)``.
        \"\"\"
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)   # (B, n_heads, T, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Reshape ``(B, n_heads, T, head_dim)`` -> ``(B, T, d_model)``.
        \"\"\"
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward("""
)
write("nanomind/attention/attention.py", src)
commit("feat: add _split_heads() and _merge_heads() reshaping helpers")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — implement forward() (full attention, no KV-cache)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
src = src.replace(
    "        raise NotImplementedError",
    """\
        \"\"\"
        Compute causal self-attention.

        Args:
            x:        Input tensor ``(B, T, d_model)``
            kv_cache: Optional dict for incremental decoding (see Commit 11).

        Returns:
            Tuple of:
            - ``out``     : attended output ``(B, T, d_model)``
            - ``weights`` : attention weights ``(B, n_heads, T, T)``
        \"\"\"
        B, T, C = x.shape

        # Fused QKV — single matrix multiply then split
        qkv = self.qkv_proj(x)                           # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)        # each (B, T, d_model)

        # Reshape to (B, n_heads, T, head_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply causal mask (sliced to current T)
        mask = self.causal_mask[:, :, :T, :T]

        # Attention
        attn_out, weights = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        # Merge heads and project
        out = self._merge_heads(attn_out)                # (B, T, d_model)
        out = self.resid_drop(self.out_proj(out))
        return out, weights"""
)
write("nanomind/attention/attention.py", src)
commit("feat: implement CausalSelfAttention.forward() with causal mask and fused QKV")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — KV-cache data structure
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/kv_cache.py", '''\
"""
nanomind/attention/kv_cache.py — Key-Value cache for fast autoregressive inference.

During generation, the model runs one token at a time. Without a KV-cache,
the full past key/value tensors would be recomputed every step (O(n^2) time).
With a KV-cache we store past K and V and only compute attention for the
new token against the full history.
"""

from __future__ import annotations

import torch


class KVCache:
    """
    Fixed-capacity key-value cache for one attention layer.

    Grows token-by-token up to ``max_seq_len``. After that, older
    entries are evicted (sliding window).

    Args:
        max_seq_len: Maximum number of past tokens to cache.
        n_heads:     Number of attention heads.
        head_dim:    Dimension of each head.
        device:      Device to allocate tensors on.
        dtype:       Data type for cache tensors.
    """

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.device      = device
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Current number of cached tokens."""
        return 0 if self._k is None else self._k.size(2)

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value vectors and return the full cached tensors.

        Args:
            new_k: New keys   ``(B, n_heads, T_new, head_dim)``
            new_v: New values ``(B, n_heads, T_new, head_dim)``

        Returns:
            ``(cached_k, cached_v)`` — full past + new tensors.
        """
        if self._k is None:
            self._k = new_k
            self._v = new_v
        else:
            self._k = torch.cat([self._k, new_k], dim=2)
            self._v = torch.cat([self._v, new_v], dim=2)

        # Evict oldest if over capacity
        if self._k.size(2) > self.max_seq_len:
            self._k = self._k[:, :, -self.max_seq_len:, :]
            self._v = self._v[:, :, -self.max_seq_len:, :]

        return self._k, self._v

    def reset(self) -> None:
        """Clear the cache (call between independent generation runs)."""
        self._k = None
        self._v = None

    def __repr__(self) -> str:
        return (
            f"KVCache(length={self.length}/{self.max_seq_len}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim})"
        )
''')
commit("feat: add KVCache class — fixed-capacity KV store for fast inference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — forward() with KV-cache support
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
# Replace forward to add kv_cache support
old_fwd = '''\
        B, T, C = x.shape

        # Fused QKV — single matrix multiply then split
        qkv = self.qkv_proj(x)                           # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)        # each (B, T, d_model)

        # Reshape to (B, n_heads, T, head_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply causal mask (sliced to current T)
        mask = self.causal_mask[:, :, :T, :T]

        # Attention
        attn_out, weights = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        # Merge heads and project
        out = self._merge_heads(attn_out)                # (B, T, d_model)
        out = self.resid_drop(self.out_proj(out))
        return out, weights'''

new_fwd = '''\
        B, T, C = x.shape

        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # KV-cache: append new k/v and retrieve full history
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        T_full = k.size(2)   # full cached sequence length

        # Causal mask — only needed when attending to multiple positions
        if T_full > 1:
            mask = self.causal_mask[:, :, :T_full, :T_full]
            # When using cache, query only covers the new token(s)
            if kv_cache is not None and T < T_full:
                mask = mask[:, :, T_full - T:, :]
        else:
            mask = None

        attn_out, weights = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        out = self._merge_heads(attn_out)
        out = self.resid_drop(self.out_proj(out))
        return out, weights'''

src = src.replace(old_fwd, new_fwd)
write("nanomind/attention/attention.py", src)
commit("feat: extend forward() to support KV-cache for fast autoregressive decoding")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — attention weight visualization hook
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/attention.py")
src += '''
    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_attention_map(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return attention weight matrix without applying output projection.

        Useful for visualizing which tokens the model attends to.

        Args:
            x: Input ``(B, T, d_model)``

        Returns:
            Attention weights ``(B, n_heads, T, T)``
        """
        with torch.no_grad():
            _, weights = self.forward(x)
        return weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout}"
        )
'''
write("nanomind/attention/attention.py", src)
commit("feat: add get_attention_map() diagnostic method for attention visualization")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — use torch SDPA when available (Flash Attention path)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/attention/functional.py")
src += '''

def fast_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Dispatch to PyTorch's built-in SDPA when available (>=2.0).

    On PyTorch 2.0+ with CUDA, this uses Flash Attention or memory-efficient
    attention automatically. Falls back to our manual implementation otherwise.

    Args:
        q, k, v:   Query, key, value tensors ``(B, n_heads, T, head_dim)``
        mask:      Boolean causal mask ``(1, 1, T, T)``
        dropout_p: Dropout probability
        training:  Training mode flag

    Returns:
        Output tensor ``(B, n_heads, T, head_dim)``
    """
    use_builtin = hasattr(F, "scaled_dot_product_attention")
    if use_builtin:
        # PyTorch 2.0+ path — may use Flash Attention internally
        is_causal = mask is not None
        attn_mask = None if is_causal else mask
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=is_causal,
        )
    # Fallback
    out, _ = scaled_dot_product_attention(q, k, v, mask, dropout_p, training)
    return out
'''
write("nanomind/attention/functional.py", src)
commit("perf: add fast_scaled_dot_product_attention() using torch SDPA (Flash Attention path)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — AttentionConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/config.py", '''\
"""
nanomind/attention/config.py — Configuration for the attention mechanism.
"""

from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """
    Configuration for :class:`~nanomind.attention.CausalSelfAttention`.

    Attributes:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        dropout:    Attention and residual dropout probability.
        bias:       Whether to add bias to Q/K/V and output projections.
        use_flash:  Whether to use PyTorch 2.0 SDPA (Flash Attention) when available.
    """
    d_model: int    = 128
    n_heads: int    = 4
    block_size: int = 128
    dropout: float  = 0.1
    bias: bool      = False
    use_flash: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads
''')
commit("feat: add AttentionConfig dataclass with validation and head_dim property")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update attention __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/attention/__init__.py", '''\
"""NanoMind attention sub-package.

Core components:
    - :class:`CausalSelfAttention` — multi-head causal self-attention layer
    - :class:`KVCache`             — key-value cache for fast inference
    - :class:`AttentionConfig`     — attention configuration dataclass
    - :func:`make_causal_mask`     — causal mask utility
    - :func:`scaled_dot_product_attention` — pure-function attention math
"""

from nanomind.attention.attention import CausalSelfAttention
from nanomind.attention.kv_cache import KVCache
from nanomind.attention.config import AttentionConfig
from nanomind.attention.functional import (
    scaled_dot_product_attention,
    fast_scaled_dot_product_attention,
    make_causal_mask,
)

__all__ = [
    "CausalSelfAttention",
    "KVCache",
    "AttentionConfig",
    "scaled_dot_product_attention",
    "fast_scaled_dot_product_attention",
    "make_causal_mask",
]
''')
commit("refactor: export all attention components from nanomind/attention/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: attention output shapes
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_attention.py", '''\
"""
tests/test_attention.py — Tests for the NanoMind attention mechanism.
"""

import pytest
import torch

from nanomind.attention import (
    CausalSelfAttention,
    KVCache,
    AttentionConfig,
    make_causal_mask,
    scaled_dot_product_attention,
)

B, T, D, H = 2, 16, 64, 4   # batch, seq_len, d_model, n_heads


@pytest.fixture
def attn() -> CausalSelfAttention:
    return CausalSelfAttention(d_model=D, n_heads=H, block_size=T, dropout=0.0)


# ── Output shapes ─────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_forward_output_shape(self, attn):
        x = torch.randn(B, T, D)
        out, weights = attn(x)
        assert out.shape == (B, T, D)

    def test_attention_weights_shape(self, attn):
        x = torch.randn(B, T, D)
        _, weights = attn(x)
        assert weights.shape == (B, H, T, T)

    def test_single_token(self, attn):
        x = torch.randn(B, 1, D)
        out, _ = attn(x)
        assert out.shape == (B, 1, D)

    def test_full_block_size(self, attn):
        x = torch.randn(B, T, D)
        out, _ = attn(x)
        assert out.shape == (B, T, D)
''')
commit("test: add attention output shape tests (forward, weights, single token)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: causal mask correctness
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_attention.py")
src += '''

# ── Causal mask ───────────────────────────────────────────────────────────────

class TestCausalMask:
    def test_shape(self):
        mask = make_causal_mask(8, torch.device("cpu"))
        assert mask.shape == (1, 1, 8, 8)

    def test_lower_triangle_is_false(self):
        mask = make_causal_mask(4, torch.device("cpu")).squeeze()
        # Lower triangle (and diagonal) should be False (allowed to attend)
        for i in range(4):
            for j in range(i + 1):
                assert not mask[i, j].item(), f"Position ({i},{j}) should not be masked"

    def test_upper_triangle_is_true(self):
        mask = make_causal_mask(4, torch.device("cpu")).squeeze()
        # Upper triangle should be True (masked out)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j].item(), f"Position ({i},{j}) should be masked"

    def test_attention_is_causal(self, attn):
        # Token 0 should not be influenced by token 1 (future)
        x = torch.zeros(1, T, D)
        x[0, 0] = 1.0   # only token 0 is non-zero
        x[0, 1] = 2.0   # future token
        out_full, _ = attn(x)
        x2 = x.clone(); x2[0, 1] = 999.0   # change future token drastically
        out_changed, _ = attn(x2)
        # Token 0 output should be identical regardless of future tokens
        assert torch.allclose(out_full[0, 0], out_changed[0, 0], atol=1e-5)
'''
write("tests/test_attention.py", src)
commit("test: add causal mask correctness tests (lower/upper triangle, causal property)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: multi-head split/merge
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_attention.py")
src += '''

# ── Head splitting / merging ──────────────────────────────────────────────────

class TestHeadSplitMerge:
    def test_split_shape(self, attn):
        x = torch.randn(B, T, D)
        split = attn._split_heads(x)
        assert split.shape == (B, H, T, D // H)

    def test_merge_shape(self, attn):
        x = torch.randn(B, H, T, D // H)
        merged = attn._merge_heads(x)
        assert merged.shape == (B, T, D)

    def test_split_merge_roundtrip(self, attn):
        x = torch.randn(B, T, D)
        roundtrip = attn._merge_heads(attn._split_heads(x))
        assert torch.allclose(x, roundtrip)
'''
write("tests/test_attention.py", src)
commit("test: add multi-head split/merge shape and roundtrip tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: KV-cache correctness
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_attention.py")
src += '''

# ── KV-Cache ──────────────────────────────────────────────────────────────────

class TestKVCache:
    def test_initial_length_zero(self):
        cache = KVCache(max_seq_len=32, n_heads=H, head_dim=D//H,
                        device=torch.device("cpu"))
        assert cache.length == 0

    def test_length_grows_after_update(self):
        cache = KVCache(max_seq_len=32, n_heads=H, head_dim=D//H,
                        device=torch.device("cpu"))
        k = torch.randn(1, H, 5, D//H)
        v = torch.randn(1, H, 5, D//H)
        cache.update(k, v)
        assert cache.length == 5

    def test_eviction_at_capacity(self):
        cache = KVCache(max_seq_len=4, n_heads=H, head_dim=D//H,
                        device=torch.device("cpu"))
        k = torch.randn(1, H, 6, D//H)
        v = torch.randn(1, H, 6, D//H)
        cache.update(k, v)
        assert cache.length == 4   # evicted 2 oldest

    def test_reset_clears_cache(self):
        cache = KVCache(max_seq_len=32, n_heads=H, head_dim=D//H,
                        device=torch.device("cpu"))
        k = torch.randn(1, H, 5, D//H)
        v = torch.randn(1, H, 5, D//H)
        cache.update(k, v)
        cache.reset()
        assert cache.length == 0
'''
write("tests/test_attention.py", src)
commit("test: add KVCache length tracking, eviction, and reset tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: pure attention function
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_attention.py")
src += '''

# ── Pure attention function ───────────────────────────────────────────────────

class TestScaledDotProductAttention:
    def test_output_shape(self):
        q = torch.randn(B, H, T, D // H)
        k = torch.randn(B, H, T, D // H)
        v = torch.randn(B, H, T, D // H)
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape     == (B, H, T, D // H)
        assert weights.shape == (B, H, T, T)

    def test_weights_sum_to_one(self):
        q = torch.randn(B, H, T, D // H)
        k = torch.randn(B, H, T, D // H)
        v = torch.randn(B, H, T, D // H)
        _, weights = scaled_dot_product_attention(q, k, v)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_mask_prevents_future_attention(self):
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = make_causal_mask(4, torch.device("cpu"))
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        # Upper triangle should be (near) zero after softmax
        upper = weights[0, 0, 0, 1]   # position 0 attending to position 1
        assert upper.item() < 1e-6
'''
write("tests/test_attention.py", src)
commit("test: add scaled_dot_product_attention output shape and mask tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 5 | Attention mechanism | 🔜 |",
    "| 5 | Attention mechanism | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Data pipeline: TextDataset, IterableTextDataset, DataLoaders, PrefetchLoader, stats (Day 4)",
    "- Data pipeline: TextDataset, IterableTextDataset, DataLoaders, PrefetchLoader, stats (Day 4)\n- Attention: CausalSelfAttention, KVCache, causal mask, Flash Attention dispatch (Day 5)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 5 complete in README and CHANGELOG")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 5 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 5 COMPLETE ===")
