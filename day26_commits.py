"""
day26_commits.py — 20 atomic commits for Day 26: Flash Attention.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

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

print("\n=== DAY 26: Flash Attention — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — flash package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/__init__.py",
      '"""NanoMind Flash Attention sub-package — tiled O(N) memory attention."""\n')
commit("feat: add nanomind/flash/ package skeleton for Flash Attention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — FlashConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/config.py", '''\
"""
nanomind/flash/config.py — Flash Attention configuration.

Standard attention materialises a full (N × N) attention weight matrix:

  Memory: O(N²)  — explodes for long sequences (N=32k → 1B floats per head)
  Speed:  O(N²)  — dominated by slow HBM reads/writes of the attention matrix

Flash Attention (Dao et al. 2022 / 2023) computes the exact same output but
avoids ever writing the full N×N matrix to HBM (GPU global memory).
Instead it tiles Q, K, V into SRAM-resident blocks and uses online softmax
to accumulate the output block by block:

  Memory: O(N)   — only blocks in SRAM; output O(N)
  Speed:  ~2-4× faster than standard attention in practice
  Math:   Exactly equivalent output to standard attention

Reference: Dao et al. (2022) "FlashAttention" — https://arxiv.org/abs/2205.14135
           Dao et al. (2023) "FlashAttention-2" — https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FlashConfig:
    """
    Configuration for Flash Attention.

    Attributes:
        block_q:    Query tile size (number of query rows per SRAM block).
        block_kv:   Key/Value tile size (number of K/V columns per SRAM block).
        causal:     Apply causal (lower-triangular) mask.
        dropout:    Attention dropout probability (training only).
        use_torch_sdpa: Use PyTorch\'s built-in scaled_dot_product_attention
                        (flash-efficient on CUDA) when available. Falls back
                        to the pure-Python tiled implementation otherwise.
    """

    block_q:         int   = 64
    block_kv:        int   = 64
    causal:          bool  = True
    dropout:         float = 0.0
    use_torch_sdpa:  bool  = True   # Use torch.nn.functional.scaled_dot_product_attention

    def __post_init__(self) -> None:
        assert self.block_q  >= 1
        assert self.block_kv >= 1
        assert 0.0 <= self.dropout <= 1.0
''')
commit("feat: add FlashConfig — block_q, block_kv, causal, use_torch_sdpa with docstring")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — online softmax (safe log-sum-exp accumulation)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/online_softmax.py", '''\
"""
nanomind/flash/online_softmax.py — Online (streaming) softmax for Flash Attention.

Standard softmax requires two passes over the data:
  Pass 1: find max(x_i) for numerical stability
  Pass 2: compute exp(x_i - max) / Σ exp(x_j - max)

Online softmax (Milakov & Gimelshein, 2018) merges both passes into one,
keeping running statistics (max and sum) that can be updated incrementally
as new tiles of K/V are processed.

This is the mathematical heart of Flash Attention: it lets us update the
running output accumulator O and the normalization constant l with each
new tile, without ever needing the full softmax row at once.

State after processing tile t:
  m_t = max(s_1 ... s_t)   — running max of scores seen so far
  l_t = Σ exp(s_i - m_t)   — running normalisation denominator
  O_t = output accumulator

When a new tile t+1 arrives with block scores S_{t+1}:
  m_{t+1} = max(m_t, max(S_{t+1}))
  l_{t+1} = l_t × exp(m_t - m_{t+1}) + Σ exp(S_{t+1,j} - m_{t+1})
  O_{t+1} = (O_t × l_t × exp(m_t - m_{t+1}) + exp(S_{t+1}) × V_{t+1}) / l_{t+1}
"""

from __future__ import annotations

import torch


class OnlineSoftmaxState:
    """
    Running state for online (streaming) softmax accumulation.

    Holds the running max ``m``, normalisation sum ``l``, and
    output accumulator ``O`` for a batch of query rows.

    Args:
        q_block: Query block ``(B, H, Bq, Dh)`` used to infer shape and device.
    """

    __slots__ = ("m", "l", "O")

    def __init__(self, q_block: torch.Tensor) -> None:
        B, H, Bq, Dh = q_block.shape
        device, dtype = q_block.device, q_block.dtype
        self.m = torch.full((B, H, Bq, 1), float("-inf"), device=device, dtype=dtype)
        self.l = torch.zeros((B, H, Bq, 1), device=device, dtype=dtype)
        self.O = torch.zeros((B, H, Bq, Dh), device=device, dtype=dtype)

    def update(
        self,
        s_block: torch.Tensor,   # (B, H, Bq, Bkv) raw scores for this KV tile
        v_block: torch.Tensor,   # (B, H, Bkv, Dh)
    ) -> None:
        """
        Incorporate a new KV tile into the running output.

        Args:
            s_block: Attention scores for this tile ``(B, H, Bq, Bkv)``.
            v_block: Value tile ``(B, H, Bkv, Dh)``.
        """
        # New running max
        m_new = torch.maximum(self.m, s_block.max(dim=-1, keepdim=True).values)

        # Rescale existing accumulator and l
        scale_old = torch.exp(self.m - m_new)
        exp_s     = torch.exp(s_block - m_new)

        l_new = self.l * scale_old + exp_s.sum(dim=-1, keepdim=True)
        O_new = self.O * scale_old + torch.matmul(exp_s, v_block)

        self.m = m_new
        self.l = l_new
        self.O = O_new

    def finalize(self) -> torch.Tensor:
        """
        Return the final normalised attention output.

        Returns:
            ``(B, H, Bq, Dh)`` output tensor.
        """
        return self.O / (self.l + 1e-8)
''')
commit("feat: add OnlineSoftmaxState — streaming max/sum accumulator for tile-based attention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — tiled flash_attention_forward (pure Python reference)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/tiled.py", '''\
"""
nanomind/flash/tiled.py — Pure-Python tiled Flash Attention (reference implementation).

This is a pedagogical reference implementation of the Flash Attention algorithm
in pure PyTorch. It is mathematically identical to standard scaled dot-product
attention but processes K/V in tiles to demonstrate the O(N) memory property.

Performance note:
  This Python-level tiling is NOT faster than standard attention on GPU —
  it lacks the CUDA kernel fusion and SRAM management of the real Flash Attention.
  For GPU performance, use FlashConfig(use_torch_sdpa=True) which delegates to
  PyTorch\'s built-in flash-efficient SDPA.

  This reference is valuable for:
    1. Educational understanding of the algorithm
    2. CPU fallback
    3. Numerical verification of the torch.sdpa path

Reference: Dao et al. (2022) Algorithm 1, https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import math
import torch

from nanomind.flash.online_softmax import OnlineSoftmaxState


def tiled_flash_attention(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    block_q:    int  = 64,
    block_kv:   int  = 64,
    causal:     bool = True,
    scale:      float | None = None,
) -> torch.Tensor:
    """
    Tiled Flash Attention — pure PyTorch reference implementation.

    Computes scaled dot-product attention in O(N) memory by processing K/V
    in tiles and accumulating the output using online softmax.

    Args:
        q:        Query  ``(B, H, N, Dh)``
        k:        Key    ``(B, H, N, Dh)``
        v:        Value  ``(B, H, N, Dh)``
        block_q:  Number of query rows per tile.
        block_kv: Number of K/V columns per tile.
        causal:   Apply causal mask (future tokens → -inf).
        scale:    Attention scale (default: 1/√Dh).

    Returns:
        Output ``(B, H, N, Dh)`` — identical to standard SDPA.
    """
    B, H, N, Dh = q.shape
    scale        = scale or (Dh ** -0.5)
    output       = torch.empty_like(q)

    # Iterate over query tiles
    for q_start in range(0, N, block_q):
        q_end   = min(q_start + block_q, N)
        q_block = q[:, :, q_start:q_end, :]   # (B, H, Bq, Dh)

        state   = OnlineSoftmaxState(q_block)

        # Iterate over K/V tiles
        for kv_start in range(0, N, block_kv):
            kv_end   = min(kv_start + block_kv, N)
            k_block  = k[:, :, kv_start:kv_end, :]   # (B, H, Bkv, Dh)
            v_block  = v[:, :, kv_start:kv_end, :]   # (B, H, Bkv, Dh)

            # Causal: mask future K/V positions relative to query positions
            if causal and kv_start >= q_end:
                continue   # entire KV tile is in the future → skip

            # Scores: (B, H, Bq, Bkv)
            s = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            if causal:
                # Build per-block causal mask
                q_idx  = torch.arange(q_start, q_end,  device=q.device).unsqueeze(1)
                kv_idx = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
                mask   = kv_idx > q_idx   # future positions
                s      = s.masked_fill(mask, float("-inf"))

            state.update(s, v_block)

        output[:, :, q_start:q_end, :] = state.finalize()

    return output
''')
commit("feat: add tiled_flash_attention() — pure-PyTorch tiled O(N) memory reference implementation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — memory footprint analysis
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/memory.py", '''\
"""
nanomind/flash/memory.py — Memory footprint analysis for standard vs Flash Attention.
"""

from __future__ import annotations

import math


def standard_attention_memory(
    batch:    int,
    heads:    int,
    seq_len:  int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> dict:
    """
    Compute theoretical memory usage of standard attention.

    Materialises a full (N × N) attention weight matrix per head.

    Args:
        batch, heads, seq_len, head_dim: Model dimensions.
        dtype_bytes: Bytes per element (4=float32, 2=float16).

    Returns:
        Dict with ``qkv_bytes``, ``attn_matrix_bytes``, ``output_bytes``, ``total_bytes``, ``total_mb``.
    """
    qkv_bytes    = 3 * batch * heads * seq_len * head_dim * dtype_bytes
    attn_bytes   = batch * heads * seq_len * seq_len * dtype_bytes     # N×N matrix
    output_bytes = batch * heads * seq_len * head_dim * dtype_bytes
    total        = qkv_bytes + attn_bytes + output_bytes
    return {
        "qkv_bytes":         qkv_bytes,
        "attn_matrix_bytes": attn_bytes,
        "output_bytes":      output_bytes,
        "total_bytes":       total,
        "total_mb":          total / (1024 ** 2),
    }


def flash_attention_memory(
    batch:    int,
    heads:    int,
    seq_len:  int,
    head_dim: int,
    block_kv: int = 64,
    dtype_bytes: int = 4,
) -> dict:
    """
    Compute theoretical memory usage of Flash Attention.

    Only materialises two tiles (Q block + KV block) at a time,
    plus the O(N) output buffer.

    Args:
        batch, heads, seq_len, head_dim: Model dimensions.
        block_kv: KV tile size.
        dtype_bytes: Bytes per element.

    Returns:
        Dict with ``qkv_bytes``, ``tile_bytes``, ``output_bytes``, ``total_bytes``, ``total_mb``.
    """
    qkv_bytes    = 3 * batch * heads * seq_len * head_dim * dtype_bytes
    # SRAM: one Q tile + one KV tile at a time
    tile_bytes   = batch * heads * block_kv * head_dim * dtype_bytes * 3   # Q + K + V tiles
    output_bytes = batch * heads * seq_len * head_dim * dtype_bytes
    total        = qkv_bytes + tile_bytes + output_bytes
    return {
        "qkv_bytes":   qkv_bytes,
        "tile_bytes":  tile_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total,
        "total_mb":    total / (1024 ** 2),
    }


def memory_comparison_report(
    seq_len: int,
    batch:   int = 1,
    heads:   int = 8,
    head_dim:int = 64,
) -> str:
    """
    Generate a string report comparing standard vs Flash Attention memory.

    Args:
        seq_len:  Sequence length to analyse.
        batch:    Batch size.
        heads:    Number of attention heads.
        head_dim: Head dimension.

    Returns:
        Formatted report string.
    """
    std   = standard_attention_memory(batch, heads, seq_len, head_dim)
    flash = flash_attention_memory(batch, heads, seq_len, head_dim)
    ratio = std["total_bytes"] / max(flash["total_bytes"], 1)

    lines = [
        f"Memory Comparison  (N={seq_len}, B={batch}, H={heads}, Dh={head_dim})",
        "-" * 60,
        f"  Standard attention  : {std['total_mb']:>8.2f} MB  "
        f"  (attn matrix: {std['attn_matrix_bytes']/(1024**2):.2f} MB)",
        f"  Flash attention     : {flash['total_mb']:>8.2f} MB  "
        f"  (tile buffer: {flash['tile_bytes']/(1024**2):.4f} MB)",
        f"  Memory saving       : {ratio:.1f}× less memory with Flash Attention",
    ]
    return "\n".join(lines)
''')
commit("feat: add standard_attention_memory(), flash_attention_memory(), memory_comparison_report()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — FlashAttention nn.Module
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/module.py", '''\
"""
nanomind/flash/module.py — FlashAttention as an nn.Module.

Provides a drop-in replacement for standard multi-head self-attention
that uses either:
  1. torch.nn.functional.scaled_dot_product_attention (CUDA flash-efficient)
  2. The pure-PyTorch tiled reference implementation (CPU / educational)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.flash.config import FlashConfig
from nanomind.flash.tiled import tiled_flash_attention


class FlashAttention(nn.Module):
    """
    Multi-head self-attention with Flash Attention backend.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` when
    ``cfg.use_torch_sdpa=True`` (CUDA flash-efficient on compatible hardware)
    and falls back to the tiled reference implementation otherwise.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        cfg:     Flash Attention configuration.
        bias:    Use bias in Q/K/V/out projections.

    Example::

        attn = FlashAttention(256, 8, FlashConfig(causal=True))
        x    = torch.randn(2, 128, 256)
        out, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cfg:     FlashConfig | None = None,
        bias:    bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.cfg      = cfg or FlashConfig()
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop     = nn.Dropout(self.cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """
        Flash attention forward pass.

        Args:
            x:    Input ``(B, T, d_model)``.
            mask: Optional additive mask ``(B, 1, T, T)`` or ``(T, T)``.

        Returns:
            Tuple of ``(output, None)`` — None for API compatibility.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if self.cfg.use_torch_sdpa:
            # PyTorch built-in flash-efficient SDPA
            attn_mask = mask
            if attn_mask is None and self.cfg.causal:
                attn_mask = None   # is_causal handles it
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=self.cfg.causal if attn_mask is None else False,
                scale=self.scale,
            )
        else:
            # Pure-Python tiled reference
            out = tiled_flash_attention(
                q, k, v,
                block_q=self.cfg.block_q,
                block_kv=self.cfg.block_kv,
                causal=self.cfg.causal,
                scale=self.scale,
            )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), None

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"causal={self.cfg.causal}, "
            f"backend={'torch_sdpa' if self.cfg.use_torch_sdpa else 'tiled'}"
        )
''')
commit("feat: add FlashAttention nn.Module — torch.sdpa + tiled fallback, drop-in for SDPA")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — FlashTransformerBlock
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/block.py", '''\
"""
nanomind/flash/block.py — Transformer block with FlashAttention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.flash.module import FlashAttention
from nanomind.flash.config import FlashConfig
from nanomind.norm.factory import get_norm


class FlashTransformerBlock(nn.Module):
    """
    Pre-norm transformer block using FlashAttention instead of standard SDPA.

    Architecture:
        x → RMSNorm → FlashAttention → x + residual
        x → RMSNorm → SwiGLU FFN    → x + residual

    Args:
        d_model:   Embedding dimension.
        n_heads:   Number of attention heads.
        flash_cfg: Flash Attention configuration.
        dropout:   Dropout for attention and residuals.
        norm_type: Normalisation (``"rmsnorm"`` or ``"layernorm"``).
        bias:      Use bias in projections.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        flash_cfg: FlashConfig | None = None,
        dropout:   float = 0.0,
        norm_type: str   = "rmsnorm",
        bias:      bool  = False,
    ) -> None:
        super().__init__()
        d_ff = int(d_model * 8 / 3)   # SwiGLU typical d_ff

        self.norm1  = get_norm(norm_type, d_model)
        self.attn   = FlashAttention(d_model, n_heads, flash_cfg, bias)
        self.norm2  = get_norm(norm_type, d_model)
        self.gate   = nn.Linear(d_model, d_ff, bias=bias)
        self.up     = nn.Linear(d_model, d_ff, bias=bias)
        self.down   = nn.Linear(d_ff, d_model, bias=bias)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flash attention
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.drop(self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h)))
        return x
''')
commit("feat: add FlashTransformerBlock — RMSNorm + FlashAttention + SwiGLU FFN")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — NanoMindFlash model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/model.py", '''\
"""
nanomind/flash/model.py — NanoMind transformer with Flash Attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.model.config import ModelConfig
from nanomind.flash.block import FlashTransformerBlock
from nanomind.flash.config import FlashConfig
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("flash.model")


class NanoMindFlash(nn.Module):
    """
    NanoMind transformer using Flash Attention in every block.

    Drop-in replacement for the standard NanoMind model, using
    :class:`FlashTransformerBlock` with RMSNorm and SwiGLU FFN.

    Args:
        model_cfg:  Standard model configuration.
        flash_cfg:  Flash Attention configuration.

    Example::

        model = NanoMindFlash(
            ModelConfig(vocab_size=32000, d_model=512, n_layers=8, n_heads=8),
            FlashConfig(causal=True, use_torch_sdpa=True),
        )
        logits, loss = model(input_ids, targets)
    """

    def __init__(
        self,
        model_cfg:  ModelConfig,
        flash_cfg:  FlashConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg       = model_cfg
        self.flash_cfg = flash_cfg or FlashConfig()

        self.tok_emb  = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_emb  = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        self.drop     = nn.Dropout(model_cfg.dropout)
        self.blocks   = nn.ModuleList([
            FlashTransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                flash_cfg=self.flash_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm     = get_norm("rmsnorm", model_cfg.d_model)
        self.lm_head  = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        backend = "torch_sdpa" if self.flash_cfg.use_torch_sdpa else "tiled"
        log.info(f"NanoMindFlash: {n:,} params | backend={backend}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok  = self.tok_emb(idx)
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(tok + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
''')
commit("feat: add NanoMindFlash — full transformer with FlashAttention in every block")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — register flash in attention factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/pos/factory.py")
if "flash" not in src:
    src = src.rstrip() + '''

# Flash Attention is registered separately via get_flash_attention()
def get_flash_attention(d_model: int, n_heads: int, causal: bool = True):
    """
    Return a FlashAttention module. Import here to avoid circular imports.
    """
    from nanomind.flash.module import FlashAttention
    from nanomind.flash.config import FlashConfig
    return FlashAttention(d_model, n_heads, FlashConfig(causal=causal))
'''
    write("nanomind/pos/factory.py", src)
    commit("feat: add get_flash_attention() to attention factory for unified access")
else:
    commit("feat: add get_flash_attention() to attention factory for unified access")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update flash __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/flash/__init__.py", '''\
"""NanoMind Flash Attention sub-package.

Flash Attention avoids materialising the O(N²) attention weight matrix
by processing K/V in tiles using online softmax accumulation.

  Standard attention memory : O(N²) — the attention matrix
  Flash Attention memory    : O(N)  — only tile buffers in SRAM

Primary exports:
    - :class:`NanoMindFlash`        — full transformer with Flash Attention
    - :class:`FlashAttention`       — drop-in SDPA replacement (torch.sdpa / tiled)
    - :class:`FlashTransformerBlock`— RMSNorm + FlashAttention + SwiGLU block
    - :class:`FlashConfig`          — block_q, block_kv, causal, use_torch_sdpa
    - :class:`OnlineSoftmaxState`   — streaming max/sum accumulator
    - :func:`tiled_flash_attention` — pure-PyTorch reference implementation
    - :func:`standard_attention_memory`  — theoretical standard SDPA memory
    - :func:`flash_attention_memory`     — theoretical Flash Attention memory
    - :func:`memory_comparison_report`   — print N×N vs tile memory analysis
"""

from nanomind.flash.config import FlashConfig
from nanomind.flash.online_softmax import OnlineSoftmaxState
from nanomind.flash.tiled import tiled_flash_attention
from nanomind.flash.memory import (
    standard_attention_memory,
    flash_attention_memory,
    memory_comparison_report,
)
from nanomind.flash.module import FlashAttention
from nanomind.flash.block import FlashTransformerBlock
from nanomind.flash.model import NanoMindFlash

__all__ = [
    "FlashConfig",
    "OnlineSoftmaxState",
    "tiled_flash_attention",
    "standard_attention_memory",
    "flash_attention_memory",
    "memory_comparison_report",
    "FlashAttention",
    "FlashTransformerBlock",
    "NanoMindFlash",
]
''')
commit("refactor: export all Flash Attention components from nanomind/flash/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: flash_attention_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/flash_attention_demo.py", '''\
"""
examples/flash_attention_demo.py — Flash Attention demo.

Demonstrates:
  1. Memory comparison: standard vs Flash Attention
  2. Numerical equivalence of tiled vs torch.sdpa output
  3. NanoMindFlash forward pass

Usage:
    python examples/flash_attention_demo.py
"""

import torch
from nanomind.model.config import ModelConfig
from nanomind.flash import (
    FlashConfig, FlashAttention, NanoMindFlash,
    tiled_flash_attention, memory_comparison_report,
    standard_attention_memory, flash_attention_memory,
)

# ── 1. Memory analysis ────────────────────────────────────────────────────────
for N in [512, 2048, 8192]:
    print(memory_comparison_report(N, batch=1, heads=8, head_dim=64))
    print()

# ── 2. Numerical equivalence ──────────────────────────────────────────────────
torch.manual_seed(42)
B, H, N, Dh = 1, 4, 64, 32
q = torch.randn(B, H, N, Dh)
k = torch.randn(B, H, N, Dh)
v = torch.randn(B, H, N, Dh)
scale = Dh ** -0.5

# Standard SDPA
import torch.nn.functional as F
std_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

# Tiled reference
tile_out = tiled_flash_attention(q, k, v, block_q=16, block_kv=16,
                                  causal=True, scale=scale)

max_diff = (std_out - tile_out).abs().max().item()
print(f"Max diff (tiled vs torch.sdpa): {max_diff:.2e}  (should be < 1e-4)")

# ── 3. NanoMindFlash forward ──────────────────────────────────────────────────
model_cfg = ModelConfig(vocab_size=256, block_size=64, d_model=64,
                        n_layers=2, n_heads=4, dropout=0.0)

# torch.sdpa backend
model_fast = NanoMindFlash(model_cfg, FlashConfig(use_torch_sdpa=True))
# Tiled reference backend
model_tile = NanoMindFlash(model_cfg, FlashConfig(use_torch_sdpa=False))

idx = torch.randint(0, 256, (1, 32))
with torch.no_grad():
    logits_fast, _ = model_fast(idx)
    logits_tile, _ = model_tile(idx)

print(f"\nNanoMindFlash (torch_sdpa) logits shape: {logits_fast.shape}")
print(f"NanoMindFlash (tiled)      logits shape: {logits_tile.shape}")
print(f"Total params: {model_fast.num_parameters():,}")
''')
commit("feat: add examples/flash_attention_demo.py — memory analysis, equivalence, forward demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: FlashConfig
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_flash.py", '''\
"""
tests/test_flash.py — Tests for Flash Attention.
"""

import pytest
import torch
import torch.nn.functional as F

from nanomind.flash import (
    FlashConfig, FlashAttention, FlashTransformerBlock, NanoMindFlash,
    tiled_flash_attention, OnlineSoftmaxState,
    standard_attention_memory, flash_attention_memory, memory_comparison_report,
)
from nanomind.model.config import ModelConfig

B, H, N, Dh = 2, 4, 32, 16
D = H * Dh


def make_qkv():
    torch.manual_seed(0)
    q = torch.randn(B, H, N, Dh)
    k = torch.randn(B, H, N, Dh)
    v = torch.randn(B, H, N, Dh)
    return q, k, v


# ── FlashConfig ───────────────────────────────────────────────────────────────

class TestFlashConfig:
    def test_defaults(self):
        cfg = FlashConfig()
        assert cfg.block_q  == 64
        assert cfg.causal   is True
        assert cfg.use_torch_sdpa is True

    def test_invalid_block_q(self):
        with pytest.raises(AssertionError):
            FlashConfig(block_q=0)

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            FlashConfig(dropout=1.5)
''')
commit("test: add FlashConfig defaults, invalid block_q, and invalid dropout tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: OnlineSoftmaxState
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── OnlineSoftmaxState ────────────────────────────────────────────────────────

class TestOnlineSoftmax:
    def test_init_shapes(self):
        q   = torch.randn(B, H, N, Dh)
        st  = OnlineSoftmaxState(q)
        assert st.m.shape == (B, H, N, 1)
        assert st.l.shape == (B, H, N, 1)
        assert st.O.shape == (B, H, N, Dh)

    def test_single_tile_matches_softmax(self):
        """With one full tile, online softmax == standard softmax."""
        q, k, v = make_qkv()
        scale   = Dh ** -0.5
        s       = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B,H,N,N)

        st = OnlineSoftmaxState(q)
        st.update(s, v)
        online_out = st.finalize()

        # Standard attention output
        std_out = F.softmax(s, dim=-1) @ v
        assert torch.allclose(online_out, std_out, atol=1e-5)

    def test_two_tiles_matches_softmax(self):
        """Splitting K/V into two tiles should give the same result."""
        q, k, v = make_qkv()
        scale   = Dh ** -0.5

        # Standard
        s       = torch.matmul(q, k.transpose(-2, -1)) * scale
        std_out = F.softmax(s, dim=-1) @ v

        # Two-tile online
        half = N // 2
        st   = OnlineSoftmaxState(q)
        s1   = torch.matmul(q, k[:, :, :half].transpose(-2, -1)) * scale
        st.update(s1, v[:, :, :half])
        s2   = torch.matmul(q, k[:, :, half:].transpose(-2, -1)) * scale
        st.update(s2, v[:, :, half:])
        online_out = st.finalize()

        assert torch.allclose(online_out, std_out, atol=1e-4)
'''
write("tests/test_flash.py", src)
commit("test: add OnlineSoftmaxState init shape, single tile, and two-tile softmax tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: tiled_flash_attention vs standard SDPA
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── tiled_flash_attention ─────────────────────────────────────────────────────

class TestTiledFlashAttention:
    def test_output_shape(self):
        q, k, v = make_qkv()
        out = tiled_flash_attention(q, k, v, block_q=8, block_kv=8)
        assert out.shape == (B, H, N, Dh)

    def test_causal_matches_sdpa(self):
        """Tiled implementation should match torch.sdpa with is_causal=True."""
        q, k, v  = make_qkv()
        scale    = Dh ** -0.5
        ref      = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        tiled    = tiled_flash_attention(q, k, v, block_q=8, block_kv=8,
                                          causal=True, scale=scale)
        assert torch.allclose(ref, tiled, atol=1e-4), \
            f"Max diff: {(ref-tiled).abs().max():.2e}"

    def test_non_causal_matches_sdpa(self):
        q, k, v = make_qkv()
        scale   = Dh ** -0.5
        ref     = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
        tiled   = tiled_flash_attention(q, k, v, block_q=8, block_kv=8,
                                         causal=False, scale=scale)
        assert torch.allclose(ref, tiled, atol=1e-4)

    def test_output_finite(self):
        q, k, v = make_qkv()
        out = tiled_flash_attention(q, k, v, block_q=8, block_kv=8, causal=True)
        assert out.isfinite().all()
'''
write("tests/test_flash.py", src)
commit("test: add tiled_flash_attention output shape, causal/non-causal vs sdpa, finite tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: FlashAttention module
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── FlashAttention module ─────────────────────────────────────────────────────

class TestFlashAttentionModule:
    def test_output_shape_sdpa(self):
        attn = FlashAttention(D, H, FlashConfig(use_torch_sdpa=True))
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.shape == (B, N, D)

    def test_output_shape_tiled(self):
        attn = FlashAttention(D, H, FlashConfig(use_torch_sdpa=False))
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.shape == (B, N, D)

    def test_sdpa_and_tiled_close(self):
        """Both backends should produce numerically close outputs."""
        torch.manual_seed(1)
        x     = torch.randn(1, N, D)
        attn1 = FlashAttention(D, H, FlashConfig(use_torch_sdpa=True))
        attn2 = FlashAttention(D, H, FlashConfig(use_torch_sdpa=False))
        # Copy weights
        attn2.load_state_dict(attn1.state_dict())
        with torch.no_grad():
            out1, _ = attn1(x)
            out2, _ = attn2(x)
        assert torch.allclose(out1, out2, atol=1e-4), \
            f"Max diff: {(out1-out2).abs().max():.2e}"

    def test_output_finite(self):
        attn = FlashAttention(D, H)
        x    = torch.randn(B, N, D)
        out, _ = attn(x)
        assert out.isfinite().all()
'''
write("tests/test_flash.py", src)
commit("test: add FlashAttention module shape, sdpa/tiled backends, and numerical closeness tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: FlashTransformerBlock
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── FlashTransformerBlock ─────────────────────────────────────────────────────

class TestFlashTransformerBlock:
    def test_output_shape(self):
        block = FlashTransformerBlock(D, H, FlashConfig(use_torch_sdpa=True))
        x     = torch.randn(B, N, D)
        out   = block(x)
        assert out.shape == (B, N, D)

    def test_residual_unchanged_dim(self):
        block = FlashTransformerBlock(D, H)
        x     = torch.randn(1, N, D)
        assert block(x).shape == x.shape

    def test_gradient_flows(self):
        block = FlashTransformerBlock(D, H)
        x     = torch.randn(B, N, D, requires_grad=True)
        out   = block(x)
        out.sum().backward()
        assert x.grad is not None
'''
write("tests/test_flash.py", src)
commit("test: add FlashTransformerBlock shape, residual dim, and gradient flow tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: memory analysis functions
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── Memory analysis ───────────────────────────────────────────────────────────

class TestMemoryAnalysis:
    def test_standard_larger_than_flash(self):
        std   = standard_attention_memory(1, 4, 512, 64)
        flash = flash_attention_memory(1, 4, 512, 64)
        assert std["total_bytes"] > flash["total_bytes"]

    def test_standard_scales_quadratically(self):
        s256 = standard_attention_memory(1, 1, 256, 64)
        s512 = standard_attention_memory(1, 1, 512, 64)
        # Attention matrix: (2N)² = 4× more memory
        ratio = s512["attn_matrix_bytes"] / s256["attn_matrix_bytes"]
        assert abs(ratio - 4.0) < 0.01

    def test_flash_scales_linearly(self):
        f256  = flash_attention_memory(1, 1, 256, 64)
        f512  = flash_attention_memory(1, 1, 512, 64)
        # QKV buffers scale linearly; tile is constant
        # output buffer 2× bigger for 2× seq_len
        assert f512["output_bytes"] == f256["output_bytes"] * 2

    def test_report_is_string(self):
        report = memory_comparison_report(1024)
        assert isinstance(report, str)
        assert "Flash" in report
'''
write("tests/test_flash.py", src)
commit("test: add memory analysis — std>flash, quadratic/linear scaling, report string tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: NanoMindFlash end-to-end
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_flash.py")
src += '''

# ── NanoMindFlash ─────────────────────────────────────────────────────────────

class TestNanoMindFlash:
    def _make(self, sdpa=True):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=32, block_size=N, d_model=D,
                          n_layers=2, n_heads=H, dropout=0.0)
        return NanoMindFlash(cfg, FlashConfig(use_torch_sdpa=sdpa))

    def test_forward_shape(self):
        model  = self._make()
        idx    = torch.randint(0, 32, (B, N))
        logits, loss = model(idx)
        assert logits.shape == (B, N, 32)
        assert loss is None

    def test_training_loss(self):
        model   = self._make()
        idx     = torch.randint(0, 32, (B, N))
        targets = torch.randint(0, 32, (B, N))
        _, loss = model(idx, targets)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        model   = self._make()
        idx     = torch.randint(0, 32, (B, N))
        targets = torch.randint(0, 32, (B, N))
        _, loss = model(idx, targets)
        loss.backward()
        for n, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {n}"

    def test_tiled_backend(self):
        model  = self._make(sdpa=False)
        idx    = torch.randint(0, 32, (B, N))
        logits, _ = model(idx)
        assert logits.shape == (B, N, 32)
        assert logits.isfinite().all()
'''
write("tests/test_flash.py", src)
commit("test: add NanoMindFlash forward, training loss, gradient flow, and tiled backend tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v2.2.0 + expose flash in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"2.1.0\"", "__version__ = \"2.2.0\"")
src = src.replace(
    "from nanomind.cache import KVCacheConfig, NanoMindCached, CachedGenerator, KVCacheManager",
    "from nanomind.cache import KVCacheConfig, NanoMindCached, CachedGenerator, KVCacheManager\n"
    "from nanomind.flash import FlashConfig, FlashAttention, NanoMindFlash"
)
src = src.replace(
    "    \"KVCacheManager\",\n    \"__version__\",\n]",
    "    \"KVCacheManager\",\n"
    "    \"FlashConfig\",\n"
    "    \"FlashAttention\",\n"
    "    \"NanoMindFlash\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v2.2.0 — expose FlashConfig, FlashAttention, NanoMindFlash in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Inference** | KV Cache — prefill + O(1) decode, CachedGenerator API |",
    "| **Inference** | KV Cache — prefill + O(1) decode, CachedGenerator API |\n"
    "| **Efficiency** | Flash Attention — O(N) memory tiled SDPA, SwiGLU FFN |"
)
readme = readme.replace(
    "**Total: 505 commits across 25 days.**",
    "**Total: 525 commits across 26 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [2.1.0] — 2024 — KV Cache for Fast Inference",
    "## [2.2.0] — 2024 — Flash Attention\n\n### Added\n"
    "- `NanoMindFlash` — transformer model using FlashAttention in every block\n"
    "- `FlashAttention` — drop-in SDPA with torch.sdpa + tiled fallback backends\n"
    "- `FlashTransformerBlock` — RMSNorm + FlashAttention + SwiGLU FFN block\n"
    "- `FlashConfig` — block_q, block_kv, causal, use_torch_sdpa\n"
    "- `tiled_flash_attention()` — pure-PyTorch O(N) memory reference implementation\n"
    "- `OnlineSoftmaxState` — streaming max/sum accumulator (core of Flash Attention)\n"
    "- `standard_attention_memory()` / `flash_attention_memory()` — memory analysis\n"
    "- `memory_comparison_report()` — N×N vs tile memory savings report\n"
    "- `examples/flash_attention_demo.py` — memory analysis and equivalence demo\n\n---\n\n"
    "## [2.1.0] — 2024 — KV Cache for Fast Inference"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v2.2.0, update README and CHANGELOG for Day 26 Flash Attention")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 26 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v2.2.0",
    "-m", "NanoMind v2.2.0 — Flash Attention", check=False)
r = run("git", "push", "origin", "v2.2.0", check=False)
print("Tag v2.2.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 26 COMPLETE — v2.2.0 TAGGED! ===")
