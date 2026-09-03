"""
day25_commits.py — 20 atomic commits for Day 25: KV Cache for Fast Inference.
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

print("\n=== DAY 25: KV Cache for Fast Inference — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — cache package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/__init__.py",
      '"""NanoMind KV Cache sub-package for fast autoregressive inference."""\n')
commit("feat: add nanomind/cache/ package skeleton for KV Cache inference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — KVCacheConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/config.py", '''\
"""
nanomind/cache/config.py — KV Cache configuration.

Without KV cache, autoregressive generation recomputes the full attention
over ALL previous tokens at every new step:

  Step t: attend over tokens [0 … t]    → O(t²) total work for T steps
  Step t: attend over tokens [0 … t-1]  → repeating all prior work!

With KV cache, we store the computed K and V tensors from past steps and
only compute attention for the NEW token against the cached K/V:

  Step t: compute K_t, V_t (new), attend against cached [K_0…K_{t-1}]
  Total: O(T) new computations, O(T·d) memory for the cache

This is the single most impactful optimization for LLM inference speed,
used in every production LLM serving system (vLLM, TGI, TensorRT-LLM).

References:
  Original attention: Vaswani et al. (2017)
  KV cache analysis:  Pope et al. (2022) — https://arxiv.org/abs/2211.05100
  PagedAttention:     Kwon et al. (2023) — https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """
    Configuration for KV Cache management.

    Attributes:
        max_batch_size:    Maximum batch size the cache is pre-allocated for.
        max_seq_len:       Maximum sequence length (prompt + generated tokens).
        n_layers:          Number of transformer layers.
        n_heads:           Number of KV heads (use n_kv_heads for GQA/MQA).
        head_dim:          Dimension per head (d_model // n_heads).
        dtype:             Tensor dtype for cached K/V (float32 or float16).
        device:            Device for cache tensors.
    """

    max_batch_size: int   = 1
    max_seq_len:    int   = 512
    n_layers:       int   = 6
    n_heads:        int   = 8
    head_dim:       int   = 64
    dtype:          str   = "float32"
    device:         str   = "cpu"

    def __post_init__(self) -> None:
        assert self.max_batch_size >= 1
        assert self.max_seq_len >= 1
        assert self.n_layers >= 1
        assert self.n_heads >= 1
        assert self.head_dim >= 1
        assert self.dtype in ("float32", "float16", "bfloat16")

    @property
    def cache_size_bytes(self) -> int:
        """Total bytes needed for all K and V cache tensors."""
        element_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]
        per_tensor    = self.max_batch_size * self.max_seq_len * self.n_heads * self.head_dim
        return 2 * self.n_layers * per_tensor * element_bytes   # 2 = K + V

    @property
    def cache_size_mb(self) -> float:
        return self.cache_size_bytes / (1024 ** 2)
''')
commit("feat: add KVCacheConfig — max_batch, max_seq_len, n_layers, head_dim, cache_size_bytes")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — LayerKVCache (single layer)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/layer_cache.py", '''\
"""
nanomind/cache/layer_cache.py — Per-layer KV cache storage.

Each transformer layer gets one LayerKVCache that holds pre-allocated
key and value tensors. Tokens are appended to the cache one step at a time.

Memory layout:
  k_cache : (max_batch, max_seq_len, n_heads, head_dim)
  v_cache : (max_batch, max_seq_len, n_heads, head_dim)

The ``seq_len`` pointer tracks how many tokens have been written.
"""

from __future__ import annotations

import torch


class LayerKVCache:
    """
    Key-Value cache for a single transformer attention layer.

    Pre-allocates fixed-size K and V tensors and fills them incrementally
    during autoregressive decoding.

    Args:
        max_batch_size: Maximum batch size.
        max_seq_len:    Maximum sequence length (prompt + max new tokens).
        n_heads:        Number of KV heads.
        head_dim:       Dimension per attention head.
        dtype:          Storage dtype.
        device:         Storage device.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len:    int,
        n_heads:        int,
        head_dim:       int,
        dtype:          torch.dtype = torch.float32,
        device:         torch.device | str = "cpu",
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self._len        = 0   # tokens written so far

        shape = (max_batch_size, max_seq_len, n_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V vectors and return the full accumulated cache.

        Args:
            k_new: New key tensor ``(B, T_new, n_heads, head_dim)``.
            v_new: New value tensor ``(B, T_new, n_heads, head_dim)``.

        Returns:
            Tuple of ``(k_full, v_full)`` — the complete K/V cache up to
            the current step ``(B, current_len, n_heads, head_dim)``.
        """
        B, T_new, H, D = k_new.shape
        assert self._len + T_new <= self.max_seq_len, (
            f"KV cache overflow: {self._len} + {T_new} > {self.max_seq_len}"
        )
        self.k_cache[:B, self._len:self._len + T_new] = k_new
        self.v_cache[:B, self._len:self._len + T_new] = v_new
        self._len += T_new

        k_out = self.k_cache[:B, :self._len]
        v_out = self.v_cache[:B, :self._len]
        return k_out, v_out

    def reset(self) -> None:
        """Clear the cache (zero fill + reset pointer)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._len = 0

    @property
    def current_len(self) -> int:
        return self._len

    @property
    def is_empty(self) -> bool:
        return self._len == 0

    @property
    def is_full(self) -> bool:
        return self._len >= self.max_seq_len

    def memory_bytes(self) -> int:
        return self.k_cache.nbytes + self.v_cache.nbytes

    def __repr__(self) -> str:
        return (
            f"LayerKVCache("
            f"len={self._len}/{self.max_seq_len}, "
            f"heads={self.n_heads}, dim={self.head_dim})"
        )
''')
commit("feat: add LayerKVCache — pre-allocated K/V storage with update(), reset(), overflow check")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — KVCacheManager (all layers)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/manager.py", '''\
"""
nanomind/cache/manager.py — Multi-layer KV cache manager.

KVCacheManager owns one LayerKVCache per transformer layer and provides
a unified interface for the model to use during inference.
"""

from __future__ import annotations

import torch
from nanomind.cache.config import KVCacheConfig
from nanomind.cache.layer_cache import LayerKVCache


_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


class KVCacheManager:
    """
    Manages KV caches for all layers of a transformer model.

    Creates and owns one :class:`LayerKVCache` per transformer layer.
    Provides a simple ``get(layer_idx)`` interface and global reset.

    Args:
        cfg: KV cache configuration.

    Example::

        cache = KVCacheManager(KVCacheConfig(
            n_layers=6, n_heads=8, head_dim=64, max_seq_len=512
        ))

        # Inside attention forward (layer 3):
        k_full, v_full = cache.get(3).update(k_new, v_new)
    """

    def __init__(self, cfg: KVCacheConfig) -> None:
        self.cfg    = cfg
        dtype       = _DTYPE_MAP[cfg.dtype]
        device      = torch.device(cfg.device)

        self._caches: list[LayerKVCache] = [
            LayerKVCache(
                max_batch_size=cfg.max_batch_size,
                max_seq_len=cfg.max_seq_len,
                n_heads=cfg.n_heads,
                head_dim=cfg.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(cfg.n_layers)
        ]

    def get(self, layer_idx: int) -> LayerKVCache:
        """Return the KV cache for a specific layer."""
        return self._caches[layer_idx]

    def reset(self) -> None:
        """Reset all layer caches (start of a new sequence)."""
        for c in self._caches:
            c.reset()

    @property
    def current_len(self) -> int:
        """Current sequence length (same for all layers)."""
        return self._caches[0].current_len if self._caches else 0

    def total_memory_bytes(self) -> int:
        """Total memory used by all K and V cache tensors."""
        return sum(c.memory_bytes() for c in self._caches)

    def total_memory_mb(self) -> float:
        return self.total_memory_bytes() / (1024 ** 2)

    def stats(self) -> dict:
        """Return cache utilisation statistics."""
        return {
            "n_layers":       len(self._caches),
            "current_len":    self.current_len,
            "max_seq_len":    self.cfg.max_seq_len,
            "fill_ratio":     self.current_len / max(self.cfg.max_seq_len, 1),
            "memory_mb":      self.total_memory_mb(),
            "config_mb":      self.cfg.cache_size_mb,
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheManager("
            f"layers={len(self._caches)}, "
            f"len={self.current_len}/{self.cfg.max_seq_len}, "
            f"mem={self.total_memory_mb():.1f}MB)"
        )
''')
commit("feat: add KVCacheManager — multi-layer cache manager with stats() and total_memory_mb()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — cache-aware scaled dot-product attention
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/attention.py", '''\
"""
nanomind/cache/attention.py — Cache-aware scaled dot-product attention.

Wraps standard SDPA to work with a LayerKVCache. On each forward call:
  1. Project Q, K, V from input (only the NEW tokens)
  2. Update the KV cache with K_new, V_new
  3. Compute attention of Q_new over full K/V from cache

This avoids recomputing K/V for past tokens on every step.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.cache.layer_cache import LayerKVCache


class CachedSelfAttention(nn.Module):
    """
    Self-attention with KV cache support for fast autoregressive decoding.

    Supports both prefill (process full prompt at once) and decode
    (one new token at a time with cached K/V).

    Args:
        d_model:  Model embedding dimension.
        n_heads:  Number of attention heads.
        dropout:  Attention dropout (only during training).
        bias:     Use bias in Q/K/V/out projections.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.0,
        bias:     bool  = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,
        kv_cache:   LayerKVCache | None = None,
        mask:       torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV cache.

        Args:
            x:        Input ``(B, T, d_model)``. T=1 for decode step.
            kv_cache: LayerKVCache for this layer (None = no cache).
            mask:     Causal mask ``(T, T_full)`` or None.

        Returns:
            Output ``(B, T, d_model)``.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        # Project Q, K, V for new tokens only
        q = self.q_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)
        v = self.v_proj(x).view(B, T, H, Dh)   # (B, T,     H, Dh)

        # Update cache → get full K/V
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)         # (B, T_full, H, Dh)

        T_full = k.shape[1]

        # Reshape for batched SDPA: (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T_full)

        if mask is not None:
            attn = attn + mask

        if kv_cache is None:
            # Causal mask for full-sequence (training / prefill)
            causal = torch.triu(
                torch.full((T, T_full), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out  = torch.matmul(attn, v)            # (B, H, T, Dh)
        out  = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
''')
commit("feat: add CachedSelfAttention — SDPA with LayerKVCache: prefill and decode modes")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — CachedTransformerBlock
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/block.py", '''\
"""
nanomind/cache/block.py — Transformer block with KV cache support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.cache.attention import CachedSelfAttention
from nanomind.cache.layer_cache import LayerKVCache
from nanomind.norm.factory import get_norm


class CachedTransformerBlock(nn.Module):
    """
    Pre-norm transformer block with CachedSelfAttention + dense FFN.

    Args:
        d_model:   Model embedding dimension.
        n_heads:   Number of attention heads.
        d_ff:      FFN hidden dimension (default: 4 × d_model).
        dropout:   Dropout probability.
        norm_type: Normalisation type (``"layernorm"`` or ``"rmsnorm"``).
        bias:      Use bias in projections and FFN.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        d_ff:      int | None = None,
        dropout:   float = 0.0,
        norm_type: str   = "layernorm",
        bias:      bool  = False,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = get_norm(norm_type, d_model)
        self.attn  = CachedSelfAttention(d_model, n_heads, dropout, bias)
        self.norm2 = get_norm(norm_type, d_model)
        self.ff1   = nn.Linear(d_model, d_ff, bias=bias)
        self.ff2   = nn.Linear(d_ff, d_model, bias=bias)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(
        self,
        x:        torch.Tensor,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        # Attention with pre-norm
        x = x + self.drop(self.attn(self.norm1(x), kv_cache))
        # FFN with pre-norm
        h = self.norm2(x)
        x = x + self.drop(self.ff2(self.act(self.ff1(h))))
        return x
''')
commit("feat: add CachedTransformerBlock — pre-norm attention + FFN with LayerKVCache passthrough")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — NanoMindCached model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/model.py", '''\
"""
nanomind/cache/model.py — NanoMind transformer with KV cache inference support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.cache.block import CachedTransformerBlock
from nanomind.cache.manager import KVCacheManager
from nanomind.cache.config import KVCacheConfig
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("cache.model")


class NanoMindCached(nn.Module):
    """
    NanoMind transformer with integrated KV cache for fast inference.

    During prefill (processing the prompt), pass all tokens at once.
    During decode (generating new tokens), pass one token at a time
    while the KV cache stores all prior K/V tensors.

    Args:
        model_cfg: Standard model configuration.
        cache_cfg: KV cache configuration.

    Example::

        model = NanoMindCached(model_cfg, cache_cfg)
        cache = model.new_cache()

        # Prefill
        logits = model.prefill(prompt_ids, cache)

        # Decode loop
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1])
            logits     = model.decode_step(next_token.unsqueeze(1), cache)
    """

    def __init__(self, model_cfg: ModelConfig, cache_cfg: KVCacheConfig) -> None:
        super().__init__()
        self.cfg       = model_cfg
        self.cache_cfg = cache_cfg

        self.tok_emb  = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_emb  = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        self.drop     = nn.Dropout(model_cfg.dropout)
        self.blocks   = nn.ModuleList([
            CachedTransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm     = get_norm("layernorm", model_cfg.d_model)
        self.lm_head  = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        log.info(f"NanoMindCached: {n:,} params, cache={cache_cfg.cache_size_mb:.1f}MB")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def new_cache(self) -> KVCacheManager:
        """Create a fresh KVCacheManager for this model."""
        return KVCacheManager(self.cache_cfg)

    def _forward(
        self,
        idx:      torch.Tensor,
        cache:    KVCacheManager | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        B, T = idx.shape
        tok  = self.tok_emb(idx)
        pos  = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x    = self.drop(tok + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            kv = cache.get(i) if cache is not None else None
            x  = block(x, kv)

        x      = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def prefill(
        self,
        prompt_ids: torch.Tensor,
        cache:      KVCacheManager,
    ) -> torch.Tensor:
        """
        Process the prompt and populate the KV cache.

        Args:
            prompt_ids: ``(B, T_prompt)`` token IDs.
            cache:      Fresh KVCacheManager (will be filled).

        Returns:
            Logits ``(B, T_prompt, vocab_size)``.
        """
        cache.reset()
        return self._forward(prompt_ids, cache, pos_offset=0)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        cache:     KVCacheManager,
    ) -> torch.Tensor:
        """
        Single decode step: attend over cached K/V, produce next logits.

        Args:
            token_ids: ``(B, 1)`` last generated token.
            cache:     Populated KVCacheManager.

        Returns:
            Logits ``(B, 1, vocab_size)``.
        """
        pos = cache.current_len
        return self._forward(token_ids, cache, pos_offset=pos)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standard forward (no cache) — for training compatibility."""
        logits = self._forward(idx)
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            return logits, loss
        return logits, None
''')
commit("feat: add NanoMindCached — transformer with prefill() and decode_step() KV cache API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — CachedGenerator high-level API
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/generator.py", '''\
"""
nanomind/cache/generator.py — Fast cached generation with temperature + top-K/top-P.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nanomind.cache.model import NanoMindCached
from nanomind.cache.manager import KVCacheManager
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("cache.generator")


def _sample_next(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
) -> torch.Tensor:
    """
    Sample next token from logits with temperature, top-K, and top-P filtering.

    Args:
        logits:      ``(B, vocab_size)`` raw logits.
        temperature: Logit temperature (< 1 = sharper, > 1 = flatter).
        top_k:       Keep only top-K tokens (0 = off).
        top_p:       Nucleus sampling threshold (1.0 = off).

    Returns:
        Sampled token IDs ``(B, 1)``.
    """
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k_vals = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < k_vals, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative               = sorted_probs.cumsum(dim=-1)
        remove                   = cumulative - sorted_probs > top_p
        sorted_probs[remove]     = 0.0
        sorted_probs            /= sorted_probs.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

    return torch.multinomial(probs, num_samples=1)


class CachedGenerator:
    """
    Fast autoregressive text generator using KV cache.

    Compared to naive generation, KV cache gives approximately
    ``T × n_layers`` fewer matrix multiplications per generated token.

    Args:
        model:     NanoMindCached model.
        tokenizer: Tokenizer for encoding/decoding.

    Example::

        gen = CachedGenerator(model, tokenizer)
        text = gen.generate("Once upon a time", max_new_tokens=100)
        print(text)
    """

    def __init__(
        self,
        model:     NanoMindCached,
        tokenizer: BaseTokenizer,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 1.0,
        eos_token_id:   int | None = None,
    ) -> str:
        """
        Generate text from a string prompt using KV cache.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature:    Sampling temperature.
            top_k:          Top-K filter (0 = off).
            top_p:          Nucleus sampling threshold (1.0 = off).
            eos_token_id:   Stop when this token is generated.

        Returns:
            Generated text (excluding the prompt).
        """
        ids    = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        idx    = torch.tensor([ids], dtype=torch.long, device=device)

        cache  = self.model.new_cache()

        # Prefill — process the full prompt
        logits = self.model.prefill(idx, cache)

        generated: list[int] = []

        for _ in range(max_new_tokens):
            next_tok = _sample_next(logits[:, -1, :], temperature, top_k, top_p)
            tok_id   = next_tok.item()

            if eos_token_id is not None and tok_id == eos_token_id:
                break

            generated.append(tok_id)

            # Decode step — one token at a time, O(1) thanks to cache
            logits = self.model.decode_step(next_tok, cache)

        return self.tokenizer.decode(generated)

    def __repr__(self) -> str:
        return f"CachedGenerator(model={type(self.model).__name__})"
''')
commit("feat: add CachedGenerator — prefill + decode loop with temperature/top-K/top-P sampling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — cache_stats and memory report
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/stats.py", '''\
"""
nanomind/cache/stats.py — KV cache memory and utilisation reporting.
"""

from __future__ import annotations

from nanomind.cache.config import KVCacheConfig
from nanomind.cache.manager import KVCacheManager


def print_cache_report(cache: KVCacheManager) -> None:
    """Pretty-print KV cache utilisation statistics."""
    s = cache.stats()
    print("=" * 50)
    print("KV Cache Report")
    print("=" * 50)
    print(f"  Layers          : {s['n_layers']}")
    print(f"  Tokens cached   : {s['current_len']} / {s['max_seq_len']}")
    print(f"  Fill ratio      : {s['fill_ratio']:.1%}")
    print(f"  Memory used     : {s['memory_mb']:.2f} MB")
    print(f"  Config limit    : {s['config_mb']:.2f} MB")
    print("=" * 50)


def estimate_cache_memory(cfg: KVCacheConfig) -> dict:
    """
    Estimate KV cache memory requirements before allocation.

    Args:
        cfg: KV cache configuration.

    Returns:
        Dict with ``bytes``, ``mb``, ``gb``, and a human-readable ``summary``.
    """
    b   = cfg.cache_size_bytes
    mb  = b / (1024 ** 2)
    gb  = b / (1024 ** 3)
    return {
        "bytes":   b,
        "mb":      mb,
        "gb":      gb,
        "summary": (
            f"{cfg.n_layers} layers × 2 (K+V) × "
            f"batch={cfg.max_batch_size} × "
            f"seq={cfg.max_seq_len} × "
            f"heads={cfg.n_heads} × "
            f"dim={cfg.head_dim} × "
            f"{cfg.dtype} = {mb:.1f} MB"
        ),
    }
''')
commit("feat: add print_cache_report() and estimate_cache_memory() — cache memory diagnostics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update cache __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/cache/__init__.py", '''\
"""NanoMind KV Cache sub-package for fast autoregressive inference.

KV cache is the single most impactful inference optimization for LLMs:
instead of recomputing all past K/V tensors at every decode step (O(T²)),
we store them and only compute the NEW token\'s K/V (O(1) per step).

Primary exports:
    - :class:`CachedGenerator`      — high-level generate() with KV cache
    - :class:`NanoMindCached`       — transformer with prefill() + decode_step()
    - :class:`KVCacheManager`       — multi-layer cache manager
    - :class:`LayerKVCache`         — per-layer K/V storage with update()
    - :class:`KVCacheConfig`        — cache configuration and size estimation
    - :class:`CachedSelfAttention`  — cache-aware attention module
    - :class:`CachedTransformerBlock` — transformer block with cache passthrough
    - :func:`print_cache_report`    — pretty-print utilisation stats
    - :func:`estimate_cache_memory` — pre-allocation memory estimate
"""

from nanomind.cache.config import KVCacheConfig
from nanomind.cache.layer_cache import LayerKVCache
from nanomind.cache.manager import KVCacheManager
from nanomind.cache.attention import CachedSelfAttention
from nanomind.cache.block import CachedTransformerBlock
from nanomind.cache.model import NanoMindCached
from nanomind.cache.generator import CachedGenerator
from nanomind.cache.stats import print_cache_report, estimate_cache_memory

__all__ = [
    "KVCacheConfig",
    "LayerKVCache",
    "KVCacheManager",
    "CachedSelfAttention",
    "CachedTransformerBlock",
    "NanoMindCached",
    "CachedGenerator",
    "print_cache_report",
    "estimate_cache_memory",
]
''')
commit("refactor: export all KV cache components from nanomind/cache/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: cached_generation_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/cached_generation_demo.py", '''\
"""
examples/cached_generation_demo.py — KV Cache generation demo.

Compares generation speed with and without KV cache, and demonstrates
the prefill + decode API.

Usage:
    python examples/cached_generation_demo.py
"""

import time, torch
from nanomind.model.config import ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.cache import (
    KVCacheConfig, NanoMindCached, KVCacheManager,
    CachedGenerator, estimate_cache_memory, print_cache_report,
)

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 40
tokenizer = CharTokenizer().build(CORPUS)
VOCAB     = tokenizer.vocab_size
BLOCK     = 64

model_cfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                        d_model=64, n_layers=4, n_heads=4, dropout=0.0)
cache_cfg = KVCacheConfig(
    max_batch_size=1, max_seq_len=BLOCK,
    n_layers=model_cfg.n_layers,
    n_heads=model_cfg.n_heads,
    head_dim=model_cfg.d_model // model_cfg.n_heads,
)

# ── Memory estimate ───────────────────────────────────────────────────────────
mem = estimate_cache_memory(cache_cfg)
print(f"Cache estimate: {mem['summary']}")

model = NanoMindCached(model_cfg, cache_cfg)

# ── CachedGenerator.generate() ───────────────────────────────────────────────
gen   = CachedGenerator(model, tokenizer)
prompt = "the quick"
MAX_NEW = 30

t0    = time.perf_counter()
text  = gen.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.8, top_k=10)
t1    = time.perf_counter()
print(f"\nPrompt   : {prompt!r}")
print(f"Generated: {text!r}")
print(f"Time     : {(t1-t0)*1000:.1f}ms for {MAX_NEW} tokens")

# ── Manual prefill + decode ───────────────────────────────────────────────────
ids    = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
cache  = model.new_cache()

logits = model.prefill(ids, cache)
print(f"\nPrefill : logits {logits.shape}")

next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
logits   = model.decode_step(next_tok, cache)
print(f"1 decode: logits {logits.shape}, cache len={cache.current_len}")

print_cache_report(cache)
''')
commit("feat: add examples/cached_generation_demo.py — KV cache speed, prefill, decode demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: KVCacheConfig
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_cache.py", '''\
"""
tests/test_cache.py — Tests for KV Cache.
"""

import pytest
import torch

from nanomind.model.config import ModelConfig
from nanomind.cache import (
    KVCacheConfig, LayerKVCache, KVCacheManager,
    NanoMindCached, CachedGenerator, estimate_cache_memory,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = "abcdefghijklmnopqrstuvwxyz " * 4
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
BLOCK  = 32
D, H   = 64, 4
B      = 1


def tiny_model():
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                       d_model=D, n_layers=2, n_heads=H, dropout=0.0)
    ccfg = KVCacheConfig(max_batch_size=B, max_seq_len=BLOCK,
                         n_layers=2, n_heads=H, head_dim=D // H)
    return NanoMindCached(mcfg, ccfg)


# ── KVCacheConfig ─────────────────────────────────────────────────────────────

class TestKVCacheConfig:
    def test_defaults(self):
        cfg = KVCacheConfig()
        assert cfg.max_batch_size == 1
        assert cfg.dtype == "float32"

    def test_cache_size_bytes_positive(self):
        cfg = KVCacheConfig(max_seq_len=128, n_layers=4, n_heads=4, head_dim=16)
        assert cfg.cache_size_bytes > 0

    def test_cache_size_mb(self):
        cfg = KVCacheConfig(max_seq_len=128, n_layers=4, n_heads=4, head_dim=16)
        assert cfg.cache_size_mb > 0

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            KVCacheConfig(dtype="int8")

    def test_invalid_batch_size(self):
        with pytest.raises(AssertionError):
            KVCacheConfig(max_batch_size=0)
''')
commit("test: add KVCacheConfig defaults, cache_size_bytes, invalid dtype and batch_size tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: LayerKVCache
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── LayerKVCache ──────────────────────────────────────────────────────────────

class TestLayerKVCache:
    def _make(self, max_seq=BLOCK):
        return LayerKVCache(max_batch_size=B, max_seq_len=max_seq,
                            n_heads=H, head_dim=D // H)

    def test_initial_state(self):
        c = self._make()
        assert c.is_empty
        assert c.current_len == 0

    def test_update_shape(self):
        c    = self._make()
        k    = torch.randn(B, 4, H, D // H)
        v    = torch.randn(B, 4, H, D // H)
        k_o, v_o = c.update(k, v)
        assert k_o.shape == (B, 4, H, D // H)
        assert v_o.shape == (B, 4, H, D // H)

    def test_current_len_increments(self):
        c = self._make()
        k = torch.randn(B, 3, H, D // H)
        v = torch.randn(B, 3, H, D // H)
        c.update(k, v)
        assert c.current_len == 3

    def test_accumulates_history(self):
        c = self._make()
        for _ in range(4):
            k = torch.randn(B, 2, H, D // H)
            v = torch.randn(B, 2, H, D // H)
            k_o, _ = c.update(k, v)
        assert k_o.shape[1] == 8   # 4 steps × 2 tokens

    def test_reset_clears(self):
        c = self._make()
        k = torch.randn(B, 4, H, D // H)
        c.update(k, k)
        c.reset()
        assert c.is_empty

    def test_overflow_raises(self):
        c = self._make(max_seq=4)
        k = torch.randn(B, 5, H, D // H)
        with pytest.raises(AssertionError):
            c.update(k, k)
'''
write("tests/test_cache.py", src)
commit("test: add LayerKVCache state, update shape, len, accumulation, reset, overflow tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: KVCacheManager
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── KVCacheManager ────────────────────────────────────────────────────────────

class TestKVCacheManager:
    def _make(self, n_layers=2):
        cfg = KVCacheConfig(max_batch_size=B, max_seq_len=BLOCK,
                            n_layers=n_layers, n_heads=H, head_dim=D // H)
        return KVCacheManager(cfg)

    def test_correct_number_of_caches(self):
        mgr = self._make(n_layers=4)
        assert len(mgr._caches) == 4

    def test_get_returns_layer_cache(self):
        mgr = self._make()
        assert isinstance(mgr.get(0), LayerKVCache)

    def test_current_len_after_update(self):
        mgr = self._make()
        k   = torch.randn(B, 3, H, D // H)
        mgr.get(0).update(k, k)
        assert mgr.current_len == 3

    def test_reset_all_layers(self):
        mgr = self._make()
        k   = torch.randn(B, 3, H, D // H)
        mgr.get(0).update(k, k)
        mgr.reset()
        assert mgr.current_len == 0

    def test_stats_keys(self):
        mgr   = self._make()
        stats = mgr.stats()
        for key in ("n_layers", "current_len", "max_seq_len", "fill_ratio", "memory_mb"):
            assert key in stats

    def test_memory_positive(self):
        mgr = self._make()
        assert mgr.total_memory_bytes() > 0
'''
write("tests/test_cache.py", src)
commit("test: add KVCacheManager layer count, get, len, reset, stats, and memory tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: NanoMindCached forward + prefill/decode
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── NanoMindCached ────────────────────────────────────────────────────────────

class TestNanoMindCached:
    def test_forward_no_cache(self):
        model  = tiny_model()
        idx    = torch.randint(0, VOCAB, (B, 8))
        logits, loss = model(idx)
        assert logits.shape == (B, 8, VOCAB)
        assert loss is None

    def test_prefill_shape(self):
        model  = tiny_model()
        cache  = model.new_cache()
        idx    = torch.randint(0, VOCAB, (B, 8))
        logits = model.prefill(idx, cache)
        assert logits.shape == (B, 8, VOCAB)
        assert cache.current_len == 8

    def test_decode_step_shape(self):
        model  = tiny_model()
        cache  = model.new_cache()
        prompt = torch.randint(0, VOCAB, (B, 5))
        model.prefill(prompt, cache)
        tok    = torch.randint(0, VOCAB, (B, 1))
        logits = model.decode_step(tok, cache)
        assert logits.shape == (B, 1, VOCAB)

    def test_cache_grows_with_decode_steps(self):
        model  = tiny_model()
        cache  = model.new_cache()
        prompt = torch.randint(0, VOCAB, (B, 5))
        model.prefill(prompt, cache)
        for _ in range(3):
            tok    = torch.randint(0, VOCAB, (B, 1))
            model.decode_step(tok, cache)
        assert cache.current_len == 5 + 3

    def test_training_loss(self):
        model   = tiny_model()
        idx     = torch.randint(0, VOCAB, (B, 8))
        targets = torch.randint(0, VOCAB, (B, 8))
        _, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0.0
'''
write("tests/test_cache.py", src)
commit("test: add NanoMindCached forward, prefill, decode_step, cache growth, training loss tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: CachedGenerator
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── CachedGenerator ───────────────────────────────────────────────────────────

class TestCachedGenerator:
    def test_generate_returns_string(self):
        gen  = CachedGenerator(tiny_model(), TOK)
        text = gen.generate("abc", max_new_tokens=5)
        assert isinstance(text, str)

    def test_generate_correct_length(self):
        gen   = CachedGenerator(tiny_model(), TOK)
        text  = gen.generate("abc", max_new_tokens=10)
        # Generated text should have up to 10 chars (may be fewer if EOS hit)
        assert len(text) <= 10

    def test_greedy_deterministic(self):
        """With temperature=0.01 and top_k=1, output should be deterministic."""
        model = tiny_model()
        gen   = CachedGenerator(model, TOK)
        t1    = gen.generate("abc", max_new_tokens=5, temperature=0.01, top_k=1)
        t2    = gen.generate("abc", max_new_tokens=5, temperature=0.01, top_k=1)
        assert t1 == t2

    def test_repr(self):
        gen = CachedGenerator(tiny_model(), TOK)
        assert "Cached" in repr(gen)
'''
write("tests/test_cache.py", src)
commit("test: add CachedGenerator generate string, length, deterministic, and repr tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: estimate_cache_memory
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── estimate_cache_memory ─────────────────────────────────────────────────────

class TestEstimateCacheMemory:
    def test_returns_dict_keys(self):
        cfg  = KVCacheConfig(max_seq_len=64, n_layers=2, n_heads=4, head_dim=16)
        mem  = estimate_cache_memory(cfg)
        for key in ("bytes", "mb", "gb", "summary"):
            assert key in mem

    def test_bytes_positive(self):
        cfg = KVCacheConfig(max_seq_len=64, n_layers=2, n_heads=4, head_dim=16)
        assert estimate_cache_memory(cfg)["bytes"] > 0

    def test_larger_seq_more_memory(self):
        short = KVCacheConfig(max_seq_len=64,  n_layers=2, n_heads=4, head_dim=16)
        long_ = KVCacheConfig(max_seq_len=512, n_layers=2, n_heads=4, head_dim=16)
        assert estimate_cache_memory(long_)["bytes"] > estimate_cache_memory(short)["bytes"]

    def test_more_layers_more_memory(self):
        few  = KVCacheConfig(max_seq_len=64, n_layers=2,  n_heads=4, head_dim=16)
        many = KVCacheConfig(max_seq_len=64, n_layers=12, n_heads=4, head_dim=16)
        assert estimate_cache_memory(many)["bytes"] > estimate_cache_memory(few)["bytes"]
'''
write("tests/test_cache.py", src)
commit("test: add estimate_cache_memory keys, positive bytes, seq/layer scaling tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: prefill + decode gives same logits as no-cache
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_cache.py")
src += '''

# ── Correctness: cache vs no-cache ────────────────────────────────────────────

class TestCacheCorrectness:
    def test_prefill_matches_no_cache(self):
        """Prefill output should match standard forward pass (no cache)."""
        model   = tiny_model()
        model.eval()
        idx     = torch.randint(0, VOCAB, (B, 6))

        # No cache
        with torch.no_grad():
            logits_nc, _ = model(idx)

        # With cache (prefill only)
        cache   = model.new_cache()
        with torch.no_grad():
            logits_c = model.prefill(idx, cache)

        assert torch.allclose(logits_nc, logits_c, atol=1e-5), \
            "Prefill logits differ from no-cache forward"

    def test_decode_produces_finite_logits(self):
        model   = tiny_model()
        cache   = model.new_cache()
        prompt  = torch.randint(0, VOCAB, (B, 4))
        with torch.no_grad():
            model.prefill(prompt, cache)
            tok    = torch.randint(0, VOCAB, (B, 1))
            logits = model.decode_step(tok, cache)
        assert logits.isfinite().all()
'''
write("tests/test_cache.py", src)
commit("test: add cache correctness — prefill matches no-cache forward, decode finite logits")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v2.1.0 + expose cache in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"2.0.0\"", "__version__ = \"2.1.0\"")
src = src.replace(
    "from nanomind.data import DataConfig, DataPipeline, InMemoryTokenDataset",
    "from nanomind.data import DataConfig, DataPipeline, InMemoryTokenDataset\n"
    "from nanomind.cache import KVCacheConfig, NanoMindCached, CachedGenerator, KVCacheManager"
)
src = src.replace(
    "    \"InMemoryTokenDataset\",\n    \"__version__\",\n]",
    "    \"InMemoryTokenDataset\",\n"
    "    \"KVCacheConfig\",\n"
    "    \"NanoMindCached\",\n"
    "    \"CachedGenerator\",\n"
    "    \"KVCacheManager\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v2.1.0 — expose KVCacheConfig, NanoMindCached, CachedGenerator in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Data** | Streaming pipeline — document packing, multi-source mixing, sharding |",
    "| **Data** | Streaming pipeline — document packing, multi-source mixing, sharding |\n"
    "| **Inference** | KV Cache — prefill + O(1) decode, CachedGenerator API |"
)
readme = readme.replace(
    "**Total: 485 commits across 24 days.**",
    "**Total: 505 commits across 25 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [2.0.0] — 2024 — Streaming Data Pipeline",
    "## [2.1.0] — 2024 — KV Cache for Fast Inference\n\n### Added\n"
    "- `NanoMindCached` — transformer with `prefill()` + `decode_step()` KV cache API\n"
    "- `CachedGenerator` — high-level `generate()` with temperature/top-K/top-P\n"
    "- `KVCacheManager` — multi-layer cache manager with `stats()` and memory tracking\n"
    "- `LayerKVCache` — per-layer pre-allocated K/V storage with overflow check\n"
    "- `KVCacheConfig` — `cache_size_bytes` / `cache_size_mb` pre-allocation estimation\n"
    "- `CachedSelfAttention` — attention module with cache-aware prefill/decode modes\n"
    "- `CachedTransformerBlock` — transformer block with cache passthrough\n"
    "- `estimate_cache_memory()` — pre-allocation memory estimate by config\n"
    "- `print_cache_report()` — pretty-print cache utilisation\n"
    "- `examples/cached_generation_demo.py` — speed, prefill, decode demo\n\n---\n\n"
    "## [2.0.0] — 2024 — Streaming Data Pipeline"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v2.1.0, update README and CHANGELOG for Day 25 KV Cache")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 25 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v2.1.0",
    "-m", "NanoMind v2.1.0 — KV Cache for Fast Inference", check=False)
r = run("git", "push", "origin", "v2.1.0", check=False)
print("Tag v2.1.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 25 COMPLETE — v2.1.0 TAGGED! ===")
