"""
nanomind/longctx/chunked.py — Chunked / Tiled Attention (FlashAttention-style).

## The Memory Problem with Standard Attention

Standard attention materialises the full N×N attention matrix:
  scores = QK^T / √d    ← (B, H, T, T) tensor — HUGE for large T!

For T=16384, H=32, B=1: 16384 × 16384 × 32 × 4 bytes = 32 GB just for scores!

## FlashAttention (Dao et al., 2022)

FlashAttention avoids materialising the full matrix using:
  1. Tiling: split Q, K, V into blocks that fit in SRAM
  2. Online softmax: compute numerically stable softmax incrementally
  3. IO-awareness: minimise HBM reads/writes

Memory: O(T) instead of O(T²)
Speed: 2-4× faster than standard PyTorch attention

This module implements the pure-Python version (FlashAttention algorithm)
without CUDA kernels. For production, use flash-attn library.

## Online Softmax (Milakov & Gimelshein, 2018)

The key mathematical identity:
  softmax([x1, x2]) = softmax([x1, x2]) regardless of split

Online update rule for accumulating attention outputs in chunks:
  m_new = max(m_old, max(block_scores))
  l_new = e^{m_old - m_new} × l_old + Σ e^{scores - m_new}
  O_new = (e^{m_old - m_new} × l_old × O_old + block_contribution) / l_new

References:
  Dao et al. (2022) FlashAttention: https://arxiv.org/abs/2205.14135
  Dao et al. (2023) FlashAttention-2: https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedAttention(nn.Module):
    """
    Chunked attention: compute attention in tiles to save memory.

    This is a pure-Python reference implementation of FlashAttention's
    tiling algorithm. Reduces peak memory from O(T²) to O(T × chunk).

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads.
        chunk_size: Tile/chunk size (smaller = less memory, more overhead).
        causal:     Apply causal masking.

    Example::

        ca  = ChunkedAttention(d_model=256, n_heads=8, chunk_size=64)
        out = ca(x)   # x: (B, T, D)  — O(T × chunk) peak memory
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        chunk_size: int  = 64,
        causal:     bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.chunk_size = chunk_size
        self.causal     = causal
        self.scale      = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunked attention forward.

        Args:
            x: ``(B, T, D)`` input.

        Returns:
            ``(B, T, D)`` output.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)

        out = self._chunked_attention(Q, K, V)                      # (B,H,T,Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.o_proj(out)

    def _chunked_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Online softmax chunked attention.

        Processes Q in chunks, accumulating the output with online softmax.
        """
        B, H, T, Dh = Q.shape
        C           = self.chunk_size
        O           = torch.zeros_like(Q)           # output accumulator
        l           = torch.zeros(B, H, T, 1)       # log-sum-exp normaliser
        m           = torch.full((B, H, T, 1), float("-inf"))  # running max

        # Process K/V in chunks
        for kv_start in range(0, T, C):
            kv_end = min(kv_start + C, T)
            K_c    = K[:, :, kv_start:kv_end, :]    # (B,H,C,Dh)
            V_c    = V[:, :, kv_start:kv_end, :]

            # Scores for all Q against this K chunk
            scores = (Q @ K_c.transpose(-2, -1)) * self.scale  # (B,H,T,C)

            # Causal mask: Q position i cannot attend to K position j > i
            if self.causal:
                q_idx  = torch.arange(T).unsqueeze(-1)              # (T,1)
                kv_idx = torch.arange(kv_start, kv_end).unsqueeze(0)# (1,C)
                mask   = kv_idx > q_idx
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            exp_scores = torch.exp(scores - m_new)                  # (B,H,T,C)
            l_new = torch.exp(m - m_new) * l + exp_scores.sum(-1, keepdim=True)
            O     = (torch.exp(m - m_new) * l * O + exp_scores @ V_c) / (l_new + 1e-9)
            m     = m_new
            l     = l_new

        return O

    @property
    def peak_memory_ratio(self) -> float:
        """Peak memory relative to standard attention (chunk/T)."""
        return self.chunk_size  # lower is better
