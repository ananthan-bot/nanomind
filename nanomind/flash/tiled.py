"""
nanomind/flash/tiled.py — Pure-Python tiled Flash Attention (reference implementation).

This is a pedagogical reference implementation of the Flash Attention algorithm
in pure PyTorch. It is mathematically identical to standard scaled dot-product
attention but processes K/V in tiles to demonstrate the O(N) memory property.

Performance note:
  This Python-level tiling is NOT faster than standard attention on GPU —
  it lacks the CUDA kernel fusion and SRAM management of the real Flash Attention.
  For GPU performance, use FlashConfig(use_torch_sdpa=True) which delegates to
  PyTorch's built-in flash-efficient SDPA.

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
