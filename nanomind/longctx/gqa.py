"""
nanomind/longctx/gqa.py — Grouped Query Attention (GQA) and Multi-Query Attention (MQA).

## The KV Cache Bottleneck

During autoregressive generation, each step adds one new (K, V) pair per head.
With H heads, sequence length T, and batch B:
  KV cache = 2 × B × H × T × D_head bytes

For LLaMA-65B: 2 × 40 heads × 4096 × 128 bytes × batch_size — enormous!

## Multi-Query Attention (MQA, Shazeer 2019)

Share a single K, V across all Q heads:
  - H query heads (each with its own Q projection)
  - 1 key head  (shared across all Q heads)
  - 1 value head (shared across all Q heads)

KV cache reduction: H× smaller!
Quality: slightly worse than MHA, but faster inference.

Used by: PaLM, Falcon, StarCoder.

## Grouped Query Attention (GQA, Ainslie et al., 2023)

Compromise between MHA and MQA:
  - H query heads split into G groups
  - Each group shares 1 K, V head
  - G KV heads total

KV cache reduction: H/G× smaller!
Quality: nearly identical to MHA.

Used by: LLaMA-2 70B, Mistral, Gemma, Command-R.

Reference:
  Shazeer (2019) MQA: https://arxiv.org/abs/1911.02150
  Ainslie et al. (2023) GQA: https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Generalises MHA (n_kv_heads=n_heads) and MQA (n_kv_heads=1).

    Args:
        d_model:     Model dimension.
        n_heads:     Number of query heads (H).
        n_kv_heads:  Number of KV heads (G). Must divide H evenly.
        max_seq:     Maximum sequence length.
        dropout:     Attention dropout.
        use_rope:    Apply RoPE position encoding.

    Example::

        # LLaMA-2 70B: 64 Q heads, 8 KV heads
        gqa = GroupedQueryAttention(d_model=8192, n_heads=64, n_kv_heads=8)
        out, kv = gqa(x)   # x: (B, T, D)
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int   = 1,
        max_seq:    int   = 4096,
        dropout:    float = 0.0,
        use_rope:   bool  = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads  % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_groups    = n_heads // n_kv_heads
        self.d_head      = d_model // n_heads
        self.scale       = self.d_head ** -0.5
        self.use_rope    = use_rope

        self.q_proj = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        if use_rope:
            from nanomind.longctx.rope import RotaryEmbedding
            self.rope = RotaryEmbedding(self.d_head, max_seq=max_seq)

    def forward(
        self,
        x:       torch.Tensor,
        past_kv: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        GQA forward pass.

        Args:
            x:       ``(B, T, D)`` input.
            past_kv: Optional ``(k_cache, v_cache)`` for generation.

        Returns:
            ``(output, (k, v))``
        """
        B, T, D = x.shape
        H, G    = self.n_heads, self.n_groups
        Hkv, Dh = self.n_kv_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)     # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)   # (B,Hkv,T,Dh)
        V = self.v_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            K = torch.cat([pk, K], dim=2)
            V = torch.cat([pv, V], dim=2)

        T_kv = K.shape[2]
        offset = T_kv - T

        if self.use_rope:
            Q, K = self.rope.apply(Q, K, seq_len=T, offset=offset)

        # Expand KV to match Q heads: repeat each KV head n_groups times
        K_exp = K.repeat_interleave(G, dim=1)    # (B, H, T_kv, Dh)
        V_exp = V.repeat_interleave(G, dim=1)

        scores  = (Q @ K_exp.transpose(-2, -1)) * self.scale   # (B,H,T,T_kv)
        # Causal mask
        mask    = torch.triu(torch.ones(T, T_kv, dtype=torch.bool), diagonal=1 + offset)
        scores  = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out     = (weights @ V_exp).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), (K, V)

    @property
    def kv_cache_factor(self) -> float:
        """KV cache size relative to MHA (n_kv_heads / n_heads)."""
        return self.n_kv_heads / self.n_heads

    def to_dict(self) -> dict:
        return {
            "n_heads":     self.n_heads,
            "n_kv_heads":  self.n_kv_heads,
            "n_groups":    self.n_groups,
            "kv_reduction": f"{self.n_heads//self.n_kv_heads}x",
        }
