"""
nanomind/cache/cached_attention.py — Attention with KV-Cache support.

Wraps a standard attention layer to use a KVCache during inference,
enabling O(1) per-step computation instead of O(T) recomputation.

Without cache:  Q @ K^T needs ALL T tokens' keys every step
With cache:     K_new computed only for the 1 new token,
                then concatenated with cached K[0:T-1]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.cache.kv_cache import KVCache


class CachedAttention(nn.Module):
    """
    Multi-head attention with KV-Cache support.

    Compatible drop-in for standard attention during inference.

    Args:
        d_model:   Model dimension.
        n_heads:   Number of attention heads.
        dropout:   Attention dropout (disabled during inference).
        layer_id:  Which layer this attention belongs to (for cache indexing).

    Usage::

        attn  = CachedAttention(d_model=256, n_heads=4, layer_id=0)
        cache = KVCache(CacheConfig(n_layers=4, n_heads=4, d_head=64))

        # Prefill (process prompt)
        out = attn(x_prompt, cache=cache)

        # Decode (generate one token at a time)
        for _ in range(max_new_tokens):
            out = attn(x_new_token, cache=cache)
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.0,
        layer_id: int   = 0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.layer_id = layer_id
        self.scale    = self.d_head ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, D/H)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x:     torch.Tensor,
        cache: KVCache | None = None,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     ``(B, T, D)`` input (T=1 during decode, T=prompt_len during prefill).
            cache: Optional :class:`KVCache` for fast incremental generation.
            mask:  Optional causal attention mask.

        Returns:
            ``(B, T, D)`` attention output.
        """
        B, T, D = x.shape
        q = self._split_heads(self.q_proj(x))                 # (B, H, T, Dh)
        k = self._split_heads(self.k_proj(x))                 # (B, H, T, Dh)
        v = self._split_heads(self.v_proj(x))                 # (B, H, T, Dh)

        if cache is not None:
            cache.append(self.layer_id, k, v)
            k, v = cache.get(self.layer_id)                   # includes history

        # Scaled dot-product attention
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T_c)

        if mask is not None:
            attn_w = attn_w + mask

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.drop(attn_w)
        out    = torch.matmul(attn_w, v)                      # (B, H, T, Dh)
        out    = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
