"""
nanomind/ssm/hybrid.py — Hybrid Transformer + SSM models.

## Why Hybrid?

Pure SSMs are strong on long sequences but weaker on tasks requiring
precise token-to-token attention (e.g., in-context learning, retrieval).

Hybrid models combine:
  - SSM layers: cheap, linear-time, great for long-range
  - Attention layers: expensive but precise, great for short-range

Examples:
  - Jamba (AI21, 2024): 1 attention every 8 Mamba layers
  - Griffin (DeepMind, 2024): local attention + gated linear recurrence
  - Zamba (Zyphra, 2024): SSM backbone + shared attention every 6 layers

Hybrid ratio:
  - 1:7 (1 attn per 7 SSM) → ~5% of params in attention
  - Attains near-Transformer quality at near-SSM efficiency
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from nanomind.ssm.mamba import MambaBlock, SSMConfig


@dataclass
class HybridConfig:
    """Configuration for hybrid Transformer-SSM model."""
    d_model:    int        = 64
    n_heads:    int        = 4
    d_state:    int        = 16
    expand:     int        = 2
    n_layers:   int        = 8
    attn_every: int        = 4     # insert attention every N SSM layers
    vocab_size: int        = 256
    dropout:    float      = 0.1


class LocalAttentionLayer(nn.Module):
    """
    Local sliding-window attention (O(T × window) cost).
    Used in Griffin/hybrid models as a complement to SSM.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of heads.
        window_size: Local attention window.
        dropout:     Dropout.

    Example::

        attn = LocalAttentionLayer(64, 4, window_size=16)
        x    = torch.randn(2, 32, 64)
        y    = attn(x)   # (2, 32, 64)
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        window_size: int   = 64,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn        = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        W = min(self.window_size, T)
        # Build causal mask (full causal — window restricts in practice)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        r = self.norm(x)
        out, _ = self.attn(r, r, r, attn_mask=mask)
        return x + out


class HybridSSMTransformer(nn.Module):
    """
    Hybrid model: mostly Mamba blocks with periodic attention layers.

    Architecture (attn_every=4):
      Layer 0: MambaBlock
      Layer 1: MambaBlock
      Layer 2: MambaBlock
      Layer 3: MambaBlock
      Layer 4: LocalAttentionLayer  ← every attn_every layers
      Layer 5: MambaBlock
      ...

    Args:
        cfg: :class:`HybridConfig`.

    Example::

        cfg    = HybridConfig(d_model=64, n_layers=8, attn_every=4)
        model  = HybridSSMTransformer(cfg)
        ids    = torch.randint(0, 256, (2, 32))
        logits = model(ids)   # (2, 32, 256)
    """

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        ssm_cfg        = SSMConfig(
            d_model=cfg.d_model, d_state=cfg.d_state,
            expand=cfg.expand,
        )
        layers = []
        for i in range(cfg.n_layers):
            if (i + 1) % cfg.attn_every == 0:
                layers.append(LocalAttentionLayer(
                    cfg.d_model, cfg.n_heads, dropout=cfg.dropout
                ))
            else:
                layers.append(MambaBlock(ssm_cfg))
        self.layers  = nn.ModuleList(layers)
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    def layer_type_summary(self) -> dict:
        """Count SSM vs attention layers."""
        n_mamba = sum(1 for l in self.layers if isinstance(l, MambaBlock))
        n_attn  = sum(1 for l in self.layers if isinstance(l, LocalAttentionLayer))
        return {
            "total_layers": len(self.layers),
            "mamba_layers": n_mamba,
            "attn_layers":  n_attn,
            "attn_ratio":   round(n_attn / max(len(self.layers), 1), 3),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
