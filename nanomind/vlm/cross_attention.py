"""
nanomind/vlm/cross_attention.py — Cross-modal attention for VLMs.

## Cross-Attention in VLMs

After encoding images with ViT, we need to fuse visual and language features.

Two main architectures:

1. Early Fusion (LLaVA style):
   - Concatenate image patch tokens + text tokens
   - Feed concatenated sequence to LLM
   - Image tokens act like extra "visual tokens" in the context
   - Simple but requires fine-tuning the full LLM

2. Cross-Attention Fusion (Flamingo style):
   - Keep image and text in separate streams
   - Add cross-attention layers to LLM: text attends to image features
   - More efficient: can freeze the LLM
   - Better for long image sequences

## LLaVA Architecture (Visual Instruction Tuning)

  image → ViT encoder → MLP projector → visual tokens
  question → tokenizer → text tokens
  [visual tokens] + [text tokens] → LLaMA → answer

Very simple, very effective. LLaVA-1.5 beats GPT-4V on many benchmarks!

## Flamingo Architecture (DeepMind, 2022)

  image → NFNet vision encoder → Perceiver Resampler → 64 visual tokens
  [cross-attention layers inserted into frozen LM every K layers]
  → few-shot multimodal QA

References:
  Liu et al. (2023) LLaVA: https://arxiv.org/abs/2304.08485
  Alayrac et al. (2022) Flamingo: https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-modal attention."""
    d_model:   int   = 64    # LM hidden dim
    d_visual:  int   = 32    # vision encoder output dim
    n_heads:   int   = 4
    dropout:   float = 0.1
    n_visual_tokens: int = 64  # Perceiver: compress to this many tokens


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: compress variable-length image patches to
    fixed-size visual token sequence (Flamingo-style).

    Args:
        cfg: :class:`CrossAttentionConfig`.

    Example::

        resampler = PerceiverResampler(cfg)
        patches   = torch.randn(2, 197, 32)   # ViT output
        visual    = resampler(patches)          # (2, 64, 32)
    """

    def __init__(self, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        # Learnable latent visual tokens (the "queries" in Perceiver)
        self.latents = nn.Parameter(
            torch.randn(1, cfg.n_visual_tokens, cfg.d_visual)
        )
        self.attn = nn.MultiheadAttention(
            cfg.d_visual, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(cfg.d_visual)
        self.norm_kv = nn.LayerNorm(cfg.d_visual)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_visual, cfg.d_visual * 4),
            nn.GELU(),
            nn.Linear(cfg.d_visual * 4, cfg.d_visual),
        )
        self.norm_ff = nn.LayerNorm(cfg.d_visual)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress image tokens to fixed-size visual tokens.

        Args:
            image_tokens: ``(B, N, d_visual)`` from vision encoder.

        Returns:
            ``(B, n_visual_tokens, d_visual)`` compressed visual tokens.
        """
        B    = image_tokens.shape[0]
        q    = self.latents.expand(B, -1, -1)   # (B, n_vis, d_visual)
        kv   = image_tokens

        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(q_n, kv_n, kv_n)
        q    = q + out
        q    = q + self.ff(self.norm_ff(q))
        return q


class VisualProjector(nn.Module):
    """
    Linear MLP projector from vision dim → LM dim (LLaVA-style).

    Projects ViT output to LM's token embedding space so visual
    tokens can be concatenated with text tokens.

    Args:
        d_visual: Vision encoder output dim.
        d_model:  LM embedding dim.
        n_layers: Number of MLP layers (LLaVA uses 2).

    Example::

        proj   = VisualProjector(d_visual=32, d_model=64, n_layers=2)
        visual = torch.randn(2, 17, 32)   # patch embeddings
        lm_vis = proj(visual)              # (2, 17, 64) — same dim as LM
    """

    def __init__(self, d_visual: int, d_model: int, n_layers: int = 2) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim  = d_visual if i == 0 else d_model
            layers.append(nn.Linear(in_dim, d_model))
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Project visual tokens to LM embedding space."""
        return self.proj(visual_tokens)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: text tokens attend to visual tokens.

    Used in Flamingo-style architectures where cross-attention layers
    are inserted into the LM at regular intervals.

    Args:
        cfg: :class:`CrossAttentionConfig`.

    Example::

        xattn   = CrossModalAttention(cfg)
        text_h  = torch.randn(2, 12, 64)   # LM hidden states
        visual  = torch.randn(2, 64, 32)   # visual tokens
        out     = xattn(text_h, visual)     # (2, 12, 64)
    """

    def __init__(self, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Project visual to d_model for cross-attention keys/values
        self.visual_proj = nn.Linear(cfg.d_visual, cfg.d_model)
        self.cross_attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm_q  = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)
        self.gate    = nn.Parameter(torch.zeros(1))   # gating (tanh)

    def forward(
        self,
        text_hidden:   torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attend text hidden states to visual tokens.

        Args:
            text_hidden:   ``(B, T, d_model)`` LM hidden states.
            visual_tokens: ``(B, N, d_visual)`` visual token sequence.

        Returns:
            ``(B, T, d_model)`` updated text hidden states.
        """
        # Project visual to d_model
        v    = self.visual_proj(visual_tokens)    # (B, N, d_model)
        q    = self.norm_q(text_hidden)
        kv   = self.norm_kv(v)
        out, _ = self.cross_attn(q, kv, kv)
        # Gated residual connection (Flamingo style)
        return text_hidden + self.gate.tanh() * out
