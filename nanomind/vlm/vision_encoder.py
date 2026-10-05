"""
nanomind/vlm/vision_encoder.py — Vision Transformer (ViT) image encoder.

## Vision Transformer (Dosovitskiy et al., 2020)

Instead of convolutions, ViT treats an image as a sequence of patches:
  1. Divide 224×224 image into 16×16 patches → 196 patches
  2. Flatten each patch: 16×16×3 = 768 values
  3. Project to embedding dim D via linear layer
  4. Prepend [CLS] token (used as image representation)
  5. Add 2D positional embeddings
  6. Pass through Transformer encoder
  7. Use [CLS] output as image embedding

Patch sizes:
  ViT-B/16:  16×16 patches, 12 layers, D=768  → 196 patches
  ViT-B/32:  32×32 patches, 12 layers, D=768  → 49 patches (faster)
  ViT-L/14:  14×14 patches, 24 layers, D=1024 → 256 patches (CLIP quality)

## CLIP Vision Encoder

CLIP (Contrastive Language-Image Pre-Training) uses a ViT backbone
trained with contrastive loss to align image and text embeddings.

Input:  224×224×3 image
Output: 512-dim or 768-dim image embedding

Reference:
  Dosovitskiy et al. (2020) "An Image is Worth 16×16 Words"
  https://arxiv.org/abs/2010.11929
  Radford et al. (2021) CLIP: https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """Configuration for the ViT vision encoder."""
    image_size:   int   = 224
    patch_size:   int   = 16
    in_channels:  int   = 3
    d_model:      int   = 64    # embedding dim (768 in ViT-B)
    n_layers:     int   = 4     # transformer layers (12 in ViT-B)
    n_heads:      int   = 4     # attention heads (12 in ViT-B)
    d_ff:         int   = 128   # FFN dim (3072 in ViT-B)
    dropout:      float = 0.1
    d_embed:      int   = 32    # final projection dim for CLIP

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.in_channels

    @property
    def seq_len(self) -> int:
        return self.n_patches + 1  # +1 for CLS token


class PatchEmbedding(nn.Module):
    """
    Split image into patches and project to embedding dimension.

    Args:
        cfg: :class:`VisionEncoderConfig`.

    Example::

        embed = PatchEmbedding(cfg)
        x     = torch.randn(2, 3, 224, 224)
        out   = embed(x)   # (2, 197, d_model) — 196 patches + CLS
    """

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.cfg        = cfg
        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, cfg.seq_len, cfg.d_model)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Split (B, C, H, W) into (B, N, patch_dim) patches."""
        B, C, H, W = x.shape
        P = self.cfg.patch_size
        # Reshape into patches
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, H/P, W/P, C, P, P)
        x = x.reshape(B, -1, C * P * P)   # (B, N, patch_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed image patches.

        Args:
            x: Image tensor ``(B, C, H, W)``.

        Returns:
            ``(B, N+1, d_model)`` — patches + CLS token with pos embeddings.
        """
        B = x.shape[0]
        patches = self._patchify(x)                  # (B, N, patch_dim)
        tokens  = self.patch_proj(patches)            # (B, N, d_model)
        cls     = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens  = torch.cat([cls, tokens], dim=1)     # (B, N+1, d_model)
        return tokens + self.pos_embed


class VisionTransformerBlock(nn.Module):
    """Single ViT transformer block (Pre-LN style)."""

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    ViT-style vision encoder for VLM.

    Encodes images into patch sequences and a global [CLS] embedding.

    Args:
        cfg: :class:`VisionEncoderConfig`.

    Example::

        cfg     = VisionEncoderConfig(image_size=32, patch_size=8,
                                       d_model=64, n_layers=2, d_embed=32)
        encoder = VisionEncoder(cfg)
        imgs    = torch.randn(2, 3, 32, 32)
        cls_emb, patch_embs = encoder(imgs)
        # cls_emb:    (2, 32)  — global image embedding (for CLIP)
        # patch_embs: (2, 17, 32) — per-patch embeddings (for VQA)
    """

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.patch_emb = PatchEmbedding(cfg)
        self.layers    = nn.ModuleList([
            VisionTransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.proj      = nn.Linear(cfg.d_model, cfg.d_embed)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images.

        Args:
            x: ``(B, C, H, W)`` images.

        Returns:
            Tuple of:
              - cls_embedding: ``(B, d_embed)`` — global image vector
              - patch_embeddings: ``(B, N+1, d_embed)`` — all tokens projected
        """
        tokens = self.patch_emb(x)     # (B, N+1, d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)     # (B, N+1, d_model)
        proj   = self.proj(tokens)     # (B, N+1, d_embed)

        cls_embedding   = proj[:, 0, :]    # (B, d_embed) — CLS token
        patch_embeddings = proj             # (B, N+1, d_embed)
        return cls_embedding, patch_embeddings
