"""
nanomind/multimodal/vision_encoder.py — Vision encoder: patches → visual tokens.

Implements a lightweight Vision Transformer (ViT) encoder that maps
image patches to visual token embeddings compatible with the LLM.

Architecture:
  Image → Normalise → Patchify → Linear(patch_dim, vision_dim)
       → Positional Embedding → Transformer Layers → Visual Tokens

This is the vision half of LLaVA / InstructBLIP / Qwen-VL:
  - LLaVA uses CLIP ViT-L/14 (307M params) as the vision encoder
  - NanoMind implements a miniature ViT for educational clarity
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.patchify import patchify, ImageNormalizer


class ViTBlock(nn.Module):
    """
    Single Vision Transformer block: LayerNorm → MHA → FFN.

    Args:
        d_model:  Hidden dimension.
        n_heads:  Number of attention heads.
        mlp_ratio: FFN expansion ratio (default: 4×).
        dropout:  Attention and FFN dropout.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff       = int(d_model * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h  = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x  = x + h
        # Pre-norm FFN
        x  = x + self.ffn(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    Lightweight Vision Transformer encoder for NanoMind.

    Converts images to visual token sequences compatible with the LLM.

    Args:
        cfg:       Modality configuration.
        n_layers:  Number of ViT transformer layers.
        n_heads:   Number of attention heads in vision transformer.

    Example::

        encoder = VisionEncoder(ModalityConfig())
        image   = torch.randn(1, 3, 224, 224)   # batch of 1 image
        tokens  = encoder(image)
        # → (1, 196, vision_dim)  visual token sequence
    """

    def __init__(
        self,
        cfg:      ModalityConfig,
        n_layers: int = 2,
        n_heads:  int = 4,
    ) -> None:
        super().__init__()
        self.cfg        = cfg
        self.normalizer = ImageNormalizer(cfg.normalize_mean, cfg.normalize_std)
        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.vision_dim)
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, cfg.n_patches, cfg.vision_dim)
        )
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, cfg.vision_dim))
        self.layers     = nn.ModuleList([
            ViTBlock(cfg.vision_dim, n_heads) for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(cfg.vision_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch to visual tokens.

        Args:
            image: ``(B, 3, H, W)`` image tensor (float32, pixels in [0, 1]).

        Returns:
            ``(B, N, vision_dim)`` visual token tensor.
        """
        B = image.shape[0]
        x = self.normalizer(image)
        x = patchify(x, self.cfg.patch_size)          # (B, N, patch_dim)
        x = self.patch_proj(x)                         # (B, N, vision_dim)
        x = x + self.pos_embed                         # add positional embedding
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)               # (B, N+1, vision_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:, :]   # remove CLS, return patch tokens (B, N, vision_dim)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
